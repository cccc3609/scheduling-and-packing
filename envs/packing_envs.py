"""
envs/packing_envs.py

Patch 2 boundary:
  - this environment owns only production instance and nesting state;
  - terminal scheduling evaluation is performed by integration wrappers.

解耦接口：
  get_part_feats()  → [max_capacity, 5]  供 PartEncoder
  get_state_feat()  → [57]               供 StepDecoder
"""

import gymnasium as gym
import numpy as np
import copy
import math
from gymnasium import spaces

from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import MAX_PARTS_CAPACITY, TRAIN_CONFIG, MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, FEATURE_CONFIG
from core.instance import ProductionInstance, generate_instance
from core.processing import parts_cutting_time, plate_processing_time
from core.scheduling_problem import NestingTerminalResult, NestedPlateResult


class NestingSchedulingEnv(gym.Env):

    PART_FEAT_DIM  = 5
    STATE_FEAT_DIM = 57

    def __init__(self, plate_size=(200, 200)):
        super().__init__()

        self.max_capacity      = MAX_PARTS_CAPACITY
        self.max_sched_capacity = MAX_SCHED_TASKS_CAPACITY
        self.current_num_parts  = TRAIN_CONFIG['min_parts']

        self.CUTTING_SPEED = COST_CONFIG['cutting_speed']
        self.COST_MAT  = COST_CONFIG['cost_material']
        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']

        self.fixed_w, self.fixed_h = plate_size
        self.plate_w = 200
        self.plate_h = 200
        self.episode_time_scale = 100.0

        # ── 奖励权重 ──
        self.w_util        = 1.0
        self.w_new_plate   = 3.0
        self.w_terminal    = 10.0

        # 通信向量维度
        self.COMM_DIM_IN  = 16
        self.COMM_DIM_OUT = 8
        self.skyline_bins = FEATURE_CONFIG.get('skyline_bins', 20)
        self.num_strategies = 3

        # 掩码参数
        self.max_mask_candidates = 20

        # obs 维度
        _obs_dim = self.max_capacity * self.PART_FEAT_DIM + self.STATE_FEAT_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_capacity * 2 * self.num_strategies)

        self.scheduler_state_machine = SchedulerStateMachine(num_machines=3)
        self.parts_pool      = []
        self.orders          = {}
        self.packed_indices  = set()
        self.active_plates   = []
        self.history_plates  = []
        self.cost_metrics    = {}

        # 通信向量初始化
        self.sched_intent_vec   = np.zeros(self.COMM_DIM_IN,  dtype=np.float32)
        self.nesting_result_vec = np.zeros(self.COMM_DIM_OUT, dtype=np.float32)
        self.current_instance = None

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        supplied_instance = options.get('instance')
        if supplied_instance is not None:
            if not isinstance(supplied_instance, ProductionInstance):
                raise TypeError("options['instance'] must be a ProductionInstance")
            if len(supplied_instance.parts) > self.max_capacity:
                raise ValueError("ProductionInstance exceeds max part capacity")
            self.current_instance = copy.deepcopy(supplied_instance)
            self.current_num_parts = len(self.current_instance.parts)
            self.plate_w, self.plate_h = self.current_instance.plate_w, self.current_instance.plate_h
            self.parts_pool = copy.deepcopy(self.current_instance.parts)
            self.orders = copy.deepcopy(self.current_instance.orders)
        elif 'num_parts' in options:
            self.current_num_parts = options['num_parts']
        else:
            lo, hi = TRAIN_CONFIG['min_parts'], TRAIN_CONFIG['max_parts']
            self.current_num_parts = int(self.np_random.integers(lo, hi + 1))
        self.current_num_parts = min(self.current_num_parts, self.max_capacity)
        if supplied_instance is None:
            if 'plate_size' in options:
                self.plate_w, self.plate_h = options['plate_size']
            else:
                lo, hi = TRAIN_CONFIG['min_plate_dim'], TRAIN_CONFIG['max_plate_dim']
                self.plate_w = int(self.np_random.integers(lo, hi + 1))
                self.plate_h = int(self.np_random.integers(lo, hi + 1))
            self.current_instance = generate_instance(
                seed=seed, num_parts=self.current_num_parts,
                plate_size=(self.plate_w, self.plate_h),
                num_machines=self.scheduler_state_machine.num_machines,
                cutting_speed=self.CUTTING_SPEED, rng=self.np_random,
            )
            self.parts_pool = copy.deepcopy(self.current_instance.parts)
            self.orders = copy.deepcopy(self.current_instance.orders)
        self.episode_time_scale = max(10.0, parts_cutting_time(self.parts_pool, self.CUTTING_SPEED))

        self.packed_indices  = set()
        self.scheduler_state_machine.reset()
        self.active_plates   = [PlateLayoutManager(width=self.plate_w, height=self.plate_h)]
        self.history_plates  = []
        self.cost_metrics    = {}
        context = np.asarray(options.get("scheduling_context", np.zeros(self.COMM_DIM_IN)), dtype=np.float32)
        if context.shape != (self.COMM_DIM_IN,):
            raise ValueError(f"scheduling_context must have shape ({self.COMM_DIM_IN},)")
        self.sched_intent_vec = context.copy()
        self.nesting_result_vec = np.zeros(self.COMM_DIM_OUT, dtype=np.float32)

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    # ── 解耦观测接口 ─────────────────────────────────────────────────────────

    def get_part_feats(self) -> np.ndarray:
        mach_times = self.scheduler_state_machine.get_state()
        curr_time  = float(np.min(mach_times)) if len(mach_times) > 0 else 0.0
        safe_scale = max(1.0, self.episode_time_scale)
        plate_area = self.plate_w * self.plate_h

        feats = np.zeros((self.max_capacity, self.PART_FEAT_DIM), dtype=np.float32)
        for i in range(self.current_num_parts):
            p = self.parts_pool[i]
            feats[i] = [
                p['w'] / self.plate_w,
                p['h'] / self.plate_h,
                p['area'] / plate_area,
                (p['due_date'] - curr_time) / safe_scale,
                1.0 if i in self.packed_indices else 0.0,
            ]
        return feats

    def get_state_feat(self) -> np.ndarray:
        skyline_feat = np.zeros(self.skyline_bins, dtype=np.float32)
        act_util_avg = act_util_max = act_util_min = 0.0
        act_free_area_ratio = act_max_free_area = max_free_w = max_free_h = 0.0

        if self.active_plates:
            skyline_feat = self.active_plates[-1].get_normalized_skyline(self.skyline_bins)
            utils = [p.utilization for p in self.active_plates]
            act_util_avg, act_util_max, act_util_min = np.mean(utils), np.max(utils), np.min(utils)
            all_free = [r for p in self.active_plates for r in p.free_rects]
            total_area = self.plate_w * self.plate_h * len(self.active_plates)
            if all_free and total_area > 0:
                act_free_area_ratio = sum(r[2]*r[3] for r in all_free) / total_area
                act_max_free_area   = max(r[2]*r[3] for r in all_free) / (self.plate_w*self.plate_h)
                max_free_w = max(r[2] for r in all_free) / self.plate_w
                max_free_h = max(r[3] for r in all_free) / self.plate_h

        act_cnt_norm = len(self.active_plates) / 10.0
        rem_idxs     = [i for i in range(self.current_num_parts) if i not in self.packed_indices]
        mach_times   = self.scheduler_state_machine.get_state()
        curr_time    = float(np.min(mach_times)) if len(mach_times) > 0 else 0.0
        safe_scale   = max(1.0, self.episode_time_scale)
        plate_area   = self.plate_w * self.plate_h

        if rem_idxs:
            rem_areas = [self.parts_pool[i]['area'] for i in rem_idxs]
            rem_dues  = [self.parts_pool[i]['due_date'] for i in rem_idxs]
            rem_avg_area    = np.mean(rem_areas) / plate_area
            rem_max_area    = np.max(rem_areas) / plate_area
            rem_total_ratio = (sum(rem_areas) / plate_area) / 10.0
            diffs = [(d - curr_time) / safe_scale for d in rem_dues]
            rem_avg_due, rem_min_due, rem_due_std = np.mean(diffs), np.min(diffs), np.std(diffs)
        else:
            rem_avg_area = rem_max_area = rem_total_ratio = 0.0
            rem_avg_due  = rem_min_due  = rem_due_std     = 0.0

        progress = len(self.packed_indices) / max(1, self.current_num_parts)

        mach_load_avg = mach_load_std = 0.0

        act_due_std = order_frag_idx = proj_tardiness = mach_idle_gap = 0.0
        if self.active_plates and self.active_plates[-1].placed_parts:
            lp   = self.active_plates[-1]
            oids = [int(p[4]) for p in lp.placed_parts]
            dues = [self.orders[o]['due_date'] for o in oids if o in self.orders]
            if len(dues) > 1:
                act_due_std = np.std(dues) / safe_scale
            frag = []
            for oid in set(oids):
                tot = sum(1 for p in self.parts_pool if p['order_id'] == oid)
                frag.append(1.0 - oids.count(oid) / max(1, tot))
            if frag:
                order_frag_idx = float(np.mean(frag))
            cut_t = plate_processing_time(lp.placed_parts, self.CUTTING_SPEED)
            if dues:
                proj_tardiness = max(0.0, (curr_time + cut_t - min(dues)) / safe_scale)
        if len(mach_times) > 0:
            mach_idle_gap = (float(np.max(mach_times)) - curr_time) / safe_scale

        global_feats = np.array([
            progress, rem_avg_area, rem_max_area, rem_total_ratio,
            rem_avg_due, rem_min_due, rem_due_std, act_cnt_norm,
            act_util_avg, act_util_max, act_util_min,
            act_free_area_ratio, act_max_free_area, max_free_w, max_free_h,
            mach_load_avg, mach_load_std,
            act_due_std, order_frag_idx, proj_tardiness, mach_idle_gap,
        ], dtype=np.float32)

        state = np.concatenate([global_feats, skyline_feat, self.sched_intent_vec])
        return np.clip(np.nan_to_num(state, 0.0), -5.0, 5.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.get_part_feats().flatten(),
            self.get_state_feat(),
        ]).astype(np.float32)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        strategy_id = action % self.num_strategies
        is_rotated  = (action // self.num_strategies) % 2
        part_index  = action // (self.num_strategies * 2)

        if part_index >= self.current_num_parts or part_index in self.packed_indices:
            return self._get_obs(), -5.0, True, False, {}

        part = self.parts_pool[part_index]
        self.packed_indices.add(part_index)
        reward = 0.0

        part_w, part_h = part['w'], part['h']
        if is_rotated == 1:
            part_w, part_h = part_h, part_w

        # ── 在所有 active_plates 中找最佳放置位置 ──
        best_idx, best_score = -1, -float('inf')
        for idx, plate in enumerate(self.active_plates):
            sim = copy.deepcopy(plate)
            ok, sx, sy, sw, sh, _ = sim.place_part(part_w, part_h, part['order_id'], strategy_id)
            if ok:
                score = sim.utilization - plate.utilization
                if score > best_score:
                    best_score, best_idx = score, idx

        if best_idx != -1:
            target   = self.active_plates[best_idx]
            old_util = target.utilization
            s, x, y, w, h, _ = target.place_part(part_w, part_h, part['order_id'], strategy_id)
            if not s:
                s, x, y, w, h, _ = target.place_part(part_w, part_h, part['order_id'], 2)
            new_util = target.utilization

            # 步奖励：利用率增量
            reward += (new_util - old_util) * 15.0 * self.w_util
        else:
            # 开新板惩罚
            last_util = self.active_plates[-1].utilization if self.active_plates else 0.0
            reward -= self.w_new_plate * (1.0 + (1.0 - last_util))

            new_plate = PlateLayoutManager(width=self.plate_w, height=self.plate_h)
            s, x, y, w, h, _ = new_plate.place_part(part_w, part_h, part['order_id'], strategy_id)
            if s:
                self.active_plates.append(new_plate)
            else:
                reward -= 2.0

        terminated = len(self.packed_indices) == self.current_num_parts
        info = {}

        if terminated:
            self.history_plates = self.active_plates
            final_plates = [p for p in self.history_plates if len(p.placed_parts) > 0]

            snapshots = tuple(
                NestedPlateResult(
                    plate_index=index,
                    placed_parts=tuple(
                        (float(p[0]), float(p[1]), float(p[2]), float(p[3]), int(p[4]), bool(p[5]))
                        for p in plate.placed_parts
                    ),
                )
                for index, plate in enumerate(final_plates)
            )
            info["terminal_result"] = NestingTerminalResult(
                instance=self.current_instance, plates=snapshots)

        info["action_mask"] = self._get_action_mask()
        return self._get_obs(), reward, terminated, False, info

    # ── 数据生成 ─────────────────────────────────────────────────────────────

    def _generate_random_orders(self, seed=None):
        instance = generate_instance(
            seed=seed,
            num_parts=self.current_num_parts,
            plate_size=(self.plate_w, self.plate_h),
            num_machines=self.scheduler_state_machine.num_machines,
            cutting_speed=self.CUTTING_SPEED,
            rng=self.np_random,
        )
        return instance.parts, instance.orders

    def _compute_metrics(self):
        m = self.cost_metrics.copy()
        m['late_orders_count'] = sum(
            1 for o in self.orders.values() if o['finished_time'] > o['due_date'])
        if self.history_plates:
            utils = [p.utilization for p in self.history_plates]
            m['raw_avg_utilization'] = float(np.mean(utils))
            m['adj_avg_utilization'] = float(np.mean(sorted(utils)[1:])) if len(utils) > 1 else utils[0]
        else:
            m['raw_avg_utilization'] = m['adj_avg_utilization'] = 0.0
        return m

    # ── Action Mask ──────────────────────────────────────────────────────────

    def _get_action_mask(self) -> np.ndarray:
        total = self.max_capacity * 6
        mask  = np.zeros(total, dtype=bool)

        unpacked = [i for i in range(self.current_num_parts) if i not in self.packed_indices]
        if not unpacked:
            return np.ones(total, dtype=bool)

        # 1. 几何可行性检查
        fittable = set()
        for plate in self.active_plates:
            if not plate.free_rects:
                continue
            mfw = max(r[2] for r in plate.free_rects)
            mfh = max(r[3] for r in plate.free_rects)
            for i in unpacked:
                p = self.parts_pool[i]
                if (p['w'] <= mfw and p['h'] <= mfh) or \
                   (p['h'] <= mfw and p['w'] <= mfh):
                    fittable.add(i)

        # 2. 兜底候选
        top_area = sorted(unpacked, key=lambda i: self.parts_pool[i]['area'], reverse=True)[:3]
        top_due  = sorted(unpacked, key=lambda i: self.parts_pool[i]['due_date'])[:3]
        candidates = fittable | set(top_area) | set(top_due)

        # 3. 截取
        if len(candidates) > self.max_mask_candidates:
            plate_area = max(1.0, self.plate_w * self.plate_h)
            safe_scale = max(1.0, self.episode_time_scale)

            def priority_score(i):
                p = self.parts_pool[i]
                urgency = p['due_date'] / safe_scale
                area_fit = p['area'] / plate_area
                return urgency - area_fit * 0.5

            guaranteed = set(top_area) | set(top_due)
            remaining  = sorted(candidates - guaranteed, key=priority_score)
            budget = self.max_mask_candidates - len(guaranteed)
            candidates = guaranteed | set(remaining[:max(0, budget)])

        # 4. 构建掩码
        for i in candidates:
            mask[i * 6: (i + 1) * 6] = True

        if not mask.any():
            for i in unpacked:
                mask[i * 6: (i + 1) * 6] = True

        return mask
