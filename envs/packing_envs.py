"""
envs/packing_envs.py

协同优化改进：
  - 终局奖励通过 JointRewardCalculator 计算
  - Phase 1 (无 scheduling partner): EDD fallback
  - Phase 2+ (有 scheduling partner): 用真实 Scheduling agent rollout
  - 两个 agent 共享同一个 GlobalCostFunction，消除评价标准分歧
  - NestingResultEncoder 参数纳入 NestingModel 优化器（在 train_dual.py 中配置）

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
from models.comm_encoders import (
    NestingResultEncoder, GlobalCostFunction, JointRewardCalculator
)


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
        self.scheduler_model = None
        self.parts_pool      = []
        self.orders          = {}
        self.packed_indices  = set()
        self.active_plates   = []
        self.history_plates  = []
        self.cost_metrics    = {}

        # 通信向量初始化
        self.sched_intent_vec   = np.zeros(self.COMM_DIM_IN,  dtype=np.float32)
        self.nesting_result_vec = np.zeros(self.COMM_DIM_OUT, dtype=np.float32)
        self.nesting_result_encoder = NestingResultEncoder(input_dim=8, comm_dim=8)

        # ── 协同优化核心：联合奖励计算器 ──
        self.global_cost_fn = GlobalCostFunction(
            cost_mat=self.COST_MAT,
            cost_hold=self.COST_HOLD,
            cost_tard=self.COST_TARD,
            cutting_speed=self.CUTTING_SPEED,
        )
        self.joint_reward_calc = JointRewardCalculator(
            global_cost_fn=self.global_cost_fn,
            num_machines=self.scheduler_state_machine.num_machines,
        )

        # Scheduling 侧引用（Phase 2+ 时注入）
        self._sched_env_for_reward   = None
        self._sched_model_for_reward = None

    # ── 合作接口 ─────────────────────────────────────────────────────────────

    def set_scheduling_partner(self, model):
        """设置调度 partner（model 用于通信向量，reward 用另一个接口）"""
        self.scheduler_model = model

    def set_scheduling_for_reward(self, sched_env, sched_model):
        """
        注入 Scheduling env 和 model，用于终局奖励的真实调度 rollout。
        在 train_dual.py Phase 2+ 中调用。
        """
        self._sched_env_for_reward   = sched_env
        self._sched_model_for_reward = sched_model

    def set_scheduling_intent(self, intent_vec: np.ndarray):
        self.sched_intent_vec = np.array(intent_vec, dtype=np.float32)

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and 'num_parts' in options:
            self.current_num_parts = options['num_parts']
        else:
            lo, hi = TRAIN_CONFIG['min_parts'], TRAIN_CONFIG['max_parts']
            self.current_num_parts = np.random.randint(lo, hi + 1)
        self.current_num_parts = min(self.current_num_parts, self.max_capacity)

        if options and 'plate_size' in options:
            self.plate_w, self.plate_h = options['plate_size']
        else:
            lo, hi = TRAIN_CONFIG['min_plate_dim'], TRAIN_CONFIG['max_plate_dim']
            self.plate_w = np.random.randint(lo, hi + 1)
            self.plate_h = np.random.randint(lo, hi + 1)

        self.parts_pool, self.orders = self._generate_random_orders()
        total_perim = sum(2 * (p['w'] + p['h']) for p in self.parts_pool)
        self.episode_time_scale = max(10.0, total_perim / self.CUTTING_SPEED)

        self.packed_indices  = set()
        self.scheduler_state_machine.reset()
        self.active_plates   = [PlateLayoutManager(width=self.plate_w, height=self.plate_h)]
        self.history_plates  = []
        self.cost_metrics    = {}
        self.sched_intent_vec   = np.zeros(self.COMM_DIM_IN,  dtype=np.float32)
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

        if self.scheduler_model:
            m_rel = (mach_times - curr_time) / safe_scale
            mach_load_avg, mach_load_std = float(np.mean(m_rel)), float(np.std(m_rel))
        else:
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
            cut_t = sum(2*(p[2]+p[3]) for p in lp.placed_parts) / max(0.1, self.CUTTING_SPEED)
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

            # ── 协同优化核心：用联合奖励计算器替代独立的 EDD 模拟 ──
            cost_dict = self.joint_reward_calc.compute_terminal_cost(
                plates=final_plates,
                orders=self.orders,
                parts_pool=self.parts_pool,
                plate_w=self.plate_w,
                plate_h=self.plate_h,
                sched_env=self._sched_env_for_reward,
                sched_model=self._sched_model_for_reward,
            )

            # 终局奖励：共享的 reward 转换逻辑（含 EMA baseline 方差缩减）
            scaled_reward = self.joint_reward_calc.to_reward(
                cost_dict, w_terminal=self.w_terminal)
            reward += scaled_reward

            # 编码排样结果摘要
            self.nesting_result_vec = self.nesting_result_encoder.encode_result(
                self.history_plates, self.orders, self.parts_pool,
                self.plate_w * self.plate_h)

            self.cost_metrics = {
                "cost_material": cost_dict['cost_material'],
                "cost_jit": cost_dict['cost_jit'],
                "cost_total": cost_dict['cost_total'],
                "utilization": cost_dict['utilization'],
                "plate_count": cost_dict['plate_count'],
                "total_delay": cost_dict['total_delay'],
                "norm_reward": scaled_reward,
            }
            info["episode_metrics"] = self._compute_metrics()

        info["action_mask"] = self._get_action_mask()
        return self._get_obs(), reward, terminated, False, info

    # ── 数据生成 ─────────────────────────────────────────────────────────────

    def _generate_random_orders(self):
        data, orders = [], {}
        cnt, oid = 0, 0
        avg_perim    = 2 * (0.25 * self.plate_w + 0.25 * self.plate_h)
        est_makespan = (self.current_num_parts * avg_perim / max(0.1, self.CUTTING_SPEED) / 3) * 1.3

        while cnt < self.current_num_parts:
            batch = np.random.randint(1, 16)
            if cnt + batch > self.current_num_parts:
                batch = self.current_num_parts - cnt
            temp, order_p = [], 0
            for _ in range(batch):
                w_ratio = np.random.beta(a=2, b=5) * 0.9 + 0.05
                h_ratio = np.random.beta(a=2, b=5) * 0.9 + 0.05
                if np.random.rand() > 0.5:
                    w_ratio, h_ratio = h_ratio, w_ratio
                w = max(1, int(w_ratio * self.plate_w))
                h = max(1, int(h_ratio * self.plate_h))
                order_p += 2 * (w + h)
                temp.append({'w': w, 'h': h, 'area': w * h})
            self_time   = order_p / max(0.1, self.CUTTING_SPEED)
            buffer_time = (np.random.poisson(lam=2.0) + 0.1) * (est_makespan / 2.0)
            final_due   = self_time + buffer_time
            orders[oid] = {'due_date': final_due, 'finished_time': 0.0}
            for p in temp:
                data.append({'w': p['w'], 'h': p['h'], 'area': p['area'],
                             'due_date': final_due, 'order_id': oid, 'original_idx': cnt})
                cnt += 1
            oid += 1
        np.random.shuffle(data)
        return data, orders

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