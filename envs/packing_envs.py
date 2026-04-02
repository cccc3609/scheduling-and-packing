"""
envs/packing_envs.py

修复清单：
  1. COMM_DIM_IN 先用后定义 → 调整顺序
  2. nesting_result_vec 未初始化 → __init__ 和 reset 补全
  3. step() 中 nesting_result_vec 赋零 → 改调 encode_result()
  4. _get_obs 截断 / _get_obs_loop_replacement 孤立 → 删除孤立方法，合并回 _get_obs
  5. _generate_random_orders 旋转 bug (h_range) → 修复
  6. _simulate_with_edd 后游离代码 → 删除

新增：
  get_part_feats()  → [max_capacity, 5]  供 PartEncoder（episode 级，只跑一次）
  get_state_feat()  → [57]               供 StepDecoder（每步）
"""

import gymnasium as gym
import numpy as np
import copy
import math
from gymnasium import spaces

from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import MAX_PARTS_CAPACITY, TRAIN_CONFIG, MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, FEATURE_CONFIG
from models.comm_encoders import NestingResultEncoder


class NestingSchedulingEnv(gym.Env):

    # 特征维度常量
    PART_FEAT_DIM  = 5    # w, h, area, due, is_packed
    STATE_FEAT_DIM = 57   # global(21) + skyline(20) + sched_intent(16)

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

        self.w_util        = 1.0
        self.w_jit         = 12.0
        self.w_grouping    = 0.5
        self.w_step_compact = 0.5
        self.w_new_plate   = 2.0

        # 通信向量维度（先定义，后使用）
        self.COMM_DIM_IN  = 16
        self.COMM_DIM_OUT = 8
        self.skyline_bins = FEATURE_CONFIG.get('skyline_bins', 20)
        self.num_strategies = 3

        # gymnasium obs：part_feats展平(600) + state_feat(57) = 657
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

        # 通信向量在 __init__ 里初始化，避免 Phase1 访问 AttributeError
        self.sched_intent_vec   = np.zeros(self.COMM_DIM_IN,  dtype=np.float32)
        self.nesting_result_vec = np.zeros(self.COMM_DIM_OUT, dtype=np.float32)
        self.nesting_result_encoder = NestingResultEncoder(input_dim=8, comm_dim=8)

    # ── 合作接口 ─────────────────────────────────────────────────────────────

    def set_scheduling_partner(self, model):
        self.scheduler_model = model

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
        """
        零件局部静态特征 [max_capacity, 5]。
        episode reset 后调用一次，传给 PartEncoder。
        """
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
        """
        当前动态状态特征 [57]。
        每步调用，传给 StepDecoder。
        global(21) + skyline(20) + sched_intent(16)
        """
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
        ], dtype=np.float32)  # 21 维

        state = np.concatenate([global_feats, skyline_feat, self.sched_intent_vec])
        return np.clip(np.nan_to_num(state, 0.0), -5.0, 5.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """gymnasium 兼容展平 obs（自定义训练循环请用 get_part_feats + get_state_feat）"""
        return np.concatenate([
            self.get_part_feats().flatten(),
            self.get_state_feat(),
        ]).astype(np.float32)

    # ── Placement Quality ────────────────────────────────────────────────────

    def _evaluate_placement_quality(self, plate, part_w, part_h, part_due,
                                    part_area, x, y, w, h, current_util):
        geo_score = current_util * 10.0
        geo_score += sum([x == 0, y == 0,
                          x + w == self.plate_w,
                          y + h == self.plate_h]) * 0.2
        time_penalty = 0.0
        if plate.placed_parts:
            oids = list(set(int(p[4]) for p in plate.placed_parts))
            dues = [self.orders[o]['due_date'] for o in oids if o in self.orders]
            if dues:
                min_d, max_d = min(dues), max(dues)
                if part_due < min_d:
                    time_penalty += (min_d - part_due) / self.episode_time_scale * 15.0
                expansion = (max(max_d, part_due) - min(min_d, part_due)) - (max_d - min_d)
                time_penalty += (expansion / self.episode_time_scale) * 2.0
        return self.w_util * geo_score - self.w_jit * time_penalty

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        strategy_id = action % self.num_strategies
        is_rotated  = (action // self.num_strategies) % 2
        part_index  = action // (self.num_strategies * 2)

        if part_index >= self.current_num_parts or part_index in self.packed_indices:
            return self._get_obs(), -100.0, True, False, {}

        part = self.parts_pool[part_index]
        self.packed_indices.add(part_index)
        reward = 0.0

        part_w, part_h = part['w'], part['h']
        if is_rotated == 1:
            part_w, part_h = part_h, part_w

        best_idx, best_score = -1, -float('inf')
        for idx, plate in enumerate(self.active_plates):
            sim = copy.deepcopy(plate)
            ok, sx, sy, sw, sh, _ = sim.place_part(part_w, part_h, part['order_id'], strategy_id)
            if ok:
                score = self._evaluate_placement_quality(
                    plate, part_w, part_h, part['due_date'], part['area'],
                    sx, sy, sw, sh, sim.utilization)
                if score > best_score:
                    best_score, best_idx = score, idx

        if best_idx != -1:
            target   = self.active_plates[best_idx]
            old_util = target.utilization
            s, x, y, w, h, _ = target.place_part(part_w, part_h, part['order_id'], strategy_id)
            if not s:
                s, x, y, w, h, _ = target.place_part(part_w, part_h, part['order_id'], 2)
            new_util = target.utilization
            reward += (new_util - old_util) * 20.0

            oids = list(set(int(p[4]) for p in target.placed_parts))
            if len(oids) > 1:
                dues = [self.orders[o]['due_date'] for o in oids if o in self.orders]
                if len(dues) > 1:
                    reward -= (np.std(dues) / max(1.0, self.episode_time_scale)) * self.w_grouping

            cx = cy = count = 0
            for p in target.placed_parts:
                cx += p[0] + p[2] / 2; cy += p[1] + p[3] / 2; count += 1
            curr_cx, curr_cy = x + w / 2, y + h / 2
            max_d = (self.plate_w**2 + self.plate_h**2) ** 0.5
            if count > 1:
                d = ((curr_cx - cx/count)**2 + (curr_cy - cy/count)**2) ** 0.5
            else:
                d = (curr_cx**2 + curr_cy**2) ** 0.5
            reward += (1.0 - d / max_d) * self.w_step_compact
        else:
            if self.active_plates:
                reward -= (1.0 - self.active_plates[-1].utilization) * 50.0
            new_plate = PlateLayoutManager(width=self.plate_w, height=self.plate_h)
            s, x, y, w, h, _ = new_plate.place_part(part_w, part_h, part['order_id'], strategy_id)
            if s:
                self.active_plates.append(new_plate)
                reward -= self.w_new_plate
            else:
                reward -= 50.0

        terminated = len(self.packed_indices) == self.current_num_parts
        info = {}

        if terminated:
            self.history_plates = self.active_plates
            final_plates = [p for p in self.history_plates if len(p.placed_parts) > 0]

            total_part_area = sum(p['area'] for p in self.parts_pool)
            consumed_area   = len(final_plates) * (self.plate_w * self.plate_h)
            utilization     = total_part_area / consumed_area if consumed_area > 0 else 0.001
            cost_material   = max(0.0, consumed_area - total_part_area) * self.COST_MAT

            order_finishes = self._simulate_with_edd(final_plates)
            cost_jit = total_delay = 0.0
            for oid, fin in order_finishes.items():
                due  = self.orders[oid]['due_date']
                ops  = [p for p in self.parts_pool if p['order_id'] == oid]
                val  = sum(p['area'] for p in ops)
                proc = max(1.0, sum(2*(p['w']+p['h']) for p in ops) / self.CUTTING_SPEED)
                diff = fin - due
                ratio = abs(diff) / proc
                coef  = 0.0 if ratio <= 0.025 else (min(1.0, (ratio-0.025)/0.05))
                if diff > 0:
                    cost_jit += val * self.COST_TARD * coef * abs(diff); total_delay += diff
                else:
                    cost_jit += val * self.COST_HOLD * coef * abs(diff)

            total_penalty = cost_material + cost_jit
            intrinsic     = max(1.0, total_part_area * self.COST_MAT)
            pr = total_penalty / intrinsic
            if pr <= 1.0:
                scaled_reward = (0.5 - pr) * 40.0
            else:
                scaled_reward = -20.0 - math.log(max(1e-5, pr)) * 10.0
            scaled_reward = float(np.clip(scaled_reward, -100.0, 50.0))
            reward += scaled_reward

            # 修复：调用 encoder 生成真实排样摘要
            self.nesting_result_vec = self.nesting_result_encoder.encode_result(
                self.history_plates, self.orders, self.parts_pool,
                self.plate_w * self.plate_h)

            self.cost_metrics = {
                "cost_material": cost_material, "cost_jit": cost_jit,
                "cost_total": total_penalty, "utilization": utilization,
                "plate_count": len(final_plates), "total_delay": total_delay,
                "norm_reward": scaled_reward,
            }
            info["episode_metrics"] = self._compute_metrics()

        info["action_mask"] = self._get_action_mask()
        return self._get_obs(), reward, terminated, False, info

    # ── EDD 模拟（局部副本，不污染 self.orders）─────────────────────────────

    def _simulate_with_edd(self, plates_list):
        local = {oid: {'due_date': v['due_date'], 'finished_time': 0.0}
                 for oid, v in self.orders.items()}
        tasks = []
        for plate in plates_list:
            cut  = sum(2*(p[2]+p[3]) for p in plate.placed_parts) / self.CUTTING_SPEED
            oids = list(set(int(p[4]) for p in plate.placed_parts))
            due  = min((local[o]['due_date'] for o in oids if o in local), default=999.0)
            tasks.append({'cut': cut, 'due': due, 'oids': oids})
        tasks.sort(key=lambda t: t['due'])
        mach = [0.0] * self.scheduler_state_machine.num_machines
        for task in tasks:
            m = int(min(range(len(mach)), key=lambda i: mach[i]))
            end = mach[m] + task['cut']
            mach[m] = end
            for oid in task['oids']:
                if oid in local:
                    local[oid]['finished_time'] = max(local[oid]['finished_time'], end)
        return {oid: v['finished_time'] for oid, v in local.items()}

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
                    w_ratio, h_ratio = h_ratio, w_ratio   # 修复：正确的 tuple swap
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
        if len(self.packed_indices) >= self.current_num_parts:
            return np.ones(total, dtype=bool)

        unpacked = [i for i in range(self.current_num_parts) if i not in self.packed_indices]
        top_area = sorted(unpacked, key=lambda i: self.parts_pool[i]['area'], reverse=True)[:2]
        top_due  = sorted(unpacked, key=lambda i: self.parts_pool[i]['due_date'])[:2]

        bottom = []
        if self.active_plates and self.active_plates[-1].free_rects:
            fr = self.active_plates[-1].free_rects
            mfw, mfh = max(r[2] for r in fr), max(r[3] for r in fr)
            fittable = [i for i in unpacked
                        if (self.parts_pool[i]['w'] <= mfw and self.parts_pool[i]['h'] <= mfh)
                        or (self.parts_pool[i]['h'] <= mfw and self.parts_pool[i]['w'] <= mfh)]
            bottom = sorted(fittable, key=lambda i: self.parts_pool[i]['area'], reverse=True)[:2]
        if not bottom:
            bottom = sorted(unpacked, key=lambda i: self.parts_pool[i]['area'])[:2]

        for i in set(top_area + top_due + bottom):
            mask[i * 6: (i + 1) * 6] = True
        return mask