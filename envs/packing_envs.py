import gymnasium as gym
import numpy as np
import copy
from gymnasium import spaces

from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import MAX_PARTS_CAPACITY, TRAIN_CONFIG, MAX_SCHED_TASKS_CAPACITY, COST_CONFIG


class NestingSchedulingEnv(gym.Env):
    def __init__(self, plate_size=(200, 200)):
        super(NestingSchedulingEnv, self).__init__()

        self.max_capacity = MAX_PARTS_CAPACITY
        self.max_sched_capacity = MAX_SCHED_TASKS_CAPACITY
        self.current_num_parts = TRAIN_CONFIG['min_parts']

        # === 1. 读取经济与物理参数 ===
        self.CUTTING_SPEED = COST_CONFIG['cutting_speed']
        self.COST_MAT = COST_CONFIG['cost_material']
        self.COST_HOLD = COST_CONFIG['cost_earliness']  # Alpha
        self.COST_TARD = COST_CONFIG['cost_tardiness']  # Beta

        # 物理尺寸 (Reset时可变)
        self.fixed_w, self.fixed_h = plate_size
        self.plate_w = 200
        self.plate_h = 200

        # 动态归一化因子
        self.episode_time_scale = 100.0

        # === 2. 辅助引导权重 ===
        self.w_grouping = 0.5
        self.w_step_compact = 0.5
        self.w_new_plate = 8.0

        self.ALPHA = 1.0
        self.BETA = 1.0

        # === 3. 空间定义 ===
        self.obs_feature_dim = 22
        self.num_strategies = 3
        self.action_space = spaces.Discrete(self.max_capacity * 2 * self.num_strategies)
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(self.max_capacity * self.obs_feature_dim,), dtype=np.float32
        )

        self.scheduler_state_machine = SchedulerStateMachine(num_machines=3)
        self.scheduler_model = None
        self.parts_pool = []
        self.orders = {}
        self.packed_indices = set()
        self.active_plates = []
        self.history_plates = []

        self.cost_metrics = {}
        self.last_norm_util = 0.0
        self.last_norm_jit = 0.0
        self.last_raw_jit = 0.0

    def set_scheduling_partner(self, model):
        self.scheduler_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and 'num_parts' in options:
            self.current_num_parts = options['num_parts']
        else:
            min_p, max_p = TRAIN_CONFIG['min_parts'], TRAIN_CONFIG['max_parts']
            self.current_num_parts = np.random.randint(min_p, max_p + 1)

        if self.current_num_parts > self.max_capacity:
            self.current_num_parts = self.max_capacity

        if options and 'plate_size' in options:
            self.plate_w, self.plate_h = options['plate_size']
        else:
            min_d, max_d = TRAIN_CONFIG['min_plate_dim'], TRAIN_CONFIG['max_plate_dim']
            self.plate_w = np.random.randint(min_d, max_d + 1)
            self.plate_h = np.random.randint(min_d, max_d + 1)

        self.parts_pool, self.orders = self._generate_random_orders()

        # 计算本局时间标尺 (总物理工时)
        total_perimeter = sum([2 * (p['w'] + p['h']) for p in self.parts_pool])
        self.episode_time_scale = max(10.0, total_perimeter / self.CUTTING_SPEED)

        self.packed_indices = set()
        self.scheduler_state_machine.reset()
        self.active_plates = [PlateLayoutManager(width=self.plate_w, height=self.plate_h)]
        self.history_plates = []
        self.cost_metrics = {}

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def step(self, action):
        if isinstance(action, np.ndarray): action = int(action)

        strategy_id = action % self.num_strategies
        temp_action = action // self.num_strategies
        part_index = temp_action // 2

        if part_index >= self.current_num_parts or part_index in self.packed_indices:
            return self._get_obs(), -100.0, True, False, {}

        part = self.parts_pool[part_index]
        self.packed_indices.add(part_index)

        reward = 0.0

        # Best Fit
        best_plate_idx = -1
        best_plate_score = -float('inf')
        part_due = part['due_date']
        is_small_part = (part['area'] / (self.plate_w * self.plate_h)) < 0.05

        for idx, plate in enumerate(self.active_plates):
            sim_plate = copy.deepcopy(plate)
            success, _, _, _, _, _ = sim_plate.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if success:
                geo_score = sim_plate.utilization
                time_penalty = 0.0
                if not is_small_part:
                    existing_oids = list(set([int(p[4]) for p in plate.placed_parts]))
                    if existing_oids:
                        avg_due = np.mean([self.orders[oid]['due_date'] for oid in existing_oids])
                        time_penalty = abs(part_due - avg_due) / self.episode_time_scale

                final_score = geo_score - (self.w_grouping * time_penalty)
                if final_score > best_plate_score:
                    best_plate_score = final_score
                    best_plate_idx = idx

        if best_plate_idx != -1:
            target = self.active_plates[best_plate_idx]
            s, x, y, w, h, _ = target.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if not s: s, x, y, w, h, _ = target.place_part(part['w'], part['h'], part['order_id'], 2)
            reward += 0.2

            # Compactness
            cx_sum, cy_sum, count = 0, 0, 0
            for p in target.placed_parts: cx_sum += p[0] + p[2] / 2; cy_sum += p[1] + p[3] / 2; count += 1
            current_cx, current_cy = x + w / 2, y + h / 2
            max_dist = (self.plate_w ** 2 + self.plate_h ** 2) ** 0.5
            if count > 1:
                dist = ((current_cx - cx_sum / count) ** 2 + (current_cy - cy_sum / count) ** 2) ** 0.5
                reward += (1.0 - dist / max_dist) * self.w_step_compact
            else:
                dist_origin = (current_cx ** 2 + current_cy ** 2) ** 0.5
                reward += (1.0 - dist_origin / max_dist) * self.w_step_compact
        else:
            new_plate = PlateLayoutManager(width=self.plate_w, height=self.plate_h)
            s, x, y, w, h, _ = new_plate.place_part(part['w'], part['h'], part['order_id'], strategy_id)
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

            # 1. 材料成本
            consumed_area = len(final_plates) * (self.plate_w * self.plate_h)
            cost_material = consumed_area * self.COST_MAT

            # 2. JIT 成本 (全量仿真)
            order_finishes = self._simulate_batch_scheduling_detailed(final_plates)

            cost_jit = 0.0
            total_delay = 0.0

            for oid, finish_time in order_finishes.items():
                due = self.orders[oid]['due_date']

                # 🟢 获取物理工时 (重新计算或从某处获取)
                # 为了准确，我们遍历 parts_pool 计算该订单的物理工时
                order_parts = [p for p in self.parts_pool if p['order_id'] == oid]
                order_area = sum([p['area'] for p in order_parts])  # 价值
                order_perim = sum([2 * (p['w'] + p['h']) for p in order_parts])
                order_proc_time = max(1.0, order_perim / self.CUTTING_SPEED)  # 工时

                diff = finish_time - due
                abs_diff = abs(diff)

                # 梯形窗口计算
                ratio = abs_diff / order_proc_time
                R_FREE, R_FULL = 0.025, 0.075

                if ratio <= R_FREE:
                    coef = 0.0
                elif ratio <= R_FULL:
                    coef = (ratio - R_FREE) / (R_FULL - R_FREE)
                else:
                    coef = 1.0

                rate = self.COST_TARD if diff > 0 else self.COST_HOLD

                # Cost = 价值(面积权重) * 费率 * 系数 * 时间
                val_weight = order_area / (self.plate_w * self.plate_h)
                cost_jit += val_weight * (rate * coef) * abs_diff

                if diff > 0: total_delay += diff

            total_cost = cost_material + cost_jit

            # 3. 归一化
            total_part_area = sum([p['area'] for p in self.parts_pool])
            baseline_cost = total_part_area * self.COST_MAT
            if baseline_cost <= 0: baseline_cost = 1.0

            scaled_reward = - (total_cost / baseline_cost) * 10.0
            reward += scaled_reward

            self.cost_metrics = {
                "cost_material": cost_material,
                "cost_jit": cost_jit,
                "cost_total": total_cost,
                "utilization": total_part_area / consumed_area if consumed_area else 0,
                "plate_count": len(final_plates),
                "total_delay": total_delay,
                "norm_reward": scaled_reward
            }
            info["episode_metrics"] = self._compute_metrics()

        info["action_mask"] = self._get_action_mask()
        return self._get_obs(), reward, terminated, False, info

    def _simulate_batch_scheduling_detailed(self, plates_list):
        temp_sched = SchedulerStateMachine(num_machines=3)
        tasks = []
        for idx, plate in enumerate(plates_list):
            cut_time = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / self.CUTTING_SPEED
            oids = list(set([int(p[4]) for p in plate.placed_parts]))
            min_due = min([self.orders[o]['due_date'] for o in oids]) if oids else 999.0
            val = sum([p[2] * p[3] for p in plate.placed_parts])
            tasks.append({'cut': cut_time, 'due': min_due, 'idx': idx, 'oids': oids, 'val': val, 'done': False})

        for _ in range(len(tasks)):
            mach_times = temp_sched.get_state()
            min_t = np.min(mach_times)
            best_task, best_mach = -1, -1

            if self.scheduler_model:
                MAX_SIM = self.max_sched_capacity
                m_feat = (mach_times - min_t) / self.episode_time_scale
                t_feat = []
                mask = np.zeros(MAX_SIM * 3, dtype=bool)
                has_v = False
                for i in range(MAX_SIM):
                    if i < len(tasks):
                        t = tasks[i]
                        norm_val = t['val'] / 140000.0
                        t_feat.extend([t['cut'] / self.episode_time_scale,
                                       (t['due'] - min_t) / self.episode_time_scale,
                                       1.0 if t['done'] else 0.0,
                                       norm_val])
                        if not t['done']: mask[i * 3:(i + 1) * 3] = True; has_v = True
                    else:
                        t_feat.extend([0, 0, 1, 0])
                if not has_v: mask = np.ones(MAX_SIM * 3, dtype=bool)

                obs = np.concatenate([m_feat, t_feat]).astype(np.float32)
                obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
                obs = np.clip(obs, -5.0, 5.0)

                action, _ = self.scheduler_model.predict(obs, action_masks=mask, deterministic=True)
                best_task = int(action) // 3
                best_mach = int(action) % 3
            else:
                # 启发式：基于成本贪婪
                min_c = float('inf')
                for i, t in enumerate(tasks):
                    if t['done']: continue
                    for m in range(3):
                        end = mach_times[m] + t['cut']
                        diff = end - t['due']
                        # 简化启发式：只看绝对偏差，忽略梯形窗口细节
                        c = self.COST_TARD * diff if diff > 0 else self.COST_HOLD * abs(diff)
                        c += (mach_times[m] - min_t) * 0.1
                        if c < min_c: min_c, best_task, best_mach = c, i, m

            if best_task != -1:
                t = tasks[best_task]
                t['done'] = True
                end = temp_sched.execute_assignment(best_mach, t['cut'], t['idx'])
                for oid in t['oids']: self.orders[oid]['finished_time'] = max(self.orders[oid]['finished_time'], end)

        results = {}
        for oid, order in self.orders.items():
            results[oid] = order['finished_time']
        return results

    def _generate_random_orders(self):
        data, orders = [], {}
        cnt, oid = 0, 0
        avg_perim = 2 * (0.25 * self.plate_w + 0.25 * self.plate_h)
        total_work = self.current_num_parts * avg_perim
        est_makespan = (total_work / self.CUTTING_SPEED / 3) * 1.3

        while cnt < self.current_num_parts:
            batch = np.random.randint(3, 9)
            if cnt + batch > self.current_num_parts: batch = self.current_num_parts - cnt

            temp_parts = []
            order_p = 0
            for _ in range(batch):
                w = int(np.random.uniform(0.15, 0.35) * self.plate_w)
                h = int(np.random.uniform(0.15, 0.35) * self.plate_h)
                w, h = max(1, w), max(1, h)
                if np.random.rand() > 0.5: w, h = h, w
                order_p += 2 * (w + h)
                temp_parts.append({'w': w, 'h': h, 'area': w * h})

            self_time = order_p / self.CUTTING_SPEED
            base = self_time * np.random.uniform(1.1, 1.3)
            q = np.random.uniform(0, max(0, est_makespan - self_time))
            final_due = base + q

            orders[oid] = {'due_date': final_due, 'finished_time': 0.0}
            for p in temp_parts:
                data.append({'w': p['w'], 'h': p['h'], 'area': p['area'], 'due_date': final_due, 'order_id': oid,
                             'original_idx': cnt})
                cnt += 1
            oid += 1
        np.random.shuffle(data)
        return data, orders

    def _compute_metrics(self):
        m = self.cost_metrics.copy()
        m['late_orders_count'] = sum([1 for o in self.orders.values() if o['finished_time'] > o['due_date']])
        return m

    def _get_obs(self):
        obs = np.zeros((self.max_capacity, self.obs_feature_dim), dtype=np.float32)

        # 1. 板材状态
        if self.active_plates:
            utils = [p.utilization for p in self.active_plates]
            act_util_avg = np.mean(utils)
            act_util_max = np.max(utils)
            act_util_min = np.min(utils)

            all_free = []
            for p in self.active_plates: all_free.extend(p.free_rects)

            tot_a = self.plate_w * self.plate_h * len(self.active_plates)
            if all_free and tot_a > 0:
                fr = sum([r[2] * r[3] for r in all_free]) / tot_a
                mx = max([r[2] * r[3] for r in all_free]) / (self.plate_w * self.plate_h)
                mw = max([r[2] for r in all_free]) / self.plate_w
                mh = max([r[3] for r in all_free]) / self.plate_h
            else:
                fr, mx, mw, mh = 0, 0, 0, 0
        else:
            act_util_avg = act_util_max = act_util_min = 0.0
            fr = mx = mw = mh = 0.0

        act_cnt = len(self.active_plates) / 10.0

        # 2. 剩余零件状态
        prog = len(self.packed_indices) / max(1, self.current_num_parts)
        rem = [i for i in range(self.current_num_parts) if i not in self.packed_indices]
        curr_time = np.min(self.scheduler_state_machine.get_state())

        if rem:
            areas = [self.parts_pool[i]['area'] for i in rem]
            dues = [self.parts_pool[i]['due_date'] for i in rem]
            r_avg_a = np.mean(areas) / (self.plate_w * self.plate_h)
            r_max_a = np.max(areas) / (self.plate_w * self.plate_h)
            r_tot = sum(areas) / (self.plate_w * self.plate_h) / 10.0
            d_diffs = [(d - curr_time) / self.episode_time_scale for d in dues]
            r_avg_d, r_min_d, r_std_d = np.mean(d_diffs), np.min(d_diffs), np.std(d_diffs)
        else:
            r_avg_a, r_max_a, r_tot = 0, 0, 0
            r_avg_d, r_min_d, r_std_d = 0, 0, 0

        # 3. 机器状态
        if self.scheduler_model:
            m_times = self.scheduler_state_machine.get_state()
            m_rel = (m_times - np.min(m_times)) / self.episode_time_scale
            m_avg, m_std = np.mean(m_rel), np.std(m_rel)
        else:
            m_avg, m_std = 0, 0

        g_feats = [prog, r_avg_a, r_max_a, r_tot, r_avg_d, r_min_d, r_std_d,
                   act_cnt, act_util_avg, act_util_max, act_util_min, fr, mx, mw, mh, m_avg, m_std]

        for i, p in enumerate(self.parts_pool):
            pk = 1.0 if i in self.packed_indices else 0.0
            nw, nh = p['w'] / self.plate_w, p['h'] / self.plate_h
            na = p['area'] / (self.plate_w * self.plate_h)
            rd = (p['due_date'] - curr_time) / self.episode_time_scale
            obs[i] = [nw, nh, na, rd, pk] + g_feats

        return obs.flatten()

    def _get_action_mask(self):
        mask = np.ones(self.max_capacity * 6, dtype=bool)
        valid = self.current_num_parts * 6
        mask[valid:] = False
        for i in range(self.current_num_parts):
            if i in self.packed_indices: mask[i * 6:(i + 1) * 6] = False
        if len(self.packed_indices) >= self.current_num_parts:
            return np.ones(self.max_capacity * 6, dtype=bool)
        return mask