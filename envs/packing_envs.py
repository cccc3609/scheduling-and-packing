import gymnasium as gym
import numpy as np
import copy
import math
from gymnasium import spaces

from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import MAX_PARTS_CAPACITY, TRAIN_CONFIG, MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, FEATURE_CONFIG


class NestingSchedulingEnv(gym.Env):
    def __init__(self, plate_size=(200, 200)):
        super(NestingSchedulingEnv, self).__init__()

        self.max_capacity = MAX_PARTS_CAPACITY
        self.max_sched_capacity = MAX_SCHED_TASKS_CAPACITY
        self.current_num_parts = TRAIN_CONFIG['min_parts']

        # === 1. 物理参数 ===
        self.CUTTING_SPEED = COST_CONFIG['cutting_speed']
        self.COST_MAT = COST_CONFIG['cost_material']
        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']

        # 物理尺寸 (默认值)
        self.fixed_w, self.fixed_h = plate_size
        self.plate_w = 200
        self.plate_h = 200

        # 动态归一化因子
        self.episode_time_scale = 100.0

        # === 2. 权重配置 ===
        self.w_util = 4.0
        self.w_jit = 4.0
        self.w_grouping = 0.5
        self.w_step_compact = 0.5
        self.w_new_plate = 5.0

        # === 3. 空间定义 ===
        # 22维特征
        self.skyline_bins = FEATURE_CONFIG.get('skyline_bins', 20)
        self.obs_feature_dim = 22 + self.skyline_bins

        self.num_strategies = 3
        self.action_space = spaces.Discrete(self.max_capacity * 2 * self.num_strategies)
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(self.max_capacity * self.obs_feature_dim,), dtype=np.float32
        )

        self.scheduler_state_machine = SchedulerStateMachine(num_machines=3)
        self.scheduler_model = None
        self.parts_pool = [];
        self.orders = {};
        self.packed_indices = set()
        self.active_plates = [];
        self.history_plates = [];
        self.cost_metrics = {}
        self.last_norm_util = 0.0
        self.last_norm_jit = 0.0
        self.last_raw_jit = 0.0

    def set_scheduling_partner(self, model):
        self.scheduler_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 动态配置
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

        # 生成数据
        self.parts_pool, self.orders = self._generate_random_orders()

        # 计算本局时间标尺
        total_perimeter = sum([2 * (p['w'] + p['h']) for p in self.parts_pool])
        # 强制最小值为10.0，防止除以0
        self.episode_time_scale = max(10.0, total_perimeter / self.CUTTING_SPEED)

        self.packed_indices = set()
        self.scheduler_state_machine.reset()
        self.active_plates = [PlateLayoutManager(width=self.plate_w, height=self.plate_h)]
        self.history_plates = []
        self.cost_metrics = {}

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _evaluate_placement_quality(self, plate, part_w, part_h, part_due, part_area,
                                    x, y, w, h, current_util):
        # 1. 几何
        geo_score = current_util * 10.0
        touching = 0
        if x == 0: touching += 1
        if y == 0: touching += 1
        if x + w == self.plate_w: touching += 1
        if y + h == self.plate_h: touching += 1
        geo_score += touching * 0.2

        # 2. 时间
        time_penalty = 0.0
        existing_parts = plate.placed_parts
        if existing_parts:
            oids = list(set([int(p[4]) for p in existing_parts]))
            dues = [self.orders[oid]['due_date'] for oid in oids if oid in self.orders]
            if dues:
                min_d, max_d = min(dues), max(dues)
                if part_due < min_d:
                    drop = (min_d - part_due) / self.episode_time_scale
                    time_penalty += drop * 15.0
                new_min = min(min_d, part_due)
                new_max = max(max_d, part_due)
                expansion = (new_max - new_min) - (max_d - min_d)
                time_penalty += (expansion / self.episode_time_scale) * 2.0

        return (self.w_util * geo_score) - (self.w_jit * time_penalty)

    def step(self, action):
        if isinstance(action, np.ndarray): action = int(action)
        strategy_id = action % self.num_strategies
        part_index = (action // self.num_strategies) // 2

        if part_index >= self.current_num_parts or part_index in self.packed_indices:
            return self._get_obs(), -100.0, True, False, {}

        part = self.parts_pool[part_index]
        self.packed_indices.add(part_index)
        reward = 0.0

        best_idx, best_score = -1, -float('inf')
        for idx, plate in enumerate(self.active_plates):
            sim_plate = copy.deepcopy(plate)
            ok, sx, sy, sw, sh, _ = sim_plate.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if ok:
                score = self._evaluate_placement_quality(
                    plate, part['w'], part['h'], part['due_date'], part['area'],
                    sx, sy, sw, sh, sim_plate.utilization
                )
                if score > best_score: best_score, best_idx = score, idx

        if best_idx != -1:
            target = self.active_plates[best_idx]
            s, x, y, w, h, _ = target.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if not s: s, x, y, w, h, _ = target.place_part(part['w'], part['h'], part['order_id'], 2)
            reward += 0.1

            # 紧凑度
            cx_sum, cy_sum, count = 0, 0, 0
            for p in target.placed_parts:
                cx_sum += p[0] + p[2] / 2;
                cy_sum += p[1] + p[3] / 2;
                count += 1
            curr_cx, curr_cy = x + w / 2, y + h / 2
            max_d = (self.plate_w ** 2 + self.plate_h ** 2) ** 0.5
            if count > 1:
                d = ((curr_cx - cx_sum / count) ** 2 + (curr_cy - cy_sum / count) ** 2) ** 0.5
                reward += (1.0 - d / max_d) * self.w_step_compact
            else:
                d = (curr_cx ** 2 + curr_cy ** 2) ** 0.5
                reward += (1.0 - d / max_d) * self.w_step_compact
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

            consumed_area = len(final_plates) * (self.plate_w * self.plate_h)
            cost_material = consumed_area * self.COST_MAT


            order_finishes = self._simulate_batch_scheduling_detailed(final_plates)

            cost_jit = 0.0
            total_delay = 0.0

            for oid, finish_time in order_finishes.items():
                due = self.orders[oid]['due_date']


                order_parts = [p for p in self.parts_pool if p['order_id'] == oid]
                order_value = sum([p['area'] for p in order_parts])


                order_perim = sum([2 * (p['w'] + p['h']) for p in order_parts])
                order_proc_time = max(1.0, order_perim / self.CUTTING_SPEED)

                diff = finish_time - due
                abs_diff = abs(diff)

                # 梯形窗口
                ratio = abs_diff / order_proc_time
                R_FREE, R_FULL = 0.025, 0.075
                if ratio <= R_FREE:
                    coef = 0.0
                elif ratio <= R_FULL:
                    coef = (ratio - R_FREE) / (R_FULL - R_FREE)
                else:
                    coef = 1.0

                val_weight = order_value

                if diff > 0:
                    cost_jit += val_weight * (self.COST_TARD * coef) * abs_diff
                    total_delay += diff
                else:
                    cost_jit += val_weight * (self.COST_HOLD * coef) * abs_diff

            total_cost = cost_material + cost_jit


            total_part_area = sum([p['area'] for p in self.parts_pool])
            baseline_cost = total_part_area * self.COST_MAT
            if baseline_cost <= 0: baseline_cost = 1.0


            scaled_reward = - (total_cost / baseline_cost) * 10.0

            scaled_reward = np.clip(scaled_reward, -50.0, 50.0)

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
            best_task_idx, best_mach_idx = -1, -1

            if self.scheduler_model:
                MAX_SIM = self.max_sched_capacity


                safe_scale = max(1.0, self.episode_time_scale)
                m_feat = (mach_times - min_t) / safe_scale
                t_feat = []
                mask = np.zeros(MAX_SIM * 3, dtype=bool)
                has_v = False

                for i in range(MAX_SIM):
                    if i < len(tasks):
                        t = tasks[i]

                        norm_val = t['val'] / 140000.0
                        t_feat.extend([
                            t['cut'] / safe_scale,
                            (t['due'] - min_t) / safe_scale,
                            1.0 if t['done'] else 0.0,
                            norm_val
                        ])
                        if not t['done']: mask[i * 3:(i + 1) * 3] = True; has_v = True
                    else:
                        t_feat.extend([0, 0, 1, 0])

                if not has_v: mask = np.ones(MAX_SIM * 3, dtype=bool)

                obs = np.concatenate([m_feat, t_feat]).astype(np.float32)

                obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
                obs = np.clip(obs, -5.0, 5.0)

                action, _ = self.scheduler_model.predict(obs, action_masks=mask, deterministic=True)
                best_task_idx = int(action) // 3
                best_mach_idx = int(action) % 3
            else:
                # EDD 启发式
                min_cost = float('inf')
                for i, t in enumerate(tasks):
                    if t['done']: continue
                    for m in range(3):
                        end = mach_times[m] + t['cut']
                        diff = end - t['due']
                        cost = self.COST_TARD * diff if diff > 0 else self.COST_HOLD * abs(diff)
                        cost += (mach_times[m] - min_t) * 0.1
                        if cost < min_cost: min_cost, best_task_idx, best_mach_idx = cost, i, m

            if best_task_idx != -1:
                task = tasks[best_task_idx]
                task['done'] = True
                end_t = temp_sched.execute_assignment(best_mach_idx, task['cut'], task['idx'])
                for oid in task['oids']: self.orders[oid]['finished_time'] = max(self.orders[oid]['finished_time'],
                                                                                 end_t)

        results = {}
        for oid, order in self.orders.items(): results[oid] = order['finished_time']
        return results

    def _generate_random_orders(self):
        #非均匀分布生成
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

            #  0: 混合均匀 (Standard)1: 大件主导 (Big Items) 2: 长条主导 (Long Strips) 3: 碎件主导 (Small Fragments) - 填缝
            order_type = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])

            for _ in range(batch):
                if order_type == 1:  # Big
                    w_r = np.random.uniform(0.3, 0.6)
                    h_r = np.random.uniform(0.3, 0.6)
                elif order_type == 2:  # Long
                    if np.random.rand() > 0.5:
                        w_r, h_r = np.random.uniform(0.6, 0.9), np.random.uniform(0.1, 0.2)
                    else:
                        w_r, h_r = np.random.uniform(0.1, 0.2), np.random.uniform(0.6, 0.9)
                elif order_type == 3:  # Small
                    w_r = np.random.uniform(0.05, 0.2)
                    h_r = np.random.uniform(0.05, 0.2)
                else:  # Standard
                    w_r = np.random.uniform(0.1, 0.4)
                    h_r = np.random.uniform(0.1, 0.4)

                w = int(w_r * self.plate_w)
                h = int(h_r * self.plate_h)
                w, h = max(1, w), max(1, h)

                # 随机翻转
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

        # 修正利用率 (剔除最差板)
        if self.history_plates:
            utils = [p.utilization for p in self.history_plates]
            m['raw_avg_utilization'] = np.mean(utils)
            if len(utils) > 1:
                m['adj_avg_utilization'] = np.mean(sorted(utils)[1:])
            else:
                m['adj_avg_utilization'] = np.mean(utils)
        else:
            m['raw_avg_utilization'] = 0.0
            m['adj_avg_utilization'] = 0.0

        return m

    def _get_obs(self):

        # 初始化
        obs = np.zeros((self.max_capacity, self.obs_feature_dim), dtype=np.float32)
        skyline_feat = np.zeros(self.skyline_bins, dtype=np.float32)

        act_util_avg = 0.0
        act_util_max = 0.0
        act_util_min = 0.0

        act_free_area_ratio = 0.0
        act_max_free_area = 0.0
        max_free_w = 0.0
        max_free_h = 0.0

        if self.active_plates:
            target_plate = self.active_plates[-1]
            skyline_feat = target_plate.get_normalized_skyline(self.skyline_bins)


            utils = [p.utilization for p in self.active_plates]
            act_util_avg = np.mean(utils)
            act_util_max = np.max(utils)
            act_util_min = np.min(utils)

            all_free_rects = []
            for p in self.active_plates:
                all_free_rects.extend(p.free_rects)

            total_plate_area = self.plate_w * self.plate_h
            total_active_area = total_plate_area * len(self.active_plates)

            if all_free_rects and total_active_area > 0:

                total_free_area = sum([r[2] * r[3] for r in all_free_rects])
                act_free_area_ratio = total_free_area / total_active_area


                max_free_area_val = max([r[2] * r[3] for r in all_free_rects])
                act_max_free_area = max_free_area_val / total_plate_area


                max_free_w = max([r[2] for r in all_free_rects]) / self.plate_w
                max_free_h = max([r[3] for r in all_free_rects]) / self.plate_h


        act_cnt_norm = len(self.active_plates) / 10.0



        rem_idxs = [i for i in range(self.current_num_parts) if i not in self.packed_indices]

        mach_times = self.scheduler_state_machine.get_state()
        curr_time = np.min(mach_times)

        # 归一化分母
        total_plate_area_unit = self.plate_w * self.plate_h
        safe_time_scale = max(1.0, self.episode_time_scale)

        if rem_idxs:
            rem_areas = [self.parts_pool[i]['area'] for i in rem_idxs]
            rem_dues = [self.parts_pool[i]['due_date'] for i in rem_idxs]


            rem_avg_area = np.mean(rem_areas) / total_plate_area_unit
            rem_max_area = np.max(rem_areas) / total_plate_area_unit

            rem_total_ratio = (sum(rem_areas) / total_plate_area_unit) / 10.0

            rem_due_diffs = [(d - curr_time) / safe_time_scale for d in rem_dues]
            rem_avg_due = np.mean(rem_due_diffs)
            rem_min_due = np.min(rem_due_diffs)
            rem_due_std = np.std(rem_due_diffs)
        else:
            rem_avg_area = 0.0
            rem_max_area = 0.0
            rem_total_ratio = 0.0
            rem_avg_due = 0.0
            rem_min_due = 0.0
            rem_due_std = 0.0

        progress = len(self.packed_indices) / max(1, self.current_num_parts)


        if self.scheduler_model:
            m_rel = (mach_times - np.min(mach_times)) / safe_time_scale
            mach_load_avg = np.mean(m_rel)
            mach_load_std = np.std(m_rel)
        else:
            mach_load_avg = 0.0
            mach_load_std = 0.0

        # 全局特征向量
        global_feats = [
            progress,  # 1
            rem_avg_area, rem_max_area, rem_total_ratio,  # 3
            rem_avg_due, rem_min_due, rem_due_std,  # 3
            act_cnt_norm, act_util_avg, act_util_max, act_util_min,  # 4
            act_free_area_ratio, act_max_free_area, max_free_w, max_free_h,  # 4
            mach_load_avg, mach_load_std  # 2
        ]


        skyline_list = skyline_feat.tolist()


        for i in range(self.max_capacity):

            if i < self.current_num_parts:
                part = self.parts_pool[i]
                is_packed = 1.0 if i in self.packed_indices else 0.0


                norm_w = part['w'] / self.plate_w
                norm_h = part['h'] / self.plate_h
                norm_area = part['area'] / total_plate_area_unit
                rel_due = (part['due_date'] - curr_time) / safe_time_scale

                local_feats = [norm_w, norm_h, norm_area, rel_due, is_packed]


                full_feat = np.array(local_feats + global_feats + skyline_list, dtype=np.float32)

                obs[i] = full_feat
            else:

                pass

        return obs.flatten()

    def _get_action_mask(self):
        if len(self.packed_indices) >= self.current_num_parts: return np.ones(self.max_capacity * 6, dtype=bool)
        mask = np.ones(self.max_capacity * 6, dtype=bool)
        mask[self.current_num_parts * 6:] = False
        for i in range(self.current_num_parts):
            if i in self.packed_indices: mask[i * 6:(i + 1) * 6] = False
        return mask