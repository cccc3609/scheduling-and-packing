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

        # === 1. 统一物理与经济参数 ===
        self.CUTTING_SPEED = COST_CONFIG['cutting_speed']
        self.COST_MAT = COST_CONFIG['cost_material']
        self.COST_HOLD = COST_CONFIG['cost_earliness']  # Alpha
        self.COST_TARD = COST_CONFIG['cost_tardiness']  # Beta

        # 物理尺寸
        self.fixed_w, self.fixed_h = plate_size
        self.plate_w = 200
        self.plate_h = 200

        # 动态归一化因子
        self.episode_time_scale = 100.0
        self.baseline_cost = 1.0  # 统一的归一化分母

        # === 2. 辅助权重 ===
        self.w_grouping = 0.5
        self.w_step_compact = 0.5
        self.w_new_plate = 1.0

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

        self.last_norm_reward = 0.0

    def set_scheduling_partner(self, model):
        self.scheduler_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 动态参数配置
        if options and 'num_parts' in options:
            self.current_num_parts = options['num_parts']
        else:
            self.current_num_parts = np.random.randint(TRAIN_CONFIG['min_parts'], TRAIN_CONFIG['max_parts'] + 1)
        if self.current_num_parts > self.max_capacity: self.current_num_parts = self.max_capacity

        if options and 'plate_size' in options:
            self.plate_w, self.plate_h = options['plate_size']
        else:
            self.plate_w = np.random.randint(TRAIN_CONFIG['min_plate_dim'], TRAIN_CONFIG['max_plate_dim'] + 1)
            self.plate_h = np.random.randint(TRAIN_CONFIG['min_plate_dim'], TRAIN_CONFIG['max_plate_dim'] + 1)

        # 生成数据
        self.parts_pool, self.orders = self._generate_random_orders()

        # 计算本局时间标尺
        total_perimeter = sum([2 * (p['w'] + p['h']) for p in self.parts_pool])
        self.episode_time_scale = max(10.0, total_perimeter / self.CUTTING_SPEED)

        # 🟢 计算统一基准成本 (Baseline Cost)
        # 基准 = (所有零件面积 * 材料单价) + (所有零件面积 * 预期基础拖期 * 拖期费率)
        # 这里的"预期基础拖期"设为 100 单位时间，用于平衡量级
        total_parts_area = sum([p['area'] for p in self.parts_pool])
        base_mat_cost = total_parts_area * self.COST_MAT
        base_jit_risk = total_parts_area * self.COST_TARD * 100.0
        self.baseline_cost = max(1.0, base_mat_cost + base_jit_risk)

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
            return self._get_obs(), -10.0, True, False, {}

        part = self.parts_pool[part_index]
        self.packed_indices.add(part_index)
        reward = 0.0

        # === Best Fit (Heuristic) ===
        best_idx, best_score = -1, -float('inf')
        part_due = part['due_date']
        is_small = (part['area'] / (self.plate_w * self.plate_h)) < 0.05

        for idx, plate in enumerate(self.active_plates):
            sim = copy.deepcopy(plate)
            s, _, _, _, _, _ = sim.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if s:
                geo = sim.utilization
                time_pen = 0.0
                if not is_small:
                    oids = list(set([int(p[4]) for p in plate.placed_parts]))
                    if oids:
                        avg = np.mean([self.orders[o]['due_date'] for o in oids])
                        time_pen = abs(part_due - avg) / self.episode_time_scale

                score = geo - (self.w_grouping * time_pen)
                if score > best_score: best_score, best_idx = score, idx

        # === Execution ===
        if best_idx != -1:
            target = self.active_plates[best_idx]
            s, x, y, w, h, _ = target.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if not s: s, x, y, w, h, _ = target.place_part(part['w'], part['h'], part['order_id'], 2)

            reward += 0.05

            # Compactness
            cx, cy = x + w / 2, y + h / 2
            count = len(target.placed_parts)
            max_d = (self.plate_w ** 2 + self.plate_h ** 2) ** 0.5
            if count > 1:
                xs = [p[0] + p[2] / 2 for p in target.placed_parts]
                ys = [p[1] + p[3] / 2 for p in target.placed_parts]
                dist = ((cx - np.mean(xs)) ** 2 + (cy - np.mean(ys)) ** 2) ** 0.5
                reward += (1.0 - dist / max_d) * self.w_step_compact
            else:
                dist = (cx ** 2 + cy ** 2) ** 0.5
                reward += (1.0 - dist / max_d) * self.w_step_compact
        else:
            new_p = PlateLayoutManager(self.plate_w, self.plate_h)
            s, x, y, w, h, _ = new_p.place_part(part['w'], part['h'], part['order_id'], strategy_id)
            if s:
                self.active_plates.append(new_p)
                reward -= self.w_new_plate  # 开板惩罚
            else:
                reward -= 50.0

        terminated = len(self.packed_indices) == self.current_num_parts
        info = {}

        if terminated:
            self.history_plates = self.active_plates
            final_plates = [p for p in self.history_plates if len(p.placed_parts) > 0]

            # === 💰 成本结算 ===
            # 1. 材料成本
            consumed_area = len(final_plates) * (self.plate_w * self.plate_h)
            cost_material = consumed_area * self.COST_MAT

            # 2. JIT 成本 (全量仿真)
            order_finishes = self._simulate_batch_scheduling_detailed(final_plates)
            cost_jit = 0.0
            total_delay = 0.0

            for oid, finish_time in order_finishes.items():
                due = self.orders[oid]['due_date']
                # 🟢 订单价值 = 订单内所有零件面积之和
                order_parts = [p for p in self.parts_pool if p['order_id'] == oid]
                order_value = sum([p['area'] for p in order_parts])

                diff = finish_time - due
                if diff > 0:
                    cost_jit += (order_value * self.COST_TARD * diff)
                    total_delay += diff
                else:
                    cost_jit += (order_value * self.COST_HOLD * abs(diff))

            total_cost = cost_material + cost_jit

            # 3. 归一化奖励
            # Reward = - (Total Cost / Baseline Cost) * 10
            scaled_reward = - (total_cost / self.baseline_cost) * 10.0
            reward += scaled_reward

            # 记录
            total_part_area = sum([p['area'] for p in self.parts_pool])
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
            cut = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / self.CUTTING_SPEED
            oids = list(set([int(p[4]) for p in plate.placed_parts]))
            min_due = min([self.orders[o]['due_date'] for o in oids]) if oids else 999.0
            # 计算板材价值 (有效面积)
            val = sum([p[2] * p[3] for p in plate.placed_parts])
            tasks.append({'cut': cut, 'due': min_due, 'idx': idx, 'oids': oids, 'val': val, 'done': False})

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
                        # 🟢 对齐调度器输入: [Cut, RelDue, Done, NormVal]
                        # NormVal 归一化: val / (350*350) 约 120000
                        norm_val = t['val'] / 120000.0
                        t_feat.extend([
                            t['cut'] / self.episode_time_scale,
                            (t['due'] - min_t) / self.episode_time_scale,
                            1.0 if t['done'] else 0.0,
                            norm_val
                        ])
                        if not t['done']: mask[i * 3:(i + 1) * 3] = True; has_v = True
                    else:
                        t_feat.extend([0, 0, 1, 0])
                if not has_v: mask = np.ones(MAX_SIM * 3, dtype=bool)

                obs = np.concatenate([m_feat, t_feat]).astype(np.float32)
                # 数值保护
                obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
                obs = np.clip(obs, -5.0, 5.0)

                action, _ = self.scheduler_model.predict(obs, action_masks=mask, deterministic=True)
                best_task = int(action) // 3
                best_mach = int(action) % 3
            else:
                # 启发式：基于 Cost
                min_c = float('inf')
                for i, t in enumerate(tasks):
                    if t['done']: continue
                    for m in range(3):
                        end = mach_times[m] + t['cut']
                        diff = end - t['due']
                        # Cost = Value * Rate * Time
                        c = (t['val'] * self.COST_TARD * diff) if diff > 0 else (t['val'] * self.COST_HOLD * abs(diff))
                        c += (mach_times[m] - min_t) * 0.1
                        if c < min_c: min_c, best_task, best_mach = c, i, m

            if best_task != -1:
                t = tasks[best_task];
                t['done'] = True
                end = temp_sched.execute_assignment(best_mach, t['cut'], t['idx'])
                for oid in t['oids']: self.orders[oid]['finished_time'] = max(self.orders[oid]['finished_time'], end)

        results = {}
        for oid, order in self.orders.items(): results[oid] = order['finished_time']
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