import gymnasium as gym
import numpy as np
from gymnasium import spaces
import copy
from config import MAX_SCHED_TASKS_CAPACITY, COST_CONFIG


class SchedulingEnv(gym.Env):
    def __init__(self, num_machines=3, max_tasks=None):
        super().__init__()
        self.num_machines = num_machines
        self.max_tasks = max_tasks if max_tasks is not None else MAX_SCHED_TASKS_CAPACITY

        self.total_actions = self.max_tasks * num_machines
        self.action_space = spaces.Discrete(self.total_actions)

        dim = num_machines + (self.max_tasks * 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']

        self.episode_time_scale = 100.0
        self.baseline_cost = 1.0
        # 初始化默认值
        self.plate_w = 200
        self.plate_h = 200
        self.plate_area = 40000.0

        self.task_pool = []
        self.machine_times = np.zeros(num_machines)
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.orders_snapshot = {}
        self.nesting_env = None
        self.nesting_model = None

    def set_nesting_partner(self, env, model):
        self.nesting_env = env
        self.nesting_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.machine_times = np.zeros(self.num_machines)
        self.task_pool = []
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)

        # 兜底默认值
        self.episode_time_scale = 100.0
        self.baseline_cost = 100.0
        self.plate_area = 40000.0

        if self.nesting_env:
            obs, _ = self.nesting_env.reset(seed=seed)
            # 同步
            self.episode_time_scale = max(10.0, float(self.nesting_env.unwrapped.episode_time_scale))
            real = self.nesting_env.unwrapped
            self.plate_w = real.plate_w
            self.plate_h = real.plate_h
            self.plate_area = max(1.0, float(self.plate_w * self.plate_h))

            done = False
            self.orders_snapshot = copy.deepcopy(real.orders)

            # 预计算订单属性
            total_all_area = 0.0
            for oid in self.orders_snapshot:
                parts = [p for p in real.parts_pool if p['order_id'] == oid]
                area = sum([p['area'] for p in parts])
                self.orders_snapshot[oid]['total_area'] = area

                perimeter = sum([2 * (p['w'] + p['h']) for p in parts])
                speed = max(0.1, real.CUTTING_SPEED)
                self.orders_snapshot[oid]['proc_time'] = max(1.0, perimeter / speed)

                self.orders_snapshot[oid]['finished_time'] = 0.0
                total_all_area += area

            self.baseline_cost = max(1.0, total_all_area * self.COST_TARD * 100.0)

            while not done:
                try:
                    m = self.nesting_env.action_masks()
                except:
                    m = real._get_action_mask()
                a, _ = self.nesting_model.predict(obs, action_masks=m, deterministic=True)
                obs, _, done, _, _ = self.nesting_env.step(a)

            speed = max(0.1, real.CUTTING_SPEED)
            for i, plate in enumerate(real.history_plates):
                if i >= self.max_tasks: break
                cut = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / speed
                oids = list(set([int(p[4]) for p in plate.placed_parts]))
                val = sum([p[2] * p[3] for p in plate.placed_parts])

                due = 9999.0
                if oids:
                    valid_dues = [self.orders_snapshot[o]['due_date'] for o in oids if o in self.orders_snapshot]
                    if valid_dues: due = min(valid_dues)

                self.task_pool.append({'cut': cut, 'due': due, 'oids': oids, 'val': val})
        else:
            self.task_pool = [{'cut': 10, 'due': 100, 'oids': [], 'val': 100}]

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _get_obs(self):
        min_t = np.min(self.machine_times)
        scale = max(10.0, self.episode_time_scale)
        m_feat = (self.machine_times - min_t) / scale

        t_feat = []
        for i in range(self.max_tasks):
            if i < len(self.task_pool):
                t = self.task_pool[i]
                d = 1.0 if self.scheduled_mask[i] else 0.0

                norm_cut = t['cut'] / scale
                norm_due = (t['due'] - min_t) / scale
                norm_val = t['val'] / 140000.0

                t_feat.extend([norm_cut, norm_due, d, norm_val])
            else:
                t_feat.extend([0.0, 0.0, 1.0, 0.0])

        obs = np.concatenate([m_feat, t_feat]).astype(np.float32)
        # 终极防爆
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -5.0, 5.0)
        return obs

    def step(self, action):
        action = int(action)
        t_idx = action // self.num_machines
        m_idx = action % self.num_machines

        if t_idx >= len(self.task_pool) or self.scheduled_mask[t_idx]:
            return self._get_obs(), -10.0, True, False, {}

        task = self.task_pool[t_idx]
        curr_machine_time = self.machine_times[m_idx]

        # === 🟢 [JIT 核心修改] Lazy Start 策略 ===
        # 计算为了准时交货，最晚应该什么时候开始
        # 如果任务截止期是100，工时10，那么最晚90开始
        lazy_start = task['due'] - task['cut']

        # 实际开始时间 = max(机器空闲时间, 最晚开始时间)
        # 如果机器空闲时间(50) < 最晚开始时间(90)，机器会"休息"到90再开工
        # 如果机器空闲时间(95) > 最晚开始时间(90)，机器只能立即开工(95)，此时会延期
        start = max(curr_machine_time, lazy_start)

        end = start + task['cut']
        self.machine_times[m_idx] = end
        self.scheduled_mask[t_idx] = True

        for oid in task['oids']:
            if oid in self.orders_snapshot:
                self.orders_snapshot[oid]['finished_time'] = max(
                    self.orders_snapshot[oid]['finished_time'], end
                )

        bal = np.std(self.machine_times) * 0.01
        reward = -bal

        valid = len(self.task_pool)
        done = (np.sum(self.scheduled_mask[:valid]) == valid)

        if done:
            jit_cost = 0.0
            for oid, order in self.orders_snapshot.items():
                fin = order['finished_time']
                due = order['due_date']
                order_area = order.get('total_area', 1.0)
                order_proc_time = order.get('proc_time', 1.0)

                diff = fin - due
                ratio = abs(diff) / order_proc_time

                # 梯形窗口 (保持原逻辑)
                R_FREE = 0.025
                R_FULL = 0.075
                if ratio <= R_FREE:
                    coef = 0.0
                elif ratio <= R_FULL:
                    coef = (ratio - R_FREE) / (R_FULL - R_FREE)
                else:
                    coef = 1.0

                rate = self.COST_TARD if diff > 0 else self.COST_HOLD
                val_weight = order_area / self.plate_area
                jit_cost += val_weight * (rate * coef) * abs(diff)

            # 归一化并截断
            norm_reward = - (jit_cost / self.baseline_cost) * 10.0
            norm_reward = np.clip(norm_reward, -20.0, 20.0)
            reward += norm_reward

        return self._get_obs(), reward, done, False, {}

    def _get_action_mask(self):
        valid = len(self.task_pool)
        if valid == 0 or np.all(self.scheduled_mask[:valid]):
            return np.ones(self.total_actions, dtype=bool)

        mask = np.zeros(self.total_actions, dtype=bool)
        has_v = False
        for i in range(valid):
            if not self.scheduled_mask[i]:
                mask[i * self.num_machines: (i + 1) * self.num_machines] = True
                has_v = True

        if not has_v: return np.ones(self.total_actions, dtype=bool)
        return mask