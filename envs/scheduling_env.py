import gymnasium as gym
import numpy as np
from gymnasium import spaces
import copy
from config import MAX_SCHED_TASKS_CAPACITY, COST_CONFIG


class SchedulingEnv(gym.Env):
    def __init__(self, num_machines=3, max_tasks=None):
        super().__init__()
        self.num_machines = num_machines
        # 优先使用传入的 max_tasks (兼容性)，否则使用 Config
        self.max_tasks = max_tasks if max_tasks is not None else MAX_SCHED_TASKS_CAPACITY

        self.total_actions = self.max_tasks * num_machines
        self.action_space = spaces.Discrete(self.total_actions)

        # 观察空间
        dim = num_machines + (self.max_tasks * 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

        # === 成本参数 ===
        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']

        self.episode_time_scale = 100.0
        self.baseline_cost = 1.0

        # 🟢 修复：在 init 中初始化 plate_area，防止 step 中除以 None 报错
        self.plate_w = 200
        self.plate_h = 200
        self.plate_area = 40000.0

        self.task_pool = []
        self.machine_times = np.zeros(num_machines)
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.orders_snapshot = {}

    def set_nesting_partner(self, env, model):
        self.nesting_env = env
        self.nesting_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.machine_times = np.zeros(self.num_machines)
        self.task_pool = []
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)

        if self.nesting_env:
            # 1. 运行排样环境生成数据
            obs, _ = self.nesting_env.reset(seed=seed)
            self.episode_time_scale = self.nesting_env.unwrapped.episode_time_scale

            # 🟢 修复：同步排样环境的板材尺寸
            real = self.nesting_env.unwrapped
            self.plate_w = real.plate_w
            self.plate_h = real.plate_h
            self.plate_area = self.plate_w * self.plate_h

            done = False
            self.orders_snapshot = copy.deepcopy(real.orders)

            # 🟢 预计算订单属性 (价值 & 物理工时)
            total_all_area = 0.0
            for oid in self.orders_snapshot:
                parts = [p for p in real.parts_pool if p['order_id'] == oid]

                # A. 价值 = 面积
                area = sum([p['area'] for p in parts])
                self.orders_snapshot[oid]['total_area'] = area / self.plate_area

                # B. 物理工时 = 周长 / 速度
                perimeter = sum([2 * (p['w'] + p['h']) for p in parts])
                speed = real.CUTTING_SPEED
                # 防止除以0
                self.orders_snapshot[oid]['proc_time'] = max(1.0, perimeter / speed)

                self.orders_snapshot[oid]['finished_time'] = 0.0
                total_all_area += area

            # 基准成本估算
            self.baseline_cost = max(1.0, total_all_area * self.COST_TARD * 100.0)

            while not done:
                try:
                    m = self.nesting_env.action_masks()
                except:
                    m = real._get_action_mask()
                a, _ = self.nesting_model.predict(obs, action_masks=m, deterministic=True)
                obs, _, done, _, _ = self.nesting_env.step(a)

            speed = real.CUTTING_SPEED
            for i, plate in enumerate(real.history_plates):
                if i >= self.max_tasks: break
                cut = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / speed
                oids = list(set([int(p[4]) for p in plate.placed_parts]))
                val = sum([p[2] * p[3] for p in plate.placed_parts])

                due = 999.0
                if oids:
                    valid = [o for o in oids if o in self.orders_snapshot]
                    if valid: due = min([self.orders_snapshot[o]['due_date'] for o in valid])

                self.task_pool.append({'cut': cut, 'due': due, 'oids': oids, 'val': val})
        else:
            # 🟢 兜底：如果没有排样环境 (测试模式)，给默认值
            self.task_pool = [{'cut': 10, 'due': 100, 'oids': [], 'val': 100}]
            self.baseline_cost = 100.0
            self.plate_area = 40000.0
            self.episode_time_scale = 100.0

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _get_obs(self):
        min_t = np.min(self.machine_times)
        scale = max(1.0, self.episode_time_scale)
        m_feat = (self.machine_times - min_t) / scale

        t_feat = []
        for i in range(self.max_tasks):
            if i < len(self.task_pool):
                t = self.task_pool[i]
                d = 1.0 if self.scheduled_mask[i] else 0.0
                # 板材价值归一化
                norm_val = t['val'] / self.plate_area
                t_feat.extend([t['cut'] / scale, (t['due'] - min_t) / scale, d, norm_val])
            else:
                t_feat.extend([0.0, 0.0, 1.0, 0.0])
        obs = np.concatenate([m_feat, t_feat]).astype(np.float32)
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def step(self, action):
        action = int(action)
        t_idx = action // self.num_machines
        m_idx = action % self.num_machines

        if t_idx >= len(self.task_pool) or self.scheduled_mask[t_idx]:
            return self._get_obs(), -1.0, True, False, {}

        task = self.task_pool[t_idx]
        start = self.machine_times[m_idx]
        end = start + task['cut']
        self.machine_times[m_idx] = end
        self.scheduled_mask[t_idx] = True

        # 更新订单
        for oid in task['oids']:
            if oid in self.orders_snapshot:
                self.orders_snapshot[oid]['finished_time'] = max(self.orders_snapshot[oid]['finished_time'], end)

        bal = np.std(self.machine_times) * 0.01
        reward = -bal

        valid = len(self.task_pool)
        done = (np.sum(self.scheduled_mask[:valid]) == valid)

        if done:
            # 🟢 结算: 梯形窗口 JIT 成本 (基于物理工时)
            jit_cost = 0.0

            for oid, order in self.orders_snapshot.items():
                fin = order['finished_time']
                due = order['due_date']

                order_area = order.get('total_area', 1.0)
                # B. 订单工时 (周长/速度)
                order_proc_time = order.get('proc_time', 1.0)

                # 1. 绝对时间偏差
                diff_time = fin - due

                # 2. 相对偏差比率
                ratio = abs(diff_time) / order_proc_time

                # 3. 梯形窗口系数
                R_FREE = 0.025  # ±2.5%
                R_FULL = 0.075  # ±7.5%

                if ratio <= R_FREE:
                    penalty_coef = 0.0
                elif ratio <= R_FULL:
                    penalty_coef = (ratio - R_FREE) / (R_FULL - R_FREE)
                else:
                    penalty_coef = 1.0

                # 费率选择
                base_rate = self.COST_TARD if diff_time > 0 else self.COST_HOLD

                # 4. 成本计算: 价值权重 * 费率 * 系数 * 时间
                val_weight = order_area / self.plate_area
                jit_cost += val_weight * (base_rate * penalty_coef) * abs(diff_time)

            # 归一化奖励
            norm_reward = - (jit_cost / self.baseline_cost) * 10.0
            norm_reward = max(-200.0, norm_reward)

            reward += norm_reward

        return self._get_obs(), reward, done, False, {}

    def _get_action_mask(self):
        valid = len(self.task_pool)
        if valid == 0 or np.all(self.scheduled_mask[:valid]): return np.ones(self.total_actions, dtype=bool)
        mask = np.zeros(self.total_actions, dtype=bool)
        for i in range(valid):
            if not self.scheduled_mask[i]: mask[i * self.num_machines:(i + 1) * self.num_machines] = True
        return mask if np.any(mask) else np.ones(self.total_actions, dtype=bool)