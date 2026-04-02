"""
envs/scheduling_env.py

修复清单：
  1. COMM_DIM_IN 先用后定义 → 调整顺序
  2. reset() 中 list_of_parts / cutting_speed 未定义 → 改为 real.parts_pool / real.CUTTING_SPEED
  3. intent 生成位置错误（在 rollout 后）→ 移到 rollout 开始前
  4. hasattr(real,...) 在 else 分支里 real 未定义 → 移入 if 分支
  5. reset() 末尾多余的 obs 拼接块 → 删除
  6. step() 中 import numpy as _np → 改用顶层 np
  7. pk_sched item_dim 不匹配 → scheduling obs 是任务序列，item_dim=4，global_prefix=14
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import copy
from config import MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, TRAIN_CONFIG
from models.comm_encoders import SchedulingIntentEncoder


class SchedulingEnv(gym.Env):

    def __init__(self, num_machines=3, max_tasks=None):
        super().__init__()
        self.num_machines = num_machines
        self.max_tasks = max_tasks if max_tasks is not None else MAX_SCHED_TASKS_CAPACITY

        self.total_actions = self.max_tasks * num_machines
        self.action_space = spaces.Discrete(self.total_actions)

        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']

        self.episode_time_scale = 100.0
        self.baseline_cost = 1.0
        self.plate_w = 200
        self.plate_h = 200
        self.plate_area = 40000.0

        self.task_pool = []
        self.machine_times = np.zeros(num_machines)
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.orders_snapshot = {}
        self.nesting_env   = None
        self.nesting_model = None

        _max_dim = TRAIN_CONFIG.get('max_plate_dim', 300)
        self.val_norm = float(_max_dim * _max_dim)

        # 通信向量（先定义，后使用）
        self.COMM_DIM_IN  = 8    # 接收排样结果摘要
        self.COMM_DIM_OUT = 16   # 输出调度意图向量
        self.nesting_result_vec = np.zeros(self.COMM_DIM_IN, dtype=np.float32)

        # obs 维度：
        #   global_prefix = machines(3) + upstream(3) + nesting_result(8) = 14
        #   tasks         = max_tasks × 4
        # AttentionFeatureExtractor 的 global_prefix_dim=14，item_dim=4
        dim = num_machines + (self.max_tasks * 4) + 3 + self.COMM_DIM_IN
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

        self.intent_encoder = SchedulingIntentEncoder(input_dim=12, comm_dim=16)

    def set_nesting_partner(self, env, model):
        self.nesting_env   = env
        self.nesting_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.machine_times  = np.zeros(self.num_machines)
        self.task_pool      = []
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.episode_time_scale = 100.0
        self.baseline_cost  = 100.0
        self.plate_area     = 40000.0
        self.nesting_result_vec = np.zeros(self.COMM_DIM_IN, dtype=np.float32)

        if self.nesting_env:
            real = self.nesting_env.unwrapped

            # ── 先 reset nesting env，拿到订单数据 ──
            obs, _ = self.nesting_env.reset(seed=seed)
            self.episode_time_scale = max(10.0, float(real.episode_time_scale))
            self.plate_w    = real.plate_w
            self.plate_h    = real.plate_h
            self.plate_area = max(1.0, float(self.plate_w * self.plate_h))

            self.orders_snapshot = copy.deepcopy(real.orders)
            total_all_area = 0.0
            for oid in self.orders_snapshot:
                parts = [p for p in real.parts_pool if p['order_id'] == oid]
                area  = sum(p['area'] for p in parts)
                speed = max(0.1, real.CUTTING_SPEED)
                self.orders_snapshot[oid]['total_area'] = area
                self.orders_snapshot[oid]['proc_time']  = max(
                    1.0, sum(2*(p['w']+p['h']) for p in parts) / speed)
                self.orders_snapshot[oid]['finished_time'] = 0.0
                total_all_area += area
            self.baseline_cost = max(1.0, total_all_area * self.COST_TARD * 100.0)

            # 修复1：intent 生成在 rollout 开始前，变量名用 real.parts_pool / real.CUTTING_SPEED
            intent_vec = self.intent_encoder.encode_orders(
                self.orders_snapshot,
                real.parts_pool,        # 原代码：list_of_parts（未定义）
                self.num_machines,
                real.CUTTING_SPEED      # 原代码：cutting_speed（未定义）
            )
            if hasattr(real, 'set_scheduling_intent'):
                real.set_scheduling_intent(intent_vec)

            # ── nesting rollout ──
            done = False
            while not done:
                try:
                    m = self.nesting_env.action_masks()
                except Exception:
                    m = real._get_action_mask()
                a, _ = self.nesting_model.predict(obs, action_masks=m, deterministic=True)
                obs, _, done, _, _ = self.nesting_env.step(a)

            # 修复2：在 rollout 完成后读取排样摘要（real 在 if 分支内，有效）
            if hasattr(real, 'nesting_result_vec'):
                self.nesting_result_vec = real.nesting_result_vec.copy()

            speed = max(0.1, real.CUTTING_SPEED)
            for i, plate in enumerate(real.history_plates):
                if i >= self.max_tasks:
                    break
                cut  = sum(2*(p[2]+p[3]) for p in plate.placed_parts) / speed
                oids = list(set(int(p[4]) for p in plate.placed_parts))
                val  = sum(p[2]*p[3] for p in plate.placed_parts)
                due  = min((self.orders_snapshot[o]['due_date']
                            for o in oids if o in self.orders_snapshot),
                           default=9999.0)
                self.task_pool.append({'cut': cut, 'due': due, 'oids': oids, 'val': val})
        else:
            self.task_pool = [{'cut': 10, 'due': 100, 'oids': [], 'val': 100}]
            # else 分支里 real 不存在，nesting_result_vec 已在方法开头清零

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _get_obs(self):
        min_t = float(np.min(self.machine_times))
        scale = max(10.0, self.episode_time_scale)
        m_feat = (self.machine_times - min_t) / scale

        upstream_workload = upstream_urgent_due = upstream_giant_ratio = 0.0
        if self.nesting_env:
            real_nest = self.nesting_env.unwrapped
            rem = [i for i in range(real_nest.current_num_parts)
                   if i not in real_nest.packed_indices]
            if rem:
                rp    = [real_nest.parts_pool[i] for i in rem]
                spd   = max(0.1, real_nest.CUTTING_SPEED)
                upstream_workload  = (sum(2*(p['w']+p['h']) for p in rp) / spd) / (self.num_machines * scale)
                upstream_urgent_due = (min(p['due_date'] for p in rp) - min_t) / scale
                giant = sum(1 for p in rp if p['area'] > (real_nest.plate_w * real_nest.plate_h) / 4.0)
                upstream_giant_ratio = giant / len(rp)
        upstream_feats = [upstream_workload, upstream_urgent_due, upstream_giant_ratio]

        t_feat = []
        for i in range(self.max_tasks):
            if i < len(self.task_pool):
                t = self.task_pool[i]
                t_feat.extend([t['cut']/scale, (t['due']-min_t)/scale,
                               1.0 if self.scheduled_mask[i] else 0.0,
                               t['val'] / self.val_norm])
            else:
                t_feat.extend([0.0, 0.0, 1.0, 0.0])

        obs = np.concatenate([m_feat, upstream_feats, t_feat, self.nesting_result_vec]).astype(np.float32)
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def step(self, action):
        action = int(action)
        t_idx, m_idx = action // self.num_machines, action % self.num_machines

        if t_idx >= len(self.task_pool) or self.scheduled_mask[t_idx]:
            return self._get_obs(), -10.0, True, False, {}

        task = self.task_pool[t_idx]
        curr = self.machine_times[m_idx]

        # lazy start 仅在所有机器繁忙时生效
        min_t    = np.min(self.machine_times)
        all_busy = bool(np.all(self.machine_times > min_t + 1e-6))
        start = max(curr, task['due'] - task['cut']) if all_busy else curr

        end = start + task['cut']
        self.machine_times[m_idx] = end
        self.scheduled_mask[t_idx] = True

        for oid in task['oids']:
            if oid in self.orders_snapshot:
                self.orders_snapshot[oid]['finished_time'] = max(
                    self.orders_snapshot[oid]['finished_time'], end)

        reward = -float(np.std(self.machine_times)) * 0.01
        valid  = len(self.task_pool)
        done   = bool(np.sum(self.scheduled_mask[:valid]) == valid)

        if done:
            jit_cost = 0.0
            for oid, order in self.orders_snapshot.items():
                diff  = order['finished_time'] - order['due_date']
                ratio = abs(diff) / order.get('proc_time', 1.0)
                coef  = 0.0 if ratio <= 0.025 else min(1.0, (ratio - 0.025) / 0.05)
                rate  = self.COST_TARD if diff > 0 else self.COST_HOLD
                jit_cost += order.get('total_area', 1.0) * (rate * coef) * abs(diff)
            reward += float(np.clip(-(jit_cost / self.baseline_cost) * 10.0, -20.0, 20.0))

        return self._get_obs(), reward, done, False, {}

    def _get_action_mask(self):
        valid = len(self.task_pool)
        if valid == 0 or bool(np.all(self.scheduled_mask[:valid])):
            return np.ones(self.total_actions, dtype=bool)
        mask  = np.zeros(self.total_actions, dtype=bool)
        has_v = False
        for i in range(valid):
            if not self.scheduled_mask[i]:
                mask[i * self.num_machines: (i + 1) * self.num_machines] = True
                has_v = True
        return mask if has_v else np.ones(self.total_actions, dtype=bool)