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

新增修复：
  8. 非法动作惩罚 -10 → -2，避免 value head 初期发散
  9. 增加负载均衡步奖励信号
  10. nesting rollout 使用 predictor.reset_cache() 避免重复编码
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from config import MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, TRAIN_CONFIG
from core.cost import GlobalCostFunction
from core.scheduling_problem import SchedulingProblem


class SchedulingEnv(gym.Env):

    def __init__(self, num_machines=3, max_tasks=None):
        super().__init__()
        self.num_machines = num_machines
        self.max_tasks = max_tasks if max_tasks is not None else MAX_SCHED_TASKS_CAPACITY

        self.total_actions = self.max_tasks * num_machines
        self.action_space = spaces.Discrete(self.total_actions)

        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']
        self.global_cost_fn = GlobalCostFunction(
            cost_hold=self.COST_HOLD,
            cost_tard=self.COST_TARD,
        )

        self.episode_time_scale = 100.0
        self.baseline_cost = 1.0
        self.plate_w = 200
        self.plate_h = 200
        self.plate_area = 40000.0

        self.task_pool = []
        self.machine_times = np.zeros(num_machines)
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.orders_snapshot = {}

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.machine_times  = np.zeros(self.num_machines)
        self.task_pool      = []
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.episode_time_scale = 100.0
        self.baseline_cost  = 100.0
        self.plate_area     = 40000.0
        options = options or {}
        problem = options.get("problem")
        if not isinstance(problem, SchedulingProblem):
            raise ValueError("SchedulingEnv.reset requires options={'problem': SchedulingProblem}")
        if problem.num_machines != self.num_machines:
            raise ValueError("SchedulingProblem.num_machines does not match SchedulingEnv")
        if len(problem.tasks) > self.max_tasks:
            raise ValueError(
                f"SchedulingProblem has {len(problem.tasks)} tasks, exceeds max_tasks={self.max_tasks}")
        context = np.asarray(options.get("nesting_context", np.zeros(self.COMM_DIM_IN)), dtype=np.float32)
        if context.shape != (self.COMM_DIM_IN,):
            raise ValueError(f"nesting_context must have shape ({self.COMM_DIM_IN},)")
        self.nesting_result_vec = context.copy()
        self.episode_time_scale = float(problem.episode_time_scale)
        self.plate_w, self.plate_h = problem.plate_w, problem.plate_h
        self.plate_area = max(1.0, float(self.plate_w * self.plate_h))
        self.orders_snapshot = {
            order.order_id: {
                "due_date": order.due_date,
                "total_area": order.total_area,
                "proc_time": order.proc_time,
                "finished_time": 0.0,
            }
            for order in problem.orders
        }
        self.baseline_cost = max(
            1.0, sum(order.total_area for order in problem.orders) * self.COST_TARD * 100.0)
        self.task_pool = [
            {"cut": task.processing_time, "due": task.due_date,
             "oids": list(task.order_ids), "val": task.plate_part_area}
            for task in problem.tasks
        ]

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _get_obs(self):
        min_t = float(np.min(self.machine_times))
        scale = max(10.0, self.episode_time_scale)
        m_feat = (self.machine_times - min_t) / scale

        total_work = sum(task['cut'] for task in self.task_pool)
        upstream_workload = total_work / max(1.0, self.num_machines * scale)
        upstream_urgent_due = (
            min((task['due'] for task in self.task_pool), default=min_t) - min_t
        ) / scale
        upstream_giant_ratio = (
            sum(task['val'] > self.plate_area / 4.0 for task in self.task_pool)
            / max(1, len(self.task_pool))
        )
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

        # 修复：非法动作惩罚从 -10 降为 -2
        if t_idx >= len(self.task_pool) or self.scheduled_mask[t_idx]:
            return self._get_obs(), -2.0, True, False, {}

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

        # ── 步奖励：负载均衡 + 交期感知 ──
        scale = max(10.0, self.episode_time_scale)
        # 1) 机器负载均衡：std 越小越好
        reward = -float(np.std(self.machine_times)) / scale * 0.5

        # 2) 选择最空闲机器的奖励（鼓励负载均衡）
        if m_idx == int(np.argmin(self.machine_times)):
            reward += 0.1

        valid  = len(self.task_pool)
        done   = bool(np.sum(self.scheduled_mask[:valid]) == valid)

        if done:
            jit_cost = sum(
                self.global_cost_fn.order_jit_cost(
                    oid, order['finished_time'], self.orders_snapshot)
                for oid, order in self.orders_snapshot.items()
            )
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
