"""Scheduling environment with an explicit, shared observation contract."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from config import MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, NUM_MACHINES
from core.cost import GlobalCostFunction
from core.scheduling_observation import (
    SchedulingObservationLayout,
    SchedulingTaskFeature,
)
from core.scheduling_problem import SchedulingProblem


class SchedulingEnv(gym.Env):

    def __init__(self, num_machines=NUM_MACHINES, max_tasks=None):
        super().__init__()
        self.num_machines = num_machines
        self.max_tasks = max_tasks if max_tasks is not None else MAX_SCHED_TASKS_CAPACITY
        self.observation_layout = SchedulingObservationLayout(
            num_machines=self.num_machines, max_tasks=self.max_tasks)

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
        self.valid_task_mask = np.zeros(self.max_tasks, dtype=bool)
        self.orders_snapshot = {}
        self.order_to_task_indices = {}

        # 通信向量（先定义，后使用）
        self.COMM_DIM_IN  = self.observation_layout.context_dim
        self.COMM_DIM_OUT = 16   # 输出调度意图向量
        self.nesting_result_vec = np.zeros(self.COMM_DIM_IN, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_layout.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.machine_times  = np.zeros(self.num_machines)
        self.task_pool      = []
        self.scheduled_mask = np.zeros(self.max_tasks, dtype=bool)
        self.valid_task_mask = np.zeros(self.max_tasks, dtype=bool)
        self.order_to_task_indices = {}
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
        if len(problem.tasks) == 0:
            raise ValueError("SchedulingEnv requires at least one scheduling task")
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
        self.valid_task_mask[:len(self.task_pool)] = True
        order_to_tasks = {order_id: [] for order_id in self.orders_snapshot}
        for task_index, task in enumerate(self.task_pool):
            for order_id in task["oids"]:
                if order_id not in order_to_tasks:
                    raise ValueError(
                        f"Scheduling task references unknown order_id={order_id}")
                order_to_tasks[order_id].append(task_index)
        self.order_to_task_indices = {
            order_id: tuple(task_indices)
            for order_id, task_indices in order_to_tasks.items()
        }

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _get_obs(self):
        min_t = float(np.min(self.machine_times))
        scale = max(10.0, self.episode_time_scale)
        m_feat = (self.machine_times - min_t) / scale

        unscheduled_indices = [
            index for index in range(len(self.task_pool))
            if not self.scheduled_mask[index]
        ]
        if unscheduled_indices:
            unscheduled_tasks = [self.task_pool[index] for index in unscheduled_indices]
            remaining_workload = sum(task["cut"] for task in unscheduled_tasks)
            dynamic_globals = np.asarray([
                remaining_workload / (self.num_machines * scale),
                (min(task["due"] for task in unscheduled_tasks) - min_t) / scale,
                sum(task["val"] > self.plate_area / 4.0 for task in unscheduled_tasks)
                / len(unscheduled_tasks),
            ], dtype=np.float32)
        else:
            dynamic_globals = np.zeros(
                self.observation_layout.dynamic_global_dim, dtype=np.float32)

        remaining_counts = {}
        remaining_ratios = {}
        optimistic_slacks = {}
        for order_id, linked_indices in self.order_to_task_indices.items():
            remaining_indices = [
                index for index in linked_indices if not self.scheduled_mask[index]
            ]
            remaining_count = len(remaining_indices)
            remaining_counts[order_id] = remaining_count
            remaining_ratios[order_id] = (
                remaining_count / len(linked_indices) if linked_indices else 0.0)
            remaining_work = sum(
                self.task_pool[index]["cut"] for index in remaining_indices)
            order = self.orders_snapshot[order_id]
            optimistic_finish = max(
                order["finished_time"],
                min_t + remaining_work / self.num_machines,
            )
            optimistic_slacks[order_id] = order["due_date"] - optimistic_finish

        task_tokens = np.zeros(
            (self.max_tasks, self.observation_layout.task_dim), dtype=np.float32)
        feature = SchedulingTaskFeature
        total_orders = max(1, len(self.orders_snapshot))
        for task_index, task in enumerate(self.task_pool):
            linked_orders = task["oids"]
            token = task_tokens[task_index]
            token[feature.PROCESSING_TIME] = task["cut"] / scale
            token[feature.RELATIVE_DUE] = (task["due"] - min_t) / scale
            token[feature.PLATE_AREA_RATIO] = task["val"] / self.plate_area
            token[feature.LINKED_ORDER_RATIO] = len(linked_orders) / total_orders
            token[feature.SCHEDULED] = float(self.scheduled_mask[task_index])
            if linked_orders:
                ratios = [remaining_ratios[order_id] for order_id in linked_orders]
                token[feature.MEAN_REMAINING_RATIO] = float(np.mean(ratios))
                token[feature.MAX_REMAINING_RATIO] = float(np.max(ratios))
                if not self.scheduled_mask[task_index]:
                    token[feature.RELEASE_FRACTION] = (
                        sum(remaining_counts[order_id] == 1 for order_id in linked_orders)
                        / len(linked_orders)
                    )
                token[feature.WORST_SLACK] = (
                    min(optimistic_slacks[order_id] for order_id in linked_orders)
                    / scale
                )
            token[feature.VALID] = float(self.valid_task_mask[task_index])

        obs = np.concatenate([
            m_feat,
            dynamic_globals,
            self.nesting_result_vec,
            task_tokens.reshape(-1),
        ]).astype(np.float32)
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Scheduling action {action!r} is outside the action space")
        action = int(action)
        if not bool(self._get_action_mask()[action]):
            raise ValueError(
                f"Scheduling action {action} violates the current action mask")
        t_idx, m_idx = action // self.num_machines, action % self.num_machines

        task = self.task_pool[t_idx]
        curr = self.machine_times[m_idx]

        # Formal scheduling model: non-delay parallel-machine scheduling.
        start = curr

        end = start + task['cut']
        self.machine_times[m_idx] = end
        self.scheduled_mask[t_idx] = True

        for oid in task['oids']:
            if oid in self.orders_snapshot:
                self.orders_snapshot[oid]['finished_time'] = max(
                    self.orders_snapshot[oid]['finished_time'], end)

        # Scheduling PPO optimizes the formal order-level JIT proxy only.
        reward = 0.0

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
        selectable_tasks = self.valid_task_mask & ~self.scheduled_mask
        return np.repeat(selectable_tasks, self.num_machines)
