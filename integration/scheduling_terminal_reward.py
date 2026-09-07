"""Explicit terminal scheduling evaluation outside both base environments."""

import math

import gymnasium as gym
import numpy as np

from core.cooperative_context import build_nesting_context
from core.cost import GlobalCostFunction
from core.scheduling_problem import build_scheduling_problem
from envs.scheduling_env import SchedulingEnv
from heuristic.scheduler import SchedulerStateMachine


class SchedulingPolicyRolloutError(RuntimeError):
    pass


class SchedulingEvaluator:
    """Evaluate a scheduling problem in one explicitly selected mode."""

    VALID_MODES = frozenset({"edd", "policy"})

    def __init__(self, mode="policy", scheduling_policy=None, scheduling_env_factory=None):
        self.scheduling_env_factory = scheduling_env_factory or SchedulingEnv
        self.last_evaluator_id = None
        self.mode = None
        self.scheduling_policy = None
        self.set_mode(mode, scheduling_policy)

    def set_mode(self, mode, scheduling_policy=None):
        if mode not in self.VALID_MODES:
            raise ValueError(f"evaluation mode must be one of {sorted(self.VALID_MODES)}")
        if mode == "policy" and scheduling_policy is None:
            raise ValueError("evaluation_mode='policy' requires a scheduling policy")
        self.mode = mode
        self.scheduling_policy = scheduling_policy

    def evaluate(self, problem, nesting_context):
        if self.mode == "edd":
            return self._evaluate_edd(problem)
        return self._evaluate_policy(problem, nesting_context)

    @staticmethod
    def _evaluate_edd(problem):
        scheduler = SchedulerStateMachine(num_machines=problem.num_machines)
        finish_times = {order.order_id: 0.0 for order in problem.orders}
        for task in sorted(problem.tasks, key=lambda item: item.due_date):
            machine_idx = int(np.argmin(scheduler.get_state()))
            end_time = scheduler.execute_assignment(
                machine_idx, task.processing_time, task.plate_index)
            for order_id in task.order_ids:
                finish_times[order_id] = max(finish_times[order_id], end_time)
        return finish_times

    def _evaluate_policy(self, problem, nesting_context):
        evaluator = self.scheduling_env_factory(num_machines=problem.num_machines)
        self.last_evaluator_id = id(evaluator)
        try:
            obs, _ = evaluator.reset(
                options={"problem": problem, "nesting_context": nesting_context})
            done = False
            while not done:
                mask = evaluator._get_action_mask()
                action, _ = self.scheduling_policy.predict(
                    obs, action_masks=mask, deterministic=True)
                obs, _, terminated, truncated, _ = evaluator.step(action)
                done = terminated or truncated
            return {
                order_id: order["finished_time"]
                for order_id, order in evaluator.orders_snapshot.items()
            }
        except Exception as exc:
            raise SchedulingPolicyRolloutError(
                f"Scheduling terminal rollout failed for {len(problem.tasks)} tasks"
            ) from exc


class TerminalRewardTransform:
    """The original JointRewardCalculator EMA/tanh transform, without env state."""

    def __init__(self):
        self._cost_ema = None
        self._ema_alpha = 0.1

    def to_reward(self, cost_dict, w_terminal=10.0):
        penalty_ratio = cost_dict["penalty_ratio"]
        if self._cost_ema is None:
            self._cost_ema = penalty_ratio
        else:
            self._cost_ema = self._ema_alpha * penalty_ratio + (1 - self._ema_alpha) * self._cost_ema
        advantage = -(penalty_ratio - self._cost_ema)
        scaled = w_terminal * math.tanh(advantage * 2.0)
        return float(np.clip(scaled, -w_terminal * 2, w_terminal * 2))


class SchedulingTerminalRewardWrapper(gym.Wrapper):
    """Add explicitly evaluated scheduling cost to a terminal nesting transition."""

    def __init__(
        self,
        env,
        scheduling_policy=None,
        scheduling_env_factory=None,
        evaluation_mode="policy",
    ):
        super().__init__(env)
        self.scheduling_evaluator = SchedulingEvaluator(
            mode=evaluation_mode,
            scheduling_policy=scheduling_policy,
            scheduling_env_factory=scheduling_env_factory,
        )
        self.reward_transform = TerminalRewardTransform()
        self.cost_function = GlobalCostFunction()

    @property
    def evaluation_mode(self):
        return self.scheduling_evaluator.mode

    @property
    def scheduling_policy(self):
        return self.scheduling_evaluator.scheduling_policy

    @property
    def last_evaluator_id(self):
        return self.scheduling_evaluator.last_evaluator_id

    @property
    def cost_metrics(self):
        return self.env.unwrapped.cost_metrics

    def set_evaluator(self, evaluation_mode, scheduling_policy=None):
        """Switch explicit evaluator mode without replacing the reward transform."""
        self.scheduling_evaluator.set_mode(evaluation_mode, scheduling_policy)

    def set_scheduling_policy(self, scheduling_policy):
        self.set_evaluator("policy", scheduling_policy)

    def get_part_feats(self):
        return self.env.get_part_feats()

    def get_state_feat(self):
        return self.env.get_state_feat()

    def _get_action_mask(self):
        return self.env.unwrapped._get_action_mask()

    def _rollout(self, problem, nesting_context):
        return self.scheduling_evaluator.evaluate(problem, nesting_context)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated) or "terminal_result" not in info:
            return obs, reward, terminated, truncated, info
        result = info["terminal_result"]
        problem = build_scheduling_problem(result.instance, result.plates)
        nesting_context = build_nesting_context(result.instance, result.plates)
        finishes = self._rollout(problem, nesting_context)
        cost = self.cost_function.compute(
            list(result.plates), result.instance.orders, result.instance.parts,
            result.instance.plate_w, result.instance.plate_h, finishes,
        )
        terminal_reward = self.reward_transform.to_reward(cost, w_terminal=self.env.unwrapped.w_terminal)
        reward += terminal_reward
        base = self.env.unwrapped
        base_orders = getattr(base, "orders", None)
        if base_orders is not None:
            for order_id, finish_time in finishes.items():
                if order_id in base_orders:
                    base_orders[order_id]["finished_time"] = finish_time
        base.cost_metrics = {
            "cost_material": cost["cost_material"], "cost_jit": cost["cost_jit"],
            "cost_total": cost["cost_total"], "utilization": cost["utilization"],
            "plate_count": cost["plate_count"], "total_delay": cost["total_delay"],
            "late_count": cost["late_count"],
            "norm_reward": terminal_reward,
        }
        info["episode_metrics"] = base._compute_metrics()
        info["cost_metrics"] = dict(base.cost_metrics)
        return obs, reward, terminated, truncated, info
