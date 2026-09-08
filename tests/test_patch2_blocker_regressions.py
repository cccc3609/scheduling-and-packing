import copy

import gymnasium as gym
import numpy as np
import pytest

from core.cooperative_context import build_nesting_context
from core.cost import GlobalCostFunction
from core.instance import ProductionInstance
from core.scheduling_problem import (
    NestedPlateResult,
    NestingTerminalResult,
    build_scheduling_problem,
)
from integration.scheduling_terminal_reward import (
    SchedulingEvaluator,
    SchedulingPolicyRolloutError,
    SchedulingTerminalRewardWrapper,
)


def _terminal_result():
    parts = [
        {"w": 10, "h": 20, "area": 200, "due_date": 1.0,
         "order_id": 0, "original_idx": 0},
        {"w": 5, "h": 5, "area": 25, "due_date": 20.0,
         "order_id": 1, "original_idx": 1},
    ]
    orders = {
        0: {"due_date": 1.0, "finished_time": 0.0},
        1: {"due_date": 20.0, "finished_time": 0.0},
    }
    instance = ProductionInstance(parts, orders, 100, 100, 1, 7)
    plates = (
        NestedPlateResult(0, ((0.0, 0.0, 10.0, 20.0, 0, False),)),
        NestedPlateResult(1, ((0.0, 0.0, 5.0, 5.0, 1, False),)),
    )
    return NestingTerminalResult(instance, plates)


class TerminalBase(gym.Env):
    def __init__(self, result=None, local_reward=1.25):
        super().__init__()
        self.result = result or _terminal_result()
        self.local_reward = local_reward
        self.w_terminal = 10.0
        self.cost_metrics = {}
        self.orders = copy.deepcopy(self.result.instance.orders)

    def step(self, action):
        return (
            np.zeros(1, dtype=np.float32),
            self.local_reward,
            True,
            False,
            {"terminal_result": self.result},
        )

    def _compute_metrics(self):
        metrics = dict(self.cost_metrics)
        metrics["late_orders_count"] = sum(
            order["finished_time"] > order["due_date"]
            for order in self.orders.values()
        )
        return metrics


class CountingFailPolicy:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, obs, action_masks=None, deterministic=True):
        self.predict_calls += 1
        raise RuntimeError("policy must not run in explicit EDD mode")


def test_phase1_explicit_edd_does_not_call_random_policy_and_keeps_reward():
    policy = CountingFailPolicy()
    wrapper = SchedulingTerminalRewardWrapper(
        TerminalBase(), scheduling_policy=policy, evaluation_mode="edd")
    reward_transform = wrapper.reward_transform

    _, reward, terminated, truncated, info = wrapper.step(0)

    assert terminated and not truncated
    assert policy.predict_calls == 0
    expected_terminal = -10.0 * np.tanh(
        2.0 * info["cost_metrics"]["penalty_ratio"])
    assert reward == pytest.approx(1.25 + expected_terminal)
    assert wrapper.evaluation_mode == "edd"
    assert info["cost_metrics"]["cost_total"] > 0.0
    assert "norm_reward" in info["cost_metrics"]
    wrapper.set_evaluator("policy", scheduling_policy=policy)
    assert wrapper.reward_transform is reward_transform


def test_explicit_policy_failure_never_falls_back_to_edd():
    wrapper = SchedulingTerminalRewardWrapper(
        TerminalBase(),
        scheduling_policy=CountingFailPolicy(),
        evaluation_mode="policy",
    )

    with pytest.raises(SchedulingPolicyRolloutError):
        wrapper.step(0)


def test_resume_nesting_env_retains_terminal_scheduling_reward_and_metrics():
    from train_resume import build_resume_nesting_env

    env = build_resume_nesting_env()
    obs, _ = env.reset(
        seed=31, options={"num_parts": 2, "plate_size": (100, 100)})
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        valid_actions = np.flatnonzero(env.action_masks())
        assert len(valid_actions) > 0
        obs, _, terminated, truncated, info = env.step(int(valid_actions[0]))

    assert env.env.evaluation_mode == "edd"
    assert info["cost_metrics"]
    assert set((
        "cost_material", "cost_jit", "cost_total", "late_count",
        "total_delay", "utilization", "plate_count", "norm_reward",
    )).issubset(info["cost_metrics"])


def test_explicit_edd_cost_metrics_match_global_cost_function():
    result = _terminal_result()
    problem = build_scheduling_problem(result.instance, result.plates)
    context = build_nesting_context(result.instance, result.plates)
    finish_times = SchedulingEvaluator(mode="edd").evaluate(problem, context)
    expected = GlobalCostFunction().compute(
        list(result.plates),
        result.instance.orders,
        result.instance.parts,
        result.instance.plate_w,
        result.instance.plate_h,
        finish_times,
    )

    wrapper = SchedulingTerminalRewardWrapper(
        TerminalBase(result), evaluation_mode="edd")
    _, _, _, _, info = wrapper.step(0)
    actual = info["cost_metrics"]

    for key in (
        "cost_material", "cost_jit", "cost_total", "late_count",
        "total_delay", "utilization", "plate_count",
    ):
        assert actual[key] == pytest.approx(expected[key])
