import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from core.cost import GlobalCostFunction
from core.instance import ProductionInstance
from core.scheduling_problem import (
    NestedPlateResult,
    NestingTerminalResult,
    SchedulingOrder,
    SchedulingProblem,
    SchedulingTask,
)
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_terminal_reward import (
    SchedulingEvaluator,
    SchedulingPolicyRolloutError,
    SchedulingTerminalRewardWrapper,
    TerminalRewardTransform,
    configure_terminal_evaluator_for_phase,
    make_dual_agent_terminal_wrapper,
)


class _Plate:
    def __init__(self, placed_parts):
        self.placed_parts = placed_parts


def _problem(tasks, orders, num_machines=1):
    return SchedulingProblem(
        tasks=tuple(tasks),
        orders=tuple(orders),
        num_machines=num_machines,
        plate_w=100,
        plate_h=100,
        episode_time_scale=20.0,
    )


def _task(index, processing, due, order_ids):
    return SchedulingTask(index, processing, due, tuple(order_ids), 100.0)


def _order(order_id, due=1.0, area=100.0, processing=10.0):
    return SchedulingOrder(order_id, due, area, processing)


def _jit_order(due=100.0):
    return {0: {"due_date": due, "total_area": 100.0, "proc_time": 100.0}}


def _terminal_result():
    parts = [{
        "w": 10, "h": 10, "area": 100, "order_id": 0,
        "due_date": 1.0, "original_idx": 0,
    }]
    orders = {0: {"due_date": 1.0, "finished_time": 0.0}}
    instance = ProductionInstance(parts, orders, 100, 100, 1, 7)
    plates = (NestedPlateResult(
        0, ((0.0, 0.0, 10.0, 10.0, 0, False),)),)
    return NestingTerminalResult(instance, plates)


class _TerminalBase(gym.Env):
    def __init__(self, local_reward=1.25):
        super().__init__()
        self.result = _terminal_result()
        self.local_reward = local_reward
        self.w_terminal = 10.0
        self.cost_metrics = {}
        self.orders = {0: {"due_date": 1.0, "finished_time": 0.0}}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cost_metrics = {}
        self.orders[0]["finished_time"] = 0.0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(1, dtype=np.float32), self.local_reward,
            True, False, {"terminal_result": self.result},
        )

    def _compute_metrics(self):
        return dict(self.cost_metrics)


def test_global_cost_total_equals_material_plus_jit():
    parts = [{"w": 5, "h": 5, "area": 25, "order_id": 0}]
    orders = {0: {"due_date": 1.0, "finished_time": 0.0}}
    plate = _Plate([(0, 0, 5, 5, 0, False)])
    cost = GlobalCostFunction().compute(
        [plate], orders, parts, 10, 10, {0: 5.0})
    assert cost["cost_total"] == pytest.approx(
        cost["cost_material"] + cost["cost_jit"])


@pytest.mark.parametrize("finish", [None, math.nan, math.inf, -math.inf])
def test_global_cost_rejects_nonfinite_finish_time(finish):
    parts = [{"w": 5, "h": 5, "area": 25, "order_id": 0}]
    orders = {0: {"due_date": 1.0, "finished_time": 0.0}}
    with pytest.raises(ValueError, match="finite numeric"):
        GlobalCostFunction().compute(
            [_Plate([(0, 0, 5, 5, 0, False)])],
            orders, parts, 10, 10, {0: finish})


def test_jit_zero_inside_tolerance_band():
    fn = GlobalCostFunction()
    orders = _jit_order()
    assert fn.order_jit_cost(0, 97.5, orders) == pytest.approx(0.0)
    assert fn.order_jit_cost(0, 102.5, orders) == pytest.approx(0.0)


def test_jit_linear_in_middle_band():
    fn = GlobalCostFunction()
    orders = _jit_order()
    expected_coefficient = (0.05 - 0.025) / 0.05
    expected = 100.0 * fn.cost_tard * expected_coefficient * 5.0
    assert fn.order_jit_cost(0, 105.0, orders) == pytest.approx(expected)


def test_jit_coefficient_caps_after_outer_band():
    fn = GlobalCostFunction()
    orders = _jit_order()
    assert fn.order_jit_cost(0, 110.0, orders) == pytest.approx(
        100.0 * fn.cost_tard * 10.0)


def test_jit_band_shape_is_early_late_symmetric():
    fn = GlobalCostFunction()
    orders = _jit_order()
    early = fn.order_jit_cost(0, 95.0, orders)
    late = fn.order_jit_cost(0, 105.0, orders)
    early_coefficient = early / (100.0 * fn.cost_hold * 5.0)
    late_coefficient = late / (100.0 * fn.cost_tard * 5.0)
    assert early_coefficient == pytest.approx(late_coefficient)


def test_jit_monetary_rate_is_asymmetric():
    fn = GlobalCostFunction()
    orders = _jit_order()
    early = fn.order_jit_cost(0, 95.0, orders)
    late = fn.order_jit_cost(0, 105.0, orders)
    assert late / early == pytest.approx(fn.cost_tard / fn.cost_hold)


def test_order_completion_is_latest_linked_plate_completion():
    env = SchedulingEnv(num_machines=2, max_tasks=2)
    problem = _problem(
        [_task(0, 2.0, 100.0, [0]), _task(1, 5.0, 100.0, [0])],
        [_order(0, due=100.0)], num_machines=2)
    env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
    env.step(0)  # task 0 -> machine 0
    _, _, terminated, _, _ = env.step(3)  # task 1 -> machine 1
    assert terminated
    assert env.orders_snapshot[0]["finished_time"] == pytest.approx(5.0)


def test_scheduling_nonterminal_reward_has_no_load_balance_component():
    env = SchedulingEnv(num_machines=2, max_tasks=2)
    problem = _problem(
        [_task(0, 2.0, 10.0, [0]), _task(1, 3.0, 10.0, [1])],
        [_order(0), _order(1)], num_machines=2)
    env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
    env.machine_times[:] = [0.0, 100.0]
    _, reward, terminated, _, _ = env.step(0)
    assert not terminated
    assert reward == pytest.approx(0.0)


def test_scheduling_terminal_reward_uses_formal_jit_only():
    env = SchedulingEnv(num_machines=1, max_tasks=1)
    problem = _problem([_task(0, 5.0, 1.0, [0])], [_order(0)])
    env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
    _, reward, terminated, _, _ = env.step(0)
    jit = env.global_cost_fn.order_jit_cost(
        0, env.orders_snapshot[0]["finished_time"], env.orders_snapshot)
    expected = float(np.clip(-10.0 * jit / env.baseline_cost, -20.0, 20.0))
    assert terminated
    assert reward == pytest.approx(expected)


def test_invalid_scheduling_action_fails_fast():
    env = SchedulingEnv(num_machines=2, max_tasks=2)
    problem = _problem(
        [_task(0, 2.0, 10.0, [0])], [_order(0)], num_machines=2)
    env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
    padding_action = 2
    assert not env._get_action_mask()[padding_action]

    with pytest.raises(ValueError, match="action mask"):
        env.step(padding_action)


def test_invalid_scheduling_action_does_not_mutate_state():
    env = SchedulingEnv(num_machines=1, max_tasks=1)
    problem = _problem([_task(0, 2.0, 10.0, [0])], [_order(0)])
    env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
    env.step(0)

    machine_times_before = env.machine_times.copy()
    scheduled_before = env.scheduled_mask.copy()
    finish_times_before = {
        order_id: order["finished_time"]
        for order_id, order in env.orders_snapshot.items()
    }

    with pytest.raises(ValueError, match="action mask"):
        env.step(0)

    assert np.array_equal(env.machine_times, machine_times_before)
    assert np.array_equal(env.scheduled_mask, scheduled_before)
    assert {
        order_id: order["finished_time"]
        for order_id, order in env.orders_snapshot.items()
    } == finish_times_before


def test_non_delay_start_time_is_explicit():
    env = SchedulingEnv(num_machines=1, max_tasks=1)
    problem = _problem([_task(0, 5.0, 100.0, [0])], [_order(0, due=100.0)])
    env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
    env.machine_times[0] = 7.0
    env.step(0)
    assert env.orders_snapshot[0]["finished_time"] == pytest.approx(12.0)


def test_terminal_reward_transform_is_stateless():
    transform = TerminalRewardTransform()
    cost = {"penalty_ratio": 0.25}
    assert transform.to_reward(cost) == pytest.approx(transform.to_reward(cost))
    assert not hasattr(transform, "_cost_ema")


def test_lower_global_cost_produces_higher_terminal_reward():
    transform = TerminalRewardTransform()
    low = transform.to_reward({"penalty_ratio": 0.1})
    high = transform.to_reward({"penalty_ratio": 0.5})
    assert low > high


def test_terminal_reward_transform_matches_fixed_formula():
    rho = 0.3
    actual = TerminalRewardTransform().to_reward(
        {"penalty_ratio": rho}, w_terminal=10.0)
    assert actual == pytest.approx(-10.0 * math.tanh(2.0 * rho))


def test_terminal_wrapper_adds_global_reward_exactly_once():
    wrapper = SchedulingTerminalRewardWrapper(
        _TerminalBase(), evaluation_mode="edd")
    _, reward, terminated, _, info = wrapper.step(0)
    expected_terminal = -10.0 * math.tanh(
        2.0 * info["cost_metrics"]["penalty_ratio"])
    assert terminated
    assert reward == pytest.approx(1.25 + expected_terminal)


def test_second_terminal_settlement_fails_fast():
    wrapper = SchedulingTerminalRewardWrapper(
        _TerminalBase(), evaluation_mode="edd")
    wrapper.step(0)
    with pytest.raises(RuntimeError, match="already been settled"):
        wrapper.step(0)


def test_terminal_wrapper_reset_clears_settlement_guard():
    wrapper = SchedulingTerminalRewardWrapper(
        _TerminalBase(), evaluation_mode="edd")
    wrapper.step(0)
    wrapper.reset(seed=3)
    _, _, terminated, _, _ = wrapper.step(0)
    assert terminated


def test_policy_terminal_rollout_stops_before_all_false_predict():
    problem = _problem([_task(0, 2.0, 10.0, [0])], [_order(0, due=10.0)])

    class Policy:
        def __init__(self):
            self.masks = []

        def predict(self, obs, action_masks=None, deterministic=True):
            self.masks.append(np.asarray(action_masks).copy())
            return int(np.flatnonzero(action_masks)[0]), None

    policy = Policy()
    finishes = SchedulingEvaluator("policy", policy).evaluate(
        problem, np.zeros(8, dtype=np.float32))
    assert finishes[0] == pytest.approx(2.0)
    assert len(policy.masks) == 1
    assert policy.masks[0].any()


def test_policy_mode_failure_does_not_fallback_to_edd():
    problem = _problem([_task(0, 2.0, 10.0, [0])], [_order(0, due=10.0)])

    class FailingPolicy:
        def predict(self, *args, **kwargs):
            raise RuntimeError("failure")

    with pytest.raises(SchedulingPolicyRolloutError):
        SchedulingEvaluator("policy", FailingPolicy()).evaluate(
            problem, np.zeros(8, dtype=np.float32))


def test_phase1_uses_explicit_edd_terminal_evaluator():
    wrapper = SchedulingTerminalRewardWrapper(
        _TerminalBase(), evaluation_mode="edd")
    assert wrapper.evaluation_mode == "edd"


def test_phase3_uses_policy_terminal_evaluator():
    policy = object()
    wrapper = SchedulingTerminalRewardWrapper(
        _TerminalBase(), evaluation_mode="edd")
    wrapper.set_evaluator("policy", scheduling_policy=policy)
    assert wrapper.evaluation_mode == "policy"
    assert wrapper.scheduling_policy is policy


def test_resume_phase3_uses_policy_evaluator():
    policy = object()
    wrapper = SchedulingTerminalRewardWrapper(
        _TerminalBase(), evaluation_mode="edd")
    resume_env = SimpleNamespace(env=wrapper)
    configure_terminal_evaluator_for_phase(
        resume_env.env, "phase3", scheduling_policy=policy)
    assert wrapper.evaluation_mode == "policy"
    assert wrapper.scheduling_policy is policy


def test_dual_agent_evaluation_uses_policy_evaluator():
    policy = object()
    wrapper = make_dual_agent_terminal_wrapper(_TerminalBase(), policy)
    assert wrapper.evaluation_mode == "policy"
    assert wrapper.scheduling_policy is policy
    with pytest.raises(ValueError, match="requires a scheduling policy"):
        make_dual_agent_terminal_wrapper(_TerminalBase(), None)


def test_joint_reward_calculator_removed():
    import models.comm_encoders as encoders

    assert not hasattr(encoders, "JointRewardCalculator")
