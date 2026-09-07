import copy

import numpy as np
import pytest

from core.cooperative_context import build_nesting_context
from core.instance import generate_instance
from core.scheduling_problem import build_scheduling_problem


class Plate:
    def __init__(self, parts):
        self.placed_parts = parts


def _problem():
    instance = generate_instance(seed=34, num_parts=2, plate_size=(100, 100), num_machines=1)
    placed = [(0, 0, p["w"], p["h"], p["order_id"], False) for p in instance.parts]
    plates = [Plate(placed)]
    return instance, plates, build_scheduling_problem(instance, plates)


def test_scheduling_env_consumes_problem_without_mutating_sources():
    from envs.scheduling_env import SchedulingEnv
    instance, plates, problem = _problem()
    instance_before, plates_before = copy.deepcopy(instance), copy.deepcopy(plates)
    env = SchedulingEnv(num_machines=1, max_tasks=2)
    env.reset(options={"problem": problem, "nesting_context": build_nesting_context(instance, plates)})
    env.step(0)
    assert instance == instance_before
    assert plates[0].placed_parts == plates_before[0].placed_parts
    assert problem.orders[0].due_date == instance.orders[problem.orders[0].order_id]["due_date"]
    assert not hasattr(env, "nesting_env") and not hasattr(env, "nesting_model")


def test_over_capacity_is_rejected_only_by_scheduling_env():
    from envs.scheduling_env import SchedulingEnv
    instance, plates, _ = _problem()
    problem = build_scheduling_problem(instance, plates * 3)
    env = SchedulingEnv(num_machines=1, max_tasks=2)
    with pytest.raises(ValueError, match="exceeds max_tasks"):
        env.reset(options={"problem": problem, "nesting_context": np.zeros(8)})
