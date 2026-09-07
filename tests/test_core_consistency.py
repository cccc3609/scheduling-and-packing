import numpy as np
import pytest

from core.cost import GlobalCostFunction
from core.instance import generate_instance
from core.processing import plate_processing_time
from core.scheduling_problem import build_scheduling_problem
from heuristic.scheduler import SchedulerStateMachine


class _Plate:
    def __init__(self, placed_parts):
        self.placed_parts = placed_parts


def _manual_case():
    parts = [
        {"w": 10, "h": 10, "area": 100, "order_id": 0},
        {"w": 10, "h": 20, "area": 200, "order_id": 1},
    ]
    orders = {
        0: {"due_date": 5.0, "finished_time": 0.0},
        1: {"due_date": 10.0, "finished_time": 0.0},
    }
    plates = [
        _Plate([(0, 0, 10, 10, 0, False)]),
        _Plate([(0, 0, 10, 20, 1, False)]),
    ]
    return plates, orders, parts


def test_processing_time_is_unique():
    plate = _Plate([(0, 0, 10, 20, 0, False), (10, 0, 5, 5, 0, False)])
    expected = plate_processing_time(plate.placed_parts, cutting_speed=10.0)

    # EDD builds a task from this value, then the state machine must consume it
    # unchanged (no hidden setup time or other adjustment).
    edd_task = {"cut": expected, "due": 100.0, "idx": 0}
    scheduler = SchedulerStateMachine(num_machines=1)
    assert scheduler.execute_assignment(
        0, edd_task["cut"], edd_task["idx"]
    ) == pytest.approx(expected)

    # This is intentionally a direct import: a correctly provisioned project
    # environment must execute the real SchedulingEnv path, never skip it.
    from envs.scheduling_env import SchedulingEnv

    instance = generate_instance(seed=7, num_parts=2, plate_size=(100, 100), num_machines=1)
    instance.parts = [
        {"w": 10, "h": 20, "area": 200, "order_id": 0, "due_date": 100.0},
        {"w": 5, "h": 5, "area": 25, "order_id": 0, "due_date": 100.0},
    ]
    instance.orders = {0: {"due_date": 100.0, "finished_time": 0.0}}
    problem = build_scheduling_problem(instance, [plate])
    env = SchedulingEnv(num_machines=1, max_tasks=2)
    env.reset(seed=7, options={"problem": problem, "nesting_context": np.zeros(8)})
    assert env.task_pool[0]["cut"] == pytest.approx(expected)


def test_cost_manual_case():
    plates, orders, parts = _manual_case()
    result = GlobalCostFunction().compute(
        plates, orders, parts, 20, 20, {0: 9.0, 1: 8.0})

    # material=(2*20*20 - 300)*0.05=25;
    # tardy order=100*0.002*4=0.8; early order=200*0.0005*2=0.2.
    assert result["cost_material"] == pytest.approx(25.0)
    assert result["cost_jit"] == pytest.approx(1.0)
    assert result["cost_total"] == pytest.approx(26.0)
    assert result["late_count"] == 1


def test_missing_finish_time_raises():
    plates, orders, parts = _manual_case()
    with pytest.raises(ValueError, match="Missing finish times"):
        GlobalCostFunction().compute(plates, orders, parts, 20, 20, {0: 9.0})


def test_instance_seed():
    first = generate_instance(seed=123)
    second = generate_instance(seed=123)
    third = generate_instance(seed=124)
    assert first == second
    assert first.parts != third.parts or first.orders != third.orders


def test_no_global_rng_dependency():
    np.random.seed(1)
    first = generate_instance(seed=123)
    np.random.seed(987654)
    second = generate_instance(seed=123)
    assert first == second
