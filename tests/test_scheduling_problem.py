from dataclasses import fields

import pytest

from core.cooperative_context import build_nesting_context, build_scheduling_context
from core.instance import generate_instance
from core.processing import plate_processing_time
from core.scheduling_problem import build_scheduling_problem


class Plate:
    def __init__(self, parts):
        self.placed_parts = parts


def _fixture():
    instance = generate_instance(seed=12, num_parts=3, plate_size=(100, 100), num_machines=2)
    parts = [
        (0, 0, instance.parts[0]["w"], instance.parts[0]["h"], instance.parts[0]["order_id"], False),
        (5, 0, instance.parts[1]["w"], instance.parts[1]["h"], instance.parts[1]["order_id"], False),
    ]
    return instance, [Plate(parts)]


def test_problem_is_deterministic_immutable_and_context_free():
    instance, plates = _fixture()
    first = build_scheduling_problem(instance, plates)
    second = build_scheduling_problem(instance, plates)
    assert first == second
    assert "nesting_context" not in {field.name for field in fields(first)}
    assert "finished_time" not in {field.name for field in fields(first.orders[0])}
    with pytest.raises(Exception):
        first.tasks[0].processing_time = 0.0


def test_problem_processing_time_and_raw_context_are_canonical():
    instance, plates = _fixture()
    problem = build_scheduling_problem(instance, plates)
    assert problem.tasks[0].processing_time == pytest.approx(plate_processing_time(plates[0].placed_parts))
    assert (build_scheduling_context(instance) == build_scheduling_context(instance)).all()
    assert (build_nesting_context(instance, plates) == build_nesting_context(instance, plates)).all()


def test_builder_is_not_limited_by_rl_capacity():
    instance, plates = _fixture()
    problem = build_scheduling_problem(instance, plates * 130)
    assert len(problem.tasks) == 130
