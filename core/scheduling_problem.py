"""Immutable physical scheduling problems built from completed nesting layouts."""

from dataclasses import dataclass
from typing import Iterable

from .instance import ProductionInstance
from .processing import parts_cutting_time, plate_processing_time


@dataclass(frozen=True)
class SchedulingOrder:
    order_id: int
    due_date: float
    total_area: float
    proc_time: float


@dataclass(frozen=True)
class SchedulingTask:
    plate_index: int
    processing_time: float
    due_date: float
    order_ids: tuple[int, ...]
    plate_part_area: float


@dataclass(frozen=True)
class SchedulingProblem:
    tasks: tuple[SchedulingTask, ...]
    orders: tuple[SchedulingOrder, ...]
    num_machines: int
    plate_w: int
    plate_h: int
    episode_time_scale: float


@dataclass(frozen=True)
class NestedPlateResult:
    """Immutable plate snapshot exposed by the nesting environment."""
    plate_index: int
    placed_parts: tuple[tuple[float, float, float, float, int, bool], ...]


@dataclass(frozen=True)
class NestingTerminalResult:
    """Terminal nesting output; scheduling data is built from this snapshot."""
    instance: ProductionInstance
    plates: tuple[NestedPlateResult, ...]


def _placed_parts(plate) -> tuple[tuple[float, float, float, float, int, bool], ...]:
    return tuple(
        (float(p[0]), float(p[1]), float(p[2]), float(p[3]), int(p[4]), bool(p[5]))
        for p in plate.placed_parts
    )


def build_scheduling_problem(
    instance: ProductionInstance,
    plates: Iterable[object],
) -> SchedulingProblem:
    """Build the complete physical scheduling problem without RL-capacity limits."""
    order_parts = {
        order_id: tuple(part for part in instance.parts if part["order_id"] == order_id)
        for order_id in instance.orders
    }
    orders = tuple(
        SchedulingOrder(
            order_id=int(order_id),
            due_date=float(order["due_date"]),
            total_area=float(sum(part["area"] for part in order_parts[order_id])),
            proc_time=max(1.0, parts_cutting_time(order_parts[order_id])),
        )
        for order_id, order in sorted(instance.orders.items())
    )
    due_by_order = {order.order_id: order.due_date for order in orders}

    tasks = []
    for fallback_index, plate in enumerate(plates):
        placed_parts = _placed_parts(plate)
        if not placed_parts:
            continue
        order_ids = tuple(sorted({part[4] for part in placed_parts}))
        tasks.append(SchedulingTask(
            plate_index=int(getattr(plate, "plate_index", fallback_index)),
            processing_time=plate_processing_time(placed_parts),
            due_date=min((due_by_order[order_id] for order_id in order_ids), default=9999.0),
            order_ids=order_ids,
            plate_part_area=float(sum(part[2] * part[3] for part in placed_parts)),
        ))

    variable_work = sum(task.processing_time for task in tasks)
    return SchedulingProblem(
        tasks=tuple(tasks),
        orders=orders,
        num_machines=int(instance.num_machines),
        plate_w=int(instance.plate_w),
        plate_h=int(instance.plate_h),
        episode_time_scale=max(10.0, variable_work),
    )
