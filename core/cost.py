"""Canonical material and just-in-time cost implementation."""

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

from config import COST_CONFIG
from .processing import parts_cutting_time


class GlobalCostFunction:
    """Single formal implementation of material, JIT, and total costs."""

    def __init__(
        self,
        cost_mat: float | None = None,
        cost_hold: float | None = None,
        cost_tard: float | None = None,
        cutting_speed: float | None = None,
    ):
        self.cost_mat = float(COST_CONFIG["cost_material"] if cost_mat is None else cost_mat)
        self.cost_hold = float(COST_CONFIG["cost_earliness"] if cost_hold is None else cost_hold)
        self.cost_tard = float(COST_CONFIG["cost_tardiness"] if cost_tard is None else cost_tard)
        self.cutting_speed = float(COST_CONFIG["cutting_speed"] if cutting_speed is None else cutting_speed)

    def _order_jit_breakdown(
        self,
        order_id: int,
        finish_time: float,
        orders: Mapping[int, Mapping[str, Any]],
        parts_pool: list[Mapping[str, Any]] | None = None,
    ) -> tuple[float, float, bool]:
        """Return ``(jit_cost, tardiness_delay, is_late)`` for one order."""
        order = orders[order_id]
        if parts_pool is None:
            value = float(order["total_area"])
            processing = float(order["proc_time"])
        else:
            order_parts = [part for part in parts_pool if part["order_id"] == order_id]
            value = sum(float(part["area"]) for part in order_parts)
            processing = parts_cutting_time(order_parts, self.cutting_speed)
        processing = max(1.0, processing)
        difference = float(finish_time) - float(order["due_date"])
        ratio = abs(difference) / processing
        coefficient = 0.0 if ratio <= 0.025 else min(1.0, (ratio - 0.025) / 0.05)
        is_late = difference > 0.0
        rate = self.cost_tard if is_late else self.cost_hold
        return value * rate * coefficient * abs(difference), (difference if is_late else 0.0), is_late

    def order_jit_cost(
        self,
        order_id: int,
        finish_time: float,
        orders: Mapping[int, Mapping[str, Any]],
        parts_pool: list[Mapping[str, Any]] | None = None,
    ) -> float:
        """Compute the official JIT penalty for one completed order."""
        return self._order_jit_breakdown(order_id, finish_time, orders, parts_pool)[0]

    def compute(
        self,
        plates: list[Any],
        orders: Mapping[int, Mapping[str, Any]],
        parts_pool: list[Mapping[str, Any]],
        plate_w: int | float,
        plate_h: int | float,
        order_finish_times: Mapping[int, float],
    ) -> dict[str, float | int]:
        missing_orders = set(orders).difference(order_finish_times)
        if missing_orders:
            raise ValueError(f"Missing finish times for orders: {sorted(missing_orders)}")
        nonfinite_orders = [
            order_id for order_id in orders
            if not isinstance(order_finish_times[order_id], Real)
            or not math.isfinite(float(order_finish_times[order_id]))
        ]
        if nonfinite_orders:
            raise ValueError(
                "Finish times must be finite numeric values for orders: "
                f"{sorted(nonfinite_orders)}"
            )

        final_plates = [plate for plate in plates if plate.placed_parts]
        total_part_area = sum(float(part["area"]) for part in parts_pool)
        consumed_area = len(final_plates) * float(plate_w) * float(plate_h)
        utilization = total_part_area / consumed_area if consumed_area > 0.0 else 0.001
        cost_material = max(0.0, consumed_area - total_part_area) * self.cost_mat

        cost_jit = total_delay = 0.0
        late_count = 0
        for order_id in orders:
            order_cost, delay, is_late = self._order_jit_breakdown(
                order_id, order_finish_times[order_id], orders, parts_pool)
            cost_jit += order_cost
            total_delay += delay
            late_count += int(is_late)

        total_cost = cost_material + cost_jit
        intrinsic = max(1.0, total_part_area * self.cost_mat)
        return {
            "cost_material": cost_material,
            "cost_jit": cost_jit,
            "cost_total": total_cost,
            "utilization": utilization,
            "plate_count": len(final_plates),
            "total_delay": total_delay,
            "late_count": late_count,
            "penalty_ratio": total_cost / intrinsic,
        }
