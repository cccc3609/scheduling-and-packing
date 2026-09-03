"""Deterministic production-instance generation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import COST_CONFIG, TRAIN_CONFIG
from .processing import parts_cutting_time


@dataclass
class ProductionInstance:
    parts: list[dict[str, Any]]
    orders: dict[int, dict[str, float]]
    plate_w: int
    plate_h: int
    num_machines: int
    seed: int | None


def generate_instance(
    seed: int | None = None,
    *,
    num_parts: int | None = None,
    plate_size: tuple[int, int] | None = None,
    num_machines: int = 3,
    cutting_speed: float | None = None,
    rng: np.random.Generator | None = None,
) -> ProductionInstance:
    """Generate one instance from an explicit RNG or a seed-owned RNG.

    Passing ``rng`` is intended for Gym environments, whose ``self.np_random``
    controls the complete episode. Direct callers should pass ``seed``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if num_parts is None:
        num_parts = int(rng.integers(TRAIN_CONFIG["min_parts"], TRAIN_CONFIG["max_parts"] + 1))
    if plate_size is None:
        plate_w = int(rng.integers(TRAIN_CONFIG["min_plate_dim"], TRAIN_CONFIG["max_plate_dim"] + 1))
        plate_h = int(rng.integers(TRAIN_CONFIG["min_plate_dim"], TRAIN_CONFIG["max_plate_dim"] + 1))
    else:
        plate_w, plate_h = map(int, plate_size)
    if num_parts < 1 or plate_w < 1 or plate_h < 1 or num_machines < 1:
        raise ValueError("num_parts, plate_size, and num_machines must be positive")

    speed = COST_CONFIG["cutting_speed"] if cutting_speed is None else cutting_speed
    parts: list[dict[str, Any]] = []
    orders: dict[int, dict[str, float]] = {}
    count = order_id = 0
    avg_perimeter = 2.0 * (0.25 * plate_w + 0.25 * plate_h)
    estimated_makespan = (num_parts * avg_perimeter / max(0.1, speed) / num_machines) * 1.3

    while count < num_parts:
        batch = min(int(rng.integers(1, 16)), num_parts - count)
        order_parts: list[dict[str, int]] = []
        for _ in range(batch):
            width_ratio = float(rng.beta(2, 5) * 0.9 + 0.05)
            height_ratio = float(rng.beta(2, 5) * 0.9 + 0.05)
            if float(rng.random()) > 0.5:
                width_ratio, height_ratio = height_ratio, width_ratio
            width = max(1, int(width_ratio * plate_w))
            height = max(1, int(height_ratio * plate_h))
            order_parts.append({"w": width, "h": height, "area": width * height})

        self_time = parts_cutting_time(order_parts, speed)
        buffer_time = (int(rng.poisson(2.0)) + 0.1) * (estimated_makespan / 2.0)
        due_date = self_time + buffer_time
        orders[order_id] = {"due_date": due_date, "finished_time": 0.0}
        for part in order_parts:
            parts.append({
                **part,
                "due_date": due_date,
                "order_id": order_id,
                "original_idx": count,
            })
            count += 1
        order_id += 1

    rng.shuffle(parts)
    return ProductionInstance(parts, orders, plate_w, plate_h, num_machines, seed)
