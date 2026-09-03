"""Canonical processing-time definitions for the production model."""

from collections.abc import Iterable
from typing import Any

from config import COST_CONFIG


def _dimensions(part: Any) -> tuple[float, float]:
    """Read dimensions from a source part dict or a placed-part tuple."""
    if isinstance(part, dict):
        return float(part["w"]), float(part["h"])
    return float(part[2]), float(part[3])


def parts_perimeter(parts: Iterable[Any]) -> float:
    """Total perimeter of source parts or placed-part tuples."""
    return sum(2.0 * (w + h) for w, h in map(_dimensions, parts))


def parts_cutting_time(parts: Iterable[Any], cutting_speed: float | None = None) -> float:
    """Variable cutting time, without per-plate setup time."""
    speed = float(COST_CONFIG["cutting_speed"] if cutting_speed is None else cutting_speed)
    if speed <= 0.0:
        raise ValueError("cutting_speed must be positive")
    return parts_perimeter(parts) / speed


def plate_processing_time(
    placed_parts: Iterable[Any],
    cutting_speed: float | None = None,
    setup_time: float | None = None,
) -> float:
    """Processing time for exactly one plate: setup + total perimeter / speed."""
    setup = float(COST_CONFIG["plate_setup_time"] if setup_time is None else setup_time)
    return setup + parts_cutting_time(placed_parts, cutting_speed)
