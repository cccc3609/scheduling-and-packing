"""Deterministic cooperative observation context, separate from physical problems."""

import numpy as np

from .instance import ProductionInstance
from .processing import parts_cutting_time


def build_scheduling_context(instance: ProductionInstance) -> np.ndarray:
    """Return the 16-D instance-to-nesting raw scheduling context."""
    parts, orders = instance.parts, instance.orders
    if not orders:
        return np.zeros(16, dtype=np.float32)
    dues = np.asarray([order["due_date"] for order in orders.values()], dtype=float)
    workload = parts_cutting_time(parts)
    order_sizes = np.asarray([
        sum(1 for part in parts if part["order_id"] == order_id) for order_id in orders
    ], dtype=float)
    areas = np.asarray([part["area"] for part in parts], dtype=float)
    total_area = float(areas.sum())
    plate_area = max(1.0, float(instance.plate_w * instance.plate_h))
    order_areas = np.asarray([
        sum(part["area"] for part in parts if part["order_id"] == order_id)
        for order_id in orders
    ], dtype=float)
    n_orders = max(1, len(orders))
    features = [
        float(dues.mean()) / max(1.0, workload),
        float(dues.std()) / max(1.0, workload),
        float(dues.min()) / max(1.0, workload),
        float(dues.max()) / max(1.0, workload),
        instance.num_machines / n_orders,
        workload / max(1.0, instance.num_machines * float(dues.max())),
        float(np.mean(dues < workload)),
        total_area / n_orders,
        float(order_sizes.std()) / n_orders,
        len(orders) / 10.0,
        workload / max(1.0, float(dues.min())),
        0.0,
        total_area / plate_area,
        float(areas.max()) / plate_area,
        float(order_sizes.mean()),
        float(order_sizes.max()),
    ]
    return np.asarray(features, dtype=np.float32)


def build_nesting_context(instance: ProductionInstance, plates) -> np.ndarray:
    """Return the 8-D completed-nesting summary used by SchedulingEnv."""
    snapshots = [plate for plate in plates if plate.placed_parts]
    if not snapshots:
        return np.zeros(8, dtype=np.float32)
    plate_area = max(1.0, float(instance.plate_w * instance.plate_h))
    utils = np.asarray([
        sum(part[2] * part[3] for part in plate.placed_parts) / plate_area
        for plate in snapshots
    ], dtype=float)
    due_stds, fragments = [], []
    for plate in snapshots:
        order_ids = [int(part[4]) for part in plate.placed_parts]
        dues = [instance.orders[oid]["due_date"] for oid in order_ids if oid in instance.orders]
        if len(dues) > 1:
            due_stds.append(float(np.std(dues)))
        for order_id in set(order_ids):
            total = sum(1 for part in instance.parts if part["order_id"] == order_id)
            fragments.append(1.0 - order_ids.count(order_id) / max(1, total))
    expected = max(1.0, sum(part["area"] for part in instance.parts) / plate_area)
    return np.asarray([
        len(snapshots) / expected,
        float(utils.mean()),
        float(utils.std()) if len(utils) > 1 else 0.0,
        float(np.mean(due_stds)) if due_stds else 0.0,
        float(np.std(due_stds)) if len(due_stds) > 1 else 0.0,
        float(np.mean(fragments)) if fragments else 0.0,
        float(utils.min()),
        len(snapshots) / 10.0,
    ], dtype=np.float32)
