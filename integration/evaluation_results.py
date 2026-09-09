"""Shared deterministic evaluation cases, formal metrics, and result output."""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from core.cost import GlobalCostFunction
from core.instance import ProductionInstance, generate_instance
from integration.evaluation_checkpoint import PairedCheckpoint


FORMAL_METRICS = (
    "Utilization", "Plate_Count", "Late_Count", "Total_Delay_Time",
    "JIT_Cost", "Material_Cost", "Total_Cost",
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: int
    case_seed: int
    instance: ProductionInstance
    scenario: str
    config: Mapping[str, Any]

    def independent_instance(self) -> ProductionInstance:
        return copy.deepcopy(self.instance)


def make_evaluation_case(
    case_id: int, base_seed: int, *, num_parts: int,
    plate_size: tuple[int, int], num_machines: int = 3,
    scenario: str = "default", config: Mapping[str, Any] | None = None,
) -> EvaluationCase:
    case_seed = int(base_seed) + int(case_id)
    instance = generate_instance(
        seed=case_seed, num_parts=num_parts, plate_size=plate_size,
        num_machines=num_machines,
    )
    metadata = dict(config or {})
    metadata.update({
        "scenario": scenario, "num_parts": int(num_parts),
        "plate_w": int(plate_size[0]), "plate_h": int(plate_size[1]),
        "num_machines": int(num_machines),
    })
    return EvaluationCase(
        int(case_id), case_seed, instance, scenario,
        MappingProxyType(metadata),
    )


def formal_metrics(plates, orders, parts, plate_w, plate_h) -> dict[str, float | int]:
    cost = GlobalCostFunction().compute(
        plates, orders, parts, plate_w, plate_h,
        {order_id: order["finished_time"] for order_id, order in orders.items()},
    )
    return {
        "Utilization": cost["utilization"],
        "Plate_Count": cost["plate_count"],
        "Late_Count": cost["late_count"],
        "Total_Delay_Time": cost["total_delay"],
        "JIT_Cost": cost["cost_jit"],
        "Material_Cost": cost["cost_material"],
        "Total_Cost": cost["cost_total"],
    }


def case_record(
    run_id: str, case: EvaluationCase, method: str, *,
    evaluation_mode: str, pair: PairedCheckpoint | None,
    metrics: Mapping[str, Any], status: str = "ok", error: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id, "case_id": case.case_id,
        "case_seed": case.case_seed, "method": method,
        "phase": pair.phase if pair else "N/A",
        "round": pair.round if pair else "N/A",
        "evaluation_mode": evaluation_mode,
        "nesting_checkpoint": str(pair.nesting_checkpoint) if pair else "",
        "scheduling_checkpoint": str(pair.scheduling_checkpoint) if pair else "",
        **dict(case.config), **{name: metrics.get(name, "") for name in FORMAL_METRICS},
        "status": status, "error": error,
    }


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    failed = [row for row in records if row.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"Evaluation contains {len(failed)} failed case records")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        groups.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)
    summary = []
    for (scenario, method), rows in sorted(groups.items()):
        item: dict[str, Any] = {
            "scenario": scenario, "method": method,
            "success_count": len(rows), "failed_count": 0,
        }
        for metric in FORMAL_METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = fmean(values)
            item[f"{metric}_std"] = pstdev(values)
        summary.append(item)
    return summary


def make_run_id(kind: str, pair: PairedCheckpoint | None, base_seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    round_text = f"r{pair.round}" if pair else "baseline"
    return f"{kind}_{round_text}_s{int(base_seed)}_{stamp}"


def create_run_directory(root: str | os.PathLike[str], run_id: str) -> Path:
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evaluation result")
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_results(
    run_dir: str | os.PathLike[str], manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    records = list(records)
    summary = aggregate_records(records)
    with (run_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    _write_csv(run_dir / "cases.csv", records)
    _write_csv(run_dir / "summary.csv", summary)
    return summary


def build_manifest(
    run_id: str, kind: str, mode: str, base_seed: int, case_count: int,
    pair: PairedCheckpoint | None, config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=3,
        ).stdout.strip()
    except Exception:
        git_commit = "unavailable"
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": kind, "evaluation_mode": mode,
        "base_seed": int(base_seed), "number_of_cases": int(case_count),
        "phase": pair.phase if pair else "N/A",
        "round": pair.round if pair else "N/A",
        "nesting_checkpoint": str(pair.nesting_checkpoint) if pair else "",
        "scheduling_checkpoint": str(pair.scheduling_checkpoint) if pair else "",
        "pair_metadata": str(pair.metadata_path) if pair else "",
        "case_generation_config": dict(config), "git_commit": git_commit,
    }
