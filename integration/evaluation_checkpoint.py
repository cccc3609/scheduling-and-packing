"""Strict resolver for Patch 7 Phase 3 checkpoint-pair metadata."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


PAIR_VERSION = 1
LEGACY_REJECTION = (
    "Formal dual-agent evaluation requires a Patch 7 phase3 checkpoint pair "
    "metadata file. Legacy checkpoints are not valid for correctness-mainline "
    "evaluation."
)


@dataclass(frozen=True)
class PairedCheckpoint:
    phase: str
    round: int
    nesting_checkpoint: Path
    scheduling_checkpoint: Path
    metadata_path: Path
    experiment_dir: Path


def resolve_phase3_checkpoint_pair(
    model_dir: str | os.PathLike[str], requested_round: int | None = None,
) -> PairedCheckpoint:
    """Resolve one exact Patch 7 pair; never infer or fall back to legacy files."""
    model_dir = Path(model_dir).resolve()
    if requested_round is None:
        candidates = []
        if model_dir.is_dir():
            for path in model_dir.glob("phase3_joint_c*.json"):
                match = re.fullmatch(r"phase3_joint_c(\d+)\.json", path.name)
                if match:
                    candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(LEGACY_REJECTION)
        requested_round, metadata_path = max(candidates)
    else:
        requested_round = int(requested_round)
        metadata_path = model_dir / f"phase3_joint_c{requested_round}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{LEGACY_REJECTION} Missing: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_names = {
        "nesting_checkpoint": f"nesting_joint_c{requested_round}_final.pt",
        "scheduling_checkpoint": f"scheduling_joint_c{requested_round}.zip",
    }
    if metadata.get("format_version") != PAIR_VERSION:
        raise ValueError("Unsupported Phase 3 checkpoint-pair metadata version")
    if metadata.get("phase") != "phase3":
        raise ValueError("Formal dual-agent evaluation requires phase='phase3'")
    if metadata.get("round") != requested_round:
        raise ValueError("Checkpoint-pair metadata round does not match requested round")
    for key, expected_name in expected_names.items():
        if metadata.get(key) != expected_name:
            raise ValueError(f"Checkpoint-pair metadata has inconsistent {key}")
        if Path(metadata[key]).name != metadata[key]:
            raise ValueError("Checkpoint metadata must contain local checkpoint filenames")
    nesting = model_dir / metadata["nesting_checkpoint"]
    scheduling = model_dir / metadata["scheduling_checkpoint"]
    missing = [str(path) for path in (nesting, scheduling) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Phase 3 checkpoint pair: {missing}")
    return PairedCheckpoint(
        phase="phase3", round=requested_round,
        nesting_checkpoint=nesting, scheduling_checkpoint=scheduling,
        metadata_path=metadata_path, experiment_dir=model_dir.parent,
    )


def resolve_latest_phase3_pair(
    experiment_root: str | os.PathLike[str] = "./experiments",
    requested_round: int | None = None,
) -> PairedCheckpoint:
    root = Path(experiment_root)
    if not root.is_dir():
        raise FileNotFoundError(f"{LEGACY_REJECTION} Missing: {root}")
    experiment_dirs = sorted(
        (path for path in root.glob("exp_*") if path.is_dir()),
        key=lambda path: path.stat().st_ctime, reverse=True,
    )
    if not experiment_dirs:
        raise FileNotFoundError(LEGACY_REJECTION)
    # The newest experiment is authoritative. An incomplete newest run must not
    # silently fall back to a stale pair from an older experiment.
    return resolve_phase3_checkpoint_pair(
        experiment_dirs[0] / "models", requested_round)
