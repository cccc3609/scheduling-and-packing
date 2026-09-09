import copy
import io
import json
import zipfile

import pytest
import torch
import numpy as np

from evaluate_batch import run_ffd_edd_baseline
from evaluate_generalization import run_rl_episode
from envs.packing_envs import NestingSchedulingEnv
from integration.evaluation_checkpoint import (
    LEGACY_REJECTION, resolve_latest_phase3_pair,
    resolve_phase3_checkpoint_pair,
)
from integration.evaluation_results import (
    FORMAL_METRICS, aggregate_records, build_manifest, case_record,
    create_run_directory, formal_metrics, make_evaluation_case,
    write_evaluation_results,
)
from models.sched_policy_loader import load_scheduling_policy


def _write_pair(model_dir, round_id=3, **overrides):
    model_dir.mkdir(parents=True)
    nesting = model_dir / f"nesting_joint_c{round_id}_final.pt"
    scheduling = model_dir / f"scheduling_joint_c{round_id}.zip"
    nesting.touch()
    scheduling.touch()
    metadata = {
        "format_version": 1,
        "phase": "phase3",
        "round": round_id,
        "nesting_checkpoint": nesting.name,
        "scheduling_checkpoint": scheduling.name,
    }
    metadata.update(overrides)
    path = model_dir / f"phase3_joint_c{round_id}.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _instance_signature(instance):
    return (copy.deepcopy(instance.parts), copy.deepcopy(instance.orders),
            instance.plate_w, instance.plate_h, instance.num_machines)


def test_rl_and_baseline_use_identical_instance():
    case = make_evaluation_case(
        0, 100, num_parts=8, plate_size=(120, 90), num_machines=3)
    class FirstValidNesting:
        layout = NestingSchedulingEnv().layout

        def forward_decision(self, _parts, _state, mask):
            logits = torch.where(
                mask, torch.zeros_like(mask, dtype=torch.float32),
                torch.full_like(mask, -1e9, dtype=torch.float32))
            return logits, torch.zeros((1, 1))

    class FirstValidScheduling:
        def predict(self, _obs, action_masks=None, deterministic=True):
            assert deterministic is True
            return int(np.flatnonzero(action_masks)[0]), None

    rl_env = run_rl_episode(
        FirstValidNesting(), FirstValidScheduling(), "cpu", case.case_seed,
        8, (120, 90), evaluation_mode="policy",
        instance=case.independent_instance())
    baseline_instance = case.independent_instance()
    assert _instance_signature(rl_env.current_instance) == _instance_signature(
        baseline_instance)
    run_ffd_edd_baseline(
        baseline_instance.parts, baseline_instance.orders,
        baseline_instance.plate_w, baseline_instance.plate_h)
    assert _instance_signature(case.instance) == _instance_signature(
        baseline_instance)


def test_evaluation_does_not_mutate_shared_instance():
    case = make_evaluation_case(0, 8, num_parts=5, plate_size=(100, 100))
    before = _instance_signature(case.instance)
    working = case.independent_instance()
    working.parts[0]["w"] = -1
    working.orders[0]["finished_time"] = 999
    assert _instance_signature(case.instance) == before


def test_evaluation_is_reproducible_with_same_seed():
    first = make_evaluation_case(4, 2000, num_parts=9, plate_size=(130, 110))
    second = make_evaluation_case(4, 2000, num_parts=9, plate_size=(130, 110))
    assert first.case_seed == second.case_seed == 2004
    assert _instance_signature(first.instance) == _instance_signature(second.instance)


def test_different_seed_changes_generated_case():
    first = make_evaluation_case(0, 1, num_parts=9, plate_size=(130, 110))
    second = make_evaluation_case(1, 1, num_parts=9, plate_size=(130, 110))
    assert _instance_signature(first.instance) != _instance_signature(second.instance)


def test_dual_agent_eval_requires_metadata_paired_same_round(tmp_path):
    model_dir = tmp_path / "models"
    metadata = _write_pair(model_dir, 7)
    pair = resolve_phase3_checkpoint_pair(model_dir, 7)
    assert pair.round == 7
    assert pair.metadata_path == metadata.resolve()
    assert "c7" in pair.nesting_checkpoint.name
    assert "c7" in pair.scheduling_checkpoint.name


@pytest.mark.parametrize("override", [
    {"round": 2},
    {"phase": "phase2"},
    {"nesting_checkpoint": "nesting_joint_c2_final.pt"},
    {"scheduling_checkpoint": "scheduling_joint_c2.zip"},
])
def test_pair_metadata_rejects_missing_or_cross_round_checkpoint(tmp_path, override):
    model_dir = tmp_path / "models"
    _write_pair(model_dir, 1, **override)
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_phase3_checkpoint_pair(model_dir, 1)


def test_pair_metadata_rejects_missing_checkpoint(tmp_path):
    model_dir = tmp_path / "models"
    _write_pair(model_dir, 1)
    (model_dir / "scheduling_joint_c1.zip").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        resolve_phase3_checkpoint_pair(model_dir, 1)


def test_evaluation_rejects_legacy_checkpoint(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "nesting_phase1_final.pt").touch()
    (model_dir / "scheduling_phase2.zip").touch()
    with pytest.raises(FileNotFoundError, match="Legacy checkpoints") as exc:
        resolve_phase3_checkpoint_pair(model_dir)
    assert LEGACY_REJECTION in str(exc.value)


def test_latest_experiment_does_not_fallback_to_older_pair(tmp_path):
    old_models = tmp_path / "exp_old" / "models"
    _write_pair(old_models, 1)
    new_models = tmp_path / "exp_new" / "models"
    new_models.mkdir(parents=True)
    # Make the incomplete experiment unambiguously newest.
    import os
    os.utime(tmp_path / "exp_new", (2_000_000_000, 2_000_000_000))
    with pytest.raises(FileNotFoundError, match="Legacy checkpoints"):
        resolve_latest_phase3_pair(tmp_path)


def test_scheduling_loader_rejects_missing_weight_group(tmp_path):
    checkpoint = tmp_path / "incomplete.zip"
    payload = io.BytesIO()
    torch.save({"action_net.weight": torch.zeros(1, 1)}, payload)
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("policy.pth", payload.getvalue())
    with pytest.raises(ValueError, match="missing required weight groups"):
        load_scheduling_policy(str(checkpoint))


def test_rl_and_baseline_use_same_global_cost():
    case = make_evaluation_case(0, 44, num_parts=7, plate_size=(100, 100))
    instance = case.independent_instance()
    plates, _logs, orders, metrics = run_ffd_edd_baseline(
        instance.parts, instance.orders, instance.plate_w, instance.plate_h)
    official = formal_metrics(
        plates, orders, instance.parts, instance.plate_w, instance.plate_h)
    assert official["Material_Cost"] == metrics["cost_material"]
    assert official["JIT_Cost"] == metrics["cost_jit"]
    assert official["Total_Cost"] == metrics["cost_total"]


def test_failed_case_is_not_silently_dropped():
    with pytest.raises(RuntimeError, match="1 failed"):
        aggregate_records([{"status": "failed"}])


def test_result_metadata_records_case_seed_round_mode_and_config(tmp_path):
    metadata = _write_pair(tmp_path / "exp_1" / "models", 5)
    pair = resolve_phase3_checkpoint_pair(metadata.parent, 5)
    case = make_evaluation_case(
        2, 100, num_parts=4, plate_size=(80, 70), scenario="small",
        config={"load": "low"})
    metrics = {name: 1 for name in FORMAL_METRICS}
    record = case_record(
        "run", case, "Dual-Agent RL", evaluation_mode="policy",
        pair=pair, metrics=metrics)
    assert record["case_seed"] == 102
    assert record["round"] == 5
    assert record["evaluation_mode"] == "policy"
    assert record["load"] == "low"
    assert record["num_machines"] == 3


def test_result_output_does_not_silently_overwrite_existing_run(tmp_path):
    run_dir = create_run_directory(tmp_path, "fixed")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "fixed")
    case = make_evaluation_case(0, 1, num_parts=2, plate_size=(50, 50))
    metrics = {name: 1 for name in FORMAL_METRICS}
    records = [case_record(
        "fixed", case, "FFD+EDD", evaluation_mode="edd",
        pair=None, metrics=metrics)]
    manifest = build_manifest("fixed", "baseline", "edd", 1, 1, None, {})
    write_evaluation_results(run_dir, manifest, records)
    with pytest.raises(FileExistsError):
        write_evaluation_results(run_dir, manifest, records)
