import copy

import numpy as np
import pytest

from core.instance import ProductionInstance
from envs.packing_envs import NestingSchedulingEnv
from heuristic.blf_skyline_maxrects import PlateLayoutManager


def _part(w, h, order_id, due=100.0, original_idx=0):
    return {
        "w": w,
        "h": h,
        "area": w * h,
        "due_date": due,
        "order_id": order_id,
        "original_idx": original_idx,
    }


def _make_env(parts, plate_size=(10, 10)):
    orders = {
        part["order_id"]: {"due_date": part["due_date"], "finished_time": 0.0}
        for part in parts
    }
    instance = ProductionInstance(
        parts=copy.deepcopy(parts),
        orders=orders,
        plate_w=plate_size[0],
        plate_h=plate_size[1],
        num_machines=3,
        seed=1,
    )
    env = NestingSchedulingEnv(plate_size=plate_size)
    env.reset(options={"instance": instance})
    return env


def _plate_state(plate):
    return (
        tuple(plate.placed_parts),
        plate.used_area,
        tuple(plate.free_rects),
        tuple(plate.skyline),
        tuple(plate.blf_points),
    )


def _env_state(env):
    return {
        "packed": set(env.packed_indices),
        "active_ids": tuple(id(plate) for plate in env.active_plates),
        "active": tuple(_plate_state(plate) for plate in env.active_plates),
        "history_ids": tuple(id(plate) for plate in env.history_plates),
        "history": tuple(_plate_state(plate) for plate in env.history_plates),
        "orders": copy.deepcopy(env.orders),
        "scheduler": tuple(env.scheduler_state_machine.get_state()),
        "cost": copy.deepcopy(env.cost_metrics),
    }


def _fill_plate(plate):
    candidate = plate.find_fixed_placement(
        plate.width, plate.height, -1, 0, 2)
    assert candidate is not None
    assert plate.commit_fixed_placement(candidate)


def _place_fixed(plate, width, height):
    candidate = plate.find_fixed_placement(width, height, -1, 0, 2)
    assert candidate is not None
    assert plate.commit_fixed_placement(candidate)


def test_nesting_action_decode_round_trip():
    env = _make_env([_part(2, 3, 0)])
    for part_index in range(env.max_capacity):
        for rotation in range(2):
            for strategy in range(3):
                action = env.encode_action(part_index, rotation, strategy)
                assert env.decode_action(action) == (
                    part_index, rotation, strategy)


def test_action_slot_places_exact_selected_part():
    env = _make_env([_part(2, 3, 10, original_idx=0),
                     _part(2, 3, 20, original_idx=1)])
    env.step(env.encode_action(1, 0, 0))
    assert env.packed_indices == {1}
    assert env.active_plates[0].placed_parts[-1][4] == 20


def test_duplicate_geometry_parts_keep_slot_identity():
    env = _make_env([_part(4, 2, 1), _part(4, 2, 2)])
    env.step(env.encode_action(1, 1, 2))
    placed = env.active_plates[0].placed_parts[-1]
    assert env.packed_indices == {1}
    assert placed[4] == 2
    assert placed[2:4] == (2, 4)


def test_rotation_zero_never_rotates():
    env = _make_env([_part(4, 2, 0)], plate_size=(5, 5))
    _place_fixed(env.active_plates[0], 3, 5)
    env.step(env.encode_action(0, 0, 2))
    assert len(env.active_plates) == 2
    assert env.active_plates[1].placed_parts[-1][2:] == (4, 2, 0, False)


def test_rotation_one_always_rotates():
    env = _make_env([_part(4, 2, 0)], plate_size=(5, 5))
    _place_fixed(env.active_plates[0], 5, 3)
    env.step(env.encode_action(0, 1, 2))
    assert len(env.active_plates) == 2
    assert env.active_plates[1].placed_parts[-1][2:] == (2, 4, 0, True)


@pytest.mark.parametrize("rotation", [0, 1])
def test_square_part_rotation_metadata_matches_action(rotation):
    env = _make_env([_part(3, 3, 0)], plate_size=(5, 5))
    env.step(env.encode_action(0, rotation, 0))
    placed = env.active_plates[0].placed_parts[-1]
    assert placed[2:4] == (3, 3)
    assert placed[-1] is bool(rotation)


def test_rotation_zero_masked_when_only_rotated_fits():
    env = _make_env([_part(4, 2, 0)], plate_size=(3, 5))
    mask = env._get_action_mask()
    assert not mask[env.encode_action(0, 0, 0):env.encode_action(0, 0, 2) + 1].any()
    assert mask[env.encode_action(0, 1, 0):env.encode_action(0, 1, 2) + 1].all()


def test_rotation_one_masked_when_only_original_fits():
    env = _make_env([_part(2, 4, 0)], plate_size=(3, 5))
    mask = env._get_action_mask()
    assert mask[env.encode_action(0, 0, 0):env.encode_action(0, 0, 2) + 1].all()
    assert not mask[env.encode_action(0, 1, 0):env.encode_action(0, 1, 2) + 1].any()


def _assert_only_selected_finder_called(monkeypatch, strategy, expected):
    calls = {"blf": 0, "skyline": 0, "maxrects": 0}
    originals = {
        "blf": PlateLayoutManager._find_blf,
        "skyline": PlateLayoutManager._find_skyline_wei,
        "maxrects": PlateLayoutManager._find_maxrects,
    }

    def wrap(name):
        def recorder(self, *args, **kwargs):
            calls[name] += 1
            return originals[name](self, *args, **kwargs)
        return recorder

    monkeypatch.setattr(PlateLayoutManager, "_find_blf", wrap("blf"))
    monkeypatch.setattr(PlateLayoutManager, "_find_skyline_wei", wrap("skyline"))
    monkeypatch.setattr(PlateLayoutManager, "_find_maxrects", wrap("maxrects"))
    env = _make_env([_part(2, 3, 0)])
    calls.update({name: 0 for name in calls})
    env.step(env.encode_action(0, 0, strategy))
    assert calls[expected] > 0
    assert all(count == 0 for name, count in calls.items() if name != expected)


def test_blf_action_calls_only_blf(monkeypatch):
    _assert_only_selected_finder_called(monkeypatch, 0, "blf")


def test_skyline_action_calls_only_skyline(monkeypatch):
    _assert_only_selected_finder_called(monkeypatch, 1, "skyline")


def test_maxrects_action_calls_only_maxrects(monkeypatch):
    _assert_only_selected_finder_called(monkeypatch, 2, "maxrects")


def test_failed_strategy_does_not_silently_switch_strategy(monkeypatch):
    maxrects_calls = 0

    def fail_blf(self, w, h):
        return None

    def record_maxrects(self, w, h):
        nonlocal maxrects_calls
        maxrects_calls += 1
        return (0.0, 0.0)

    monkeypatch.setattr(PlateLayoutManager, "_find_blf", fail_blf)
    monkeypatch.setattr(PlateLayoutManager, "_find_maxrects", record_maxrects)
    env = _make_env([_part(2, 3, 0)])
    action = env.encode_action(0, 0, 0)
    assert not env._get_action_mask()[action]
    maxrects_calls = 0
    with pytest.raises(ValueError, match="infeasible"):
        env.step(action)
    assert maxrects_calls == 0


def test_fixed_placement_proposal_is_pure():
    plate = PlateLayoutManager(10, 10)
    before = _plate_state(plate)
    candidate = plate.find_fixed_placement(4, 2, 7, 1, 1)
    assert candidate is not None
    assert _plate_state(plate) == before
    assert (candidate.placed_w, candidate.placed_h) == (2, 4)
    assert candidate.rotated is True
    assert candidate.strategy_id == 1


def test_mask_and_step_share_exact_candidate_contract(monkeypatch):
    calls = 0
    original = PlateLayoutManager.find_fixed_placement

    def recorder(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PlateLayoutManager, "find_fixed_placement", recorder)
    env = _make_env([_part(2, 3, 0)])
    action = env.encode_action(0, 1, 2)
    assert env._get_action_mask()[action]
    calls_before_step = calls
    env.step(action)
    assert calls > calls_before_step
    assert env.active_plates[0].placed_parts[-1][-1] is True


def test_candidate_commit_failure_is_internal_error(monkeypatch):
    env = _make_env([_part(2, 3, 0)])
    env.active_plates.append(PlateLayoutManager(env.plate_w, env.plate_h))
    action = env.encode_action(0, 0, 0)
    before = _env_state(env)
    proposal_targets = []
    original_proposal = PlateLayoutManager.find_fixed_placement

    def record_proposal(self, *args, **kwargs):
        proposal_targets.append(id(self))
        return original_proposal(self, *args, **kwargs)

    monkeypatch.setattr(
        PlateLayoutManager, "find_fixed_placement", record_proposal)
    monkeypatch.setattr(
        PlateLayoutManager, "commit_fixed_placement", lambda self, candidate: False)
    with pytest.raises(RuntimeError, match="could not be committed"):
        env.step(action)
    assert _env_state(env) == before
    assert proposal_targets == [id(env.active_plates[0])]


def test_plate_selection_policy_is_deterministic_and_documented():
    env = _make_env([_part(2, 3, 0)])
    env.active_plates.append(PlateLayoutManager(env.plate_w, env.plate_h))
    env.step(env.encode_action(0, 0, 0))
    assert len(env.active_plates[0].placed_parts) == 1
    assert len(env.active_plates[1].placed_parts) == 0


@pytest.mark.parametrize("rotation", [0, 1])
def test_new_plate_preserves_rotation_choice(rotation):
    env = _make_env([_part(4, 2, 0)], plate_size=(5, 5))
    _fill_plate(env.active_plates[0])
    env.step(env.encode_action(0, rotation, 1))
    placed = env.active_plates[1].placed_parts[-1]
    assert placed[2:4] == ((4, 2) if rotation == 0 else (2, 4))
    assert placed[-1] is bool(rotation)


@pytest.mark.parametrize("strategy", [0, 1, 2])
def test_new_plate_preserves_strategy_choice(monkeypatch, strategy):
    calls = []
    original = PlateLayoutManager._try_strategies

    def recorder(self, w, h, selected, min_w, min_h):
        calls.append(selected)
        return original(self, w, h, selected, min_w, min_h)

    monkeypatch.setattr(PlateLayoutManager, "_try_strategies", recorder)
    env = _make_env([_part(2, 3, 0)])
    _fill_plate(env.active_plates[0])
    calls.clear()
    env.step(env.encode_action(0, 0, strategy))
    assert calls and set(calls) == {strategy}


def test_every_masked_valid_action_executes():
    env = _make_env([_part(2, 3, 0)], plate_size=(5, 5))
    mask = env._get_action_mask()
    for action in np.flatnonzero(mask):
        trial = copy.deepcopy(env)
        part_index, rotation, strategy = trial.decode_action(int(action))
        trial.step(int(action))
        placed = trial.active_plates[0].placed_parts[-1]
        assert trial.packed_indices == {part_index}
        assert placed[2:4] == ((2, 3) if rotation == 0 else (3, 2))
        assert placed[-1] is bool(rotation)
        assert strategy in (0, 1, 2)


def test_masked_invalid_action_raises_without_state_mutation():
    env = _make_env([_part(4, 2, 0)], plate_size=(3, 5))
    action = env.encode_action(0, 0, 2)
    assert not env._get_action_mask()[action]
    before = _env_state(env)
    with pytest.raises(ValueError, match="infeasible"):
        env.step(action)
    assert _env_state(env) == before


def test_failed_placement_does_not_mark_part_packed():
    env = _make_env([_part(9, 9, 0)], plate_size=(5, 5))
    action = env.encode_action(0, 0, 0)
    with pytest.raises(ValueError, match="infeasible"):
        env.step(action)
    assert env.packed_indices == set()
    assert all(not plate.placed_parts for plate in env.active_plates)


def test_terminal_action_mask_is_all_false():
    env = _make_env([_part(2, 3, 0)])
    env.step(env.encode_action(0, 0, 0))
    mask = env._get_action_mask()
    assert mask.shape == (720,)
    assert not mask.any()
