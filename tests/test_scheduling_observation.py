import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from core.scheduling_observation import (
    SCHED_TASK_DIM,
    SchedulingObservationLayout,
    SchedulingTaskFeature,
)
from core.scheduling_problem import SchedulingOrder, SchedulingProblem, SchedulingTask
from envs.scheduling_env import SchedulingEnv
from models.attention_extractor import AttentionFeatureExtractor


FEATURE = SchedulingTaskFeature


def _problem(tasks, orders=None, num_machines=2, scale=100.0):
    if orders is None:
        order_ids = sorted({order_id for task in tasks for order_id in task.order_ids})
        orders = tuple(
            SchedulingOrder(order_id, 200.0 + order_id, 100.0, 10.0)
            for order_id in order_ids
        )
    return SchedulingProblem(
        tasks=tuple(tasks),
        orders=tuple(orders),
        num_machines=num_machines,
        plate_w=100,
        plate_h=100,
        episode_time_scale=scale,
    )


def _task(index, processing_time=None, due=None, order_ids=(0,), area=None):
    return SchedulingTask(
        plate_index=index,
        processing_time=float(processing_time if processing_time is not None else 10 + index),
        due_date=float(due if due is not None else 100 + index),
        order_ids=tuple(order_ids),
        plate_part_area=float(area if area is not None else 100 + index),
    )


def _reset_env(tasks, *, orders=None, num_machines=2, max_tasks=6, context=None):
    env = SchedulingEnv(num_machines=num_machines, max_tasks=max_tasks)
    if context is None:
        context = np.linspace(0.1, 0.8, env.observation_layout.context_dim, dtype=np.float32)
    obs, _ = env.reset(options={
        "problem": _problem(tasks, orders, num_machines=num_machines),
        "nesting_context": context,
    })
    return env, obs, context


def _tokens(env, obs):
    layout = env.observation_layout
    return obs[layout.task_slice].reshape(layout.max_tasks, layout.task_dim)


def _extractor(layout):
    space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(layout.obs_dim,), dtype=np.float32)
    model = AttentionFeatureExtractor(space, features_dim=32, layout=layout)
    model.eval()
    return model


def _flat_observation(layout, valid_slots):
    obs = torch.zeros((1, layout.obs_dim), dtype=torch.float32)
    tasks = obs[:, layout.task_slice].reshape(
        1, layout.max_tasks, layout.task_dim)
    for slot in valid_slots:
        tasks[0, slot, FEATURE.PROCESSING_TIME] = 0.1 + slot
        tasks[0, slot, FEATURE.VALID] = 1.0
    return obs


def test_observation_length_matches_schema():
    default_layout = SchedulingObservationLayout(num_machines=3, max_tasks=120)
    assert default_layout.global_prefix_dim == 14
    assert default_layout.obs_dim == 1214

    non_default = SchedulingObservationLayout(num_machines=2, max_tasks=7)
    assert non_default.global_prefix_dim == 13
    assert non_default.obs_dim == 13 + 7 * SCHED_TASK_DIM

    env, obs, _ = _reset_env([_task(0)], num_machines=3, max_tasks=120)
    assert obs.shape == env.observation_space.shape == (default_layout.obs_dim,)


def test_observation_offsets_are_exact():
    env, obs, context = _reset_env([_task(0, processing_time=20, due=80)])
    layout = env.observation_layout
    assert np.array_equal(obs[layout.machine_slice], np.zeros(2))
    assert np.allclose(obs[layout.context_slice], context)
    assert obs[layout.dynamic_global_slice][0] == pytest.approx(20 / (2 * 100))
    assert obs[layout.dynamic_global_slice][1] == pytest.approx(80 / 100)
    assert layout.task_slice.start == layout.context_slice.stop


def test_dynamic_globals_use_only_unscheduled_tasks():
    tasks = [
        _task(0, processing_time=20, due=80, area=3000),
        _task(1, processing_time=40, due=120, area=100),
    ]
    env, obs, _ = _reset_env(tasks, num_machines=2, max_tasks=3)
    before = obs[env.observation_layout.dynamic_global_slice]
    assert before[0] == pytest.approx(60 / (2 * 100))
    assert before[1] == pytest.approx(80 / 100)
    assert before[2] == pytest.approx(0.5)

    obs, _, _, _, _ = env.step(0)
    after = obs[env.observation_layout.dynamic_global_slice]
    assert after[0] == pytest.approx(40 / (2 * 100))
    assert after[1] == pytest.approx(120 / 100)
    assert after[2] == pytest.approx(0.0)


def test_first_and_last_real_task_round_trip():
    tasks = [_task(index, processing_time=10 * (index + 1)) for index in range(4)]
    env, obs, _ = _reset_env(tasks, max_tasks=4)
    tokens = _tokens(env, obs)
    assert tokens[0, FEATURE.PROCESSING_TIME] == pytest.approx(0.1)
    assert tokens[3, FEATURE.PROCESSING_TIME] == pytest.approx(0.4)
    assert tokens[0, FEATURE.VALID] == 1.0
    assert tokens[3, FEATURE.VALID] == 1.0


def test_nesting_context_is_not_in_task_reshape():
    context = np.linspace(1.1, 1.8, 8, dtype=np.float32)
    env, obs, _ = _reset_env([_task(0)], max_tasks=3, context=context)
    assert np.allclose(obs[env.observation_layout.context_slice], context)
    assert not np.any(np.isin(_tokens(env, obs), context))


def test_padding_task_is_exactly_zero():
    env, obs, _ = _reset_env([_task(0), _task(1)], max_tasks=5)
    tokens = _tokens(env, obs)
    assert np.array_equal(tokens[2:], np.zeros((3, SCHED_TASK_DIM)))


class _RecordingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding_mask = None

    def forward(self, source, src_key_padding_mask=None):
        self.padding_mask = src_key_padding_mask.detach().clone()
        return source


def test_transformer_receives_padding_mask():
    layout = SchedulingObservationLayout(2, 4)
    extractor = _extractor(layout)
    recorder = _RecordingTransformer()
    extractor.transformer = recorder
    extractor(_flat_observation(layout, [0, 2]))
    assert recorder.padding_mask.dtype == torch.bool
    assert recorder.padding_mask.shape == (1, 4)
    assert torch.equal(
        recorder.padding_mask, torch.tensor([[False, True, False, True]]))


def test_extractor_rejects_observation_space_outside_layout():
    layout = SchedulingObservationLayout(2, 4)
    wrong_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(layout.obs_dim + 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match layout"):
        AttentionFeatureExtractor(wrong_space, features_dim=32, layout=layout)


def test_padding_hidden_values_do_not_affect_valid_output():
    torch.manual_seed(3)
    layout = SchedulingObservationLayout(2, 4)
    extractor = _extractor(layout)
    first = _flat_observation(layout, [0, 1])
    second = first.clone()
    second_tasks = second[:, layout.task_slice].reshape(1, 4, SCHED_TASK_DIM)
    second_tasks[0, 2:, :FEATURE.VALID] = 4.0
    assert torch.allclose(extractor(first), extractor(second), atol=1e-6)


def test_valid_task_count_matches_problem():
    env, obs, _ = _reset_env([_task(0), _task(1), _task(2)], max_tasks=6)
    tokens = _tokens(env, obs)
    assert np.sum(tokens[:, FEATURE.VALID] > 0.5) == len(env.task_pool)


def test_scheduled_task_and_padding_are_distinct():
    env, _, _ = _reset_env([_task(0), _task(1)], max_tasks=4)
    obs, _, _, _, _ = env.step(0)
    tokens = _tokens(env, obs)
    assert tokens[0, FEATURE.VALID] == 1.0
    assert tokens[0, FEATURE.SCHEDULED] == 1.0
    assert tokens[2, FEATURE.VALID] == 0.0
    assert tokens[2, FEATURE.SCHEDULED] == 0.0


def test_action_mask_allows_only_real_unscheduled_tasks():
    env, _, _ = _reset_env([_task(0), _task(1)], max_tasks=4, num_machines=2)
    assert np.array_equal(
        env._get_action_mask(),
        np.array([True, True, True, True, False, False, False, False]),
    )
    env.step(0)
    assert np.array_equal(
        env._get_action_mask(),
        np.array([False, False, True, True, False, False, False, False]),
    )


def test_all_scheduled_action_mask_is_all_false():
    env, _, _ = _reset_env([_task(0)], max_tasks=3, num_machines=2)
    env.step(0)
    assert not env._get_action_mask().any()


def test_all_padding_row_is_finite():
    layout = SchedulingObservationLayout(2, 4)
    extractor = _extractor(layout)
    task_tokens = torch.zeros((1, layout.max_tasks, layout.task_dim))
    sequence_context = extractor._encode_task_sequence(task_tokens)
    assert torch.isfinite(sequence_context).all()
    assert torch.equal(sequence_context, torch.zeros_like(sequence_context))
    assert torch.isfinite(extractor(torch.zeros((1, layout.obs_dim)))).all()


def test_batched_padding_mask_shape():
    layout = SchedulingObservationLayout(2, 5)
    extractor = _extractor(layout)
    recorder = _RecordingTransformer()
    extractor.transformer = recorder
    observations = torch.cat([
        _flat_observation(layout, [0]),
        _flat_observation(layout, [0, 1, 2]),
    ])
    extractor(observations)
    assert recorder.padding_mask.shape == (2, 5)
    assert torch.equal((~recorder.padding_mask).sum(dim=1), torch.tensor([1, 3]))


def test_zero_task_problem_is_rejected():
    env = SchedulingEnv(num_machines=2, max_tasks=4)
    empty_problem = _problem([], orders=(), num_machines=2)
    with pytest.raises(ValueError, match="at least one scheduling task"):
        env.reset(options={"problem": empty_problem, "nesting_context": np.zeros(8)})


def _cross_plate_fixture():
    tasks = [
        _task(index, processing_time=10 + index, due=200, order_ids=(0,))
        for index in range(4)
    ]
    tasks.append(_task(4, processing_time=50, due=250, order_ids=(1,)))
    orders = (
        SchedulingOrder(0, 200.0, 400.0, 40.0),
        SchedulingOrder(1, 250.0, 100.0, 10.0),
    )
    return _reset_env(tasks, orders=orders, num_machines=2, max_tasks=6)


def test_remaining_ratio_is_normalized_by_order_linked_tasks():
    env, _, _ = _cross_plate_fixture()
    env.step(0)
    tokens = _tokens(env, env._get_obs())
    assert tokens[1, FEATURE.MEAN_REMAINING_RATIO] == pytest.approx(0.75)
    assert tokens[1, FEATURE.MAX_REMAINING_RATIO] == pytest.approx(0.75)
    assert tokens[1, FEATURE.MEAN_REMAINING_RATIO] != pytest.approx(3 / 5)


def test_remaining_ratio_updates_after_assignment():
    env, obs, _ = _cross_plate_fixture()
    before = _tokens(env, obs)[1, FEATURE.MEAN_REMAINING_RATIO]
    env.step(0)
    after = _tokens(env, env._get_obs())[1, FEATURE.MEAN_REMAINING_RATIO]
    assert before == pytest.approx(1.0)
    assert after == pytest.approx(0.75)


def test_unresolved_fraction_feature_does_not_exist():
    assert SCHED_TASK_DIM == 10
    names = {feature.name for feature in SchedulingTaskFeature}
    assert "UNRESOLVED_LINKED_ORDER_FRACTION" not in names


def test_relative_due_changes_with_machine_progress():
    tasks = [_task(0, processing_time=10, due=100),
             _task(1, processing_time=20, due=110),
             _task(2, processing_time=30, due=120)]
    env, obs, _ = _reset_env(tasks, num_machines=2, max_tasks=3)
    before = _tokens(env, obs)[2, FEATURE.RELATIVE_DUE]
    env.step(0)  # task 0 -> machine 0
    obs, _, _, _, _ = env.step(3)  # task 1 -> machine 1
    after = _tokens(env, obs)[2, FEATURE.RELATIVE_DUE]
    assert before == pytest.approx(1.2)
    assert after == pytest.approx((120 - 10) / 100)


def test_release_fraction_only_marks_last_unscheduled_linked_task():
    env, obs, _ = _cross_plate_fixture()
    assert _tokens(env, obs)[3, FEATURE.RELEASE_FRACTION] == 0.0
    env.step(0)
    env.step(2)
    obs, _, _, _, _ = env.step(4)
    tokens = _tokens(env, obs)
    assert tokens[3, FEATURE.RELEASE_FRACTION] == pytest.approx(1.0)
    assert tokens[2, FEATURE.RELEASE_FRACTION] == 0.0


def test_cross_plate_order_features_update():
    env, obs, _ = _cross_plate_fixture()
    before = _tokens(env, obs)[1].copy()
    obs, _, _, _, _ = env.step(0)
    after = _tokens(env, obs)[1]
    assert after[FEATURE.MEAN_REMAINING_RATIO] < before[FEATURE.MEAN_REMAINING_RATIO]
    assert after[FEATURE.MAX_REMAINING_RATIO] < before[FEATURE.MAX_REMAINING_RATIO]
    assert after[FEATURE.WORST_SLACK] != pytest.approx(before[FEATURE.WORST_SLACK])


def test_optimistic_slack_does_not_mutate_completion():
    env, _, _ = _cross_plate_fixture()
    orders_before = copy.deepcopy(env.orders_snapshot)
    machines_before = env.machine_times.copy()
    scheduled_before = env.scheduled_mask.copy()
    env._get_obs()
    env._get_obs()
    assert env.orders_snapshot == orders_before
    assert np.array_equal(env.machine_times, machines_before)
    assert np.array_equal(env.scheduled_mask, scheduled_before)


def test_observation_does_not_recompute_processing_time(monkeypatch):
    import core.processing

    def forbidden(*args, **kwargs):
        raise AssertionError("processing time was recomputed while building observation")

    monkeypatch.setattr(core.processing, "plate_processing_time", forbidden)
    monkeypatch.setattr(core.processing, "parts_cutting_time", forbidden)
    env, obs, _ = _reset_env([_task(0, processing_time=37)], max_tasks=3)
    env._get_obs()
    env._get_obs()
    assert _tokens(env, obs)[0, FEATURE.PROCESSING_TIME] == pytest.approx(0.37)


def test_task_slot_position_is_observable():
    torch.manual_seed(11)
    layout = SchedulingObservationLayout(2, 4)
    extractor = _extractor(layout)
    slot_zero = _flat_observation(layout, [0])
    slot_one = _flat_observation(layout, [1])
    slot_one_tasks = slot_one[:, layout.task_slice].reshape(1, 4, SCHED_TASK_DIM)
    slot_one_tasks[0, 1, FEATURE.PROCESSING_TIME] = 0.1
    with torch.no_grad():
        first = extractor(slot_zero)
        second = extractor(slot_one)
    assert not torch.allclose(first, second)
