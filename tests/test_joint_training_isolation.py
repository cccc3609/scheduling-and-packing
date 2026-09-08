import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.scheduling_problem import SchedulingOrder, SchedulingProblem, SchedulingTask
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_terminal_reward import SchedulingEvaluator
from train_dual import (
    NestingPPO,
    load_phase3_pair_metadata,
    start_fresh_scheduling_block,
    train_phase1,
    train_phase2,
    train_phase3_round,
    write_phase3_pair_metadata,
)
from train_resume import (
    build_isolated_scheduling_training,
    build_resume_nesting_env,
    build_resume_nesting_trainer,
    resolve_resume_checkpoints,
    run_resume_cycle,
)


class _TerminalSpy:
    def __init__(self, events):
        self.events = events
        self.mode = None
        self.policy = None

    def set_evaluator(self, mode, scheduling_policy=None):
        self.mode = mode
        self.policy = scheduling_policy
        self.events.append(f"evaluator:{mode}")


class _NestingSpy:
    def __init__(self, events, version=0):
        self.events = events
        self.version = version

    def learn(self, steps, save_path=None, tag=None, **kwargs):
        self.events.append("nesting.learn")
        self.version += 1
        if save_path and tag:
            Path(save_path, f"{tag}_final.pt").write_bytes(b"nesting")


class _VersionedSchedulingEnv:
    def __init__(self, nesting):
        self.nesting = nesting
        self.first_problem_version = None

    def reset(self):
        self.first_problem_version = self.nesting.version
        return np.zeros(1, dtype=np.float32)


class _SchedulingSpy:
    def __init__(self, events):
        self.events = events
        self._last_obs = np.asarray([999.0])
        self.env = None
        self.learn_calls = 0
        self.rollout_buffer = []

    def set_env(self, env, force_reset=True):
        self.events.append("scheduling.set_env")
        self.env = env
        if force_reset:
            self._last_obs = None

    def learn(self, steps, reset_num_timesteps=False):
        assert self._last_obs is None
        self.rollout_buffer.clear()
        self._last_obs = self.env.reset()
        self.rollout_buffer.append(self._last_obs.copy())
        self.learn_calls += 1
        self.events.append("scheduling.learn")

    def save(self, path):
        target = Path(path)
        if target.suffix != ".zip":
            target = target.with_suffix(".zip")
        target.write_bytes(b"scheduling")


def test_phase1_updates_only_nesting(tmp_path):
    events = []
    nesting = _NestingSpy(events)
    terminal = _TerminalSpy(events)
    train_phase1(nesting, terminal, 2, tmp_path)
    assert nesting.version == 1
    assert events == ["evaluator:edd", "nesting.learn"]


def test_phase2_updates_only_scheduling(tmp_path):
    events = []
    nesting = _NestingSpy(events)
    scheduling = _SchedulingSpy(events)
    env = _VersionedSchedulingEnv(nesting)
    train_phase2(scheduling, env, 2, tmp_path)
    assert nesting.version == 0
    assert scheduling.learn_calls == 1


def test_phase3_updates_both_agents_in_order(tmp_path):
    events = []
    nesting = _NestingSpy(events)
    terminal = _TerminalSpy(events)
    scheduling = _SchedulingSpy(events)
    env = _VersionedSchedulingEnv(nesting)
    train_phase3_round(nesting, terminal, scheduling, env, 2, tmp_path, 1)
    assert events == [
        "evaluator:policy", "nesting.learn",
        "scheduling.set_env", "scheduling.learn",
    ]


def test_phase3_scheduling_block_starts_with_latest_nesting_problem(tmp_path):
    events = []
    nesting = _NestingSpy(events, version=4)
    terminal = _TerminalSpy(events)
    scheduling = _SchedulingSpy(events)
    env = _VersionedSchedulingEnv(nesting)
    scheduling.env = env
    scheduling._last_obs = env.reset()
    assert env.first_problem_version == 4

    train_phase3_round(nesting, terminal, scheduling, env, 2, tmp_path, 3)

    assert nesting.version == 5
    assert env.first_problem_version == 5
    assert scheduling.rollout_buffer[0].shape == (1,)


def _one_task_problem():
    return SchedulingProblem(
        tasks=(SchedulingTask(0, 2.0, 10.0, (0,), 10.0),),
        orders=(SchedulingOrder(0, 10.0, 10.0, 2.0),),
        num_machines=1,
        plate_w=10,
        plate_h=10,
        episode_time_scale=10.0,
    )


def test_terminal_evaluator_does_not_update_scheduling_model():
    class Policy:
        def __init__(self):
            self.weight = torch.tensor([3.0], requires_grad=True)
            self.optimizer_steps = 0
            self.rollout_buffer = []

        def predict(self, obs, action_masks=None, deterministic=True):
            return int(np.flatnonzero(action_masks)[0]), None

    policy = Policy()
    before = policy.weight.detach().clone()
    SchedulingEvaluator("policy", policy).evaluate(
        _one_task_problem(), np.zeros(8, dtype=np.float32))
    assert torch.equal(policy.weight, before)
    assert policy.optimizer_steps == 0
    assert policy.rollout_buffer == []


def test_problem_generation_does_not_update_nesting_model():
    wrapped = build_resume_nesting_env("phase2")
    ppo = build_resume_nesting_trainer(wrapped.env, device="cpu")
    scheduling_env, provider_env = build_isolated_scheduling_training(ppo)
    before = [parameter.detach().clone() for parameter in ppo.model.parameters()]
    scheduling_env.reset(seed=17)
    assert all(torch.equal(old, new) for old, new in zip(before, ppo.model.parameters()))
    assert ppo.last_rollout_buffer is None
    assert provider_env.packed_indices


def test_resume_training_and_provider_nesting_envs_are_distinct():
    wrapped = build_resume_nesting_env("phase3")
    ppo = build_resume_nesting_trainer(wrapped.env, device="cpu")
    scheduling_env, provider_env = build_isolated_scheduling_training(ppo)
    assert provider_env is not ppo.env.unwrapped
    assert scheduling_env.env.nesting_env is provider_env


def test_training_and_terminal_scheduling_envs_are_distinct():
    created = []

    def factory(num_machines):
        env = SchedulingEnv(num_machines=num_machines, max_tasks=2)
        created.append(env)
        return env

    class Policy:
        def predict(self, obs, action_masks=None, deterministic=True):
            return int(np.flatnonzero(action_masks)[0]), None

    training_env = SchedulingEnv(num_machines=1, max_tasks=2)
    SchedulingEvaluator("policy", Policy(), factory).evaluate(
        _one_task_problem(), np.zeros(8, dtype=np.float32))
    assert len(created) == 1
    assert created[0] is not training_env


def test_phase3_uses_latest_scheduling_policy(tmp_path):
    events = []
    terminal = _TerminalSpy(events)
    scheduling = _SchedulingSpy(events)
    train_phase3_round(
        _NestingSpy(events), terminal, scheduling,
        _VersionedSchedulingEnv(_NestingSpy([])), 1, tmp_path, 1)
    assert terminal.policy is scheduling


def test_phase3_provider_uses_latest_nesting_policy():
    wrapped = build_resume_nesting_env("phase3")
    ppo = build_resume_nesting_trainer(wrapped.env, device="cpu")
    scheduling_env, _ = build_isolated_scheduling_training(ppo)
    assert scheduling_env.env.nesting_policy.ppo is ppo


def test_nesting_checkpoint_restores_model_optimizer_and_counter(tmp_path):
    model = nn.Linear(2, 1)
    source = NestingPPO(object(), model, n_steps=1, batch_size=1, n_epochs=1)
    loss = source.model(torch.ones(1, 2)).sum()
    source.optimizer.zero_grad()
    loss.backward()
    source.optimizer.step()
    source.total_steps = 123
    checkpoint = tmp_path / "nesting_joint_c2_final.pt"
    source.save_training_checkpoint(
        checkpoint, phase="phase3", round_id=2)

    restored = NestingPPO(
        object(), nn.Linear(2, 1), n_steps=1, batch_size=1, n_epochs=1)
    restored.load_training_checkpoint(
        checkpoint, expected_phase="phase3", expected_round=2)
    assert restored.total_steps == 123
    assert restored.optimizer.state_dict()["state"]
    assert all(torch.equal(a, b) for a, b in zip(
        source.model.parameters(), restored.model.parameters()))


def test_phase3_checkpoint_pair_is_consistent(tmp_path):
    paths = {
        "nest": tmp_path / "nesting_joint_c4_final.pt",
        "sched": tmp_path / "scheduling_joint_c4.zip",
    }
    paths["nest"].write_bytes(b"n")
    paths["sched"].write_bytes(b"s")
    metadata_path = write_phase3_pair_metadata(tmp_path, 4)
    metadata = load_phase3_pair_metadata(tmp_path, 4)
    assert metadata["phase"] == "phase3" and metadata["round"] == 4
    assert Path(metadata_path).is_file()

    Path(metadata_path).write_text(json.dumps({**metadata, "round": 3}))
    with pytest.raises(ValueError, match="inconsistent"):
        load_phase3_pair_metadata(tmp_path, 4)


def test_phase3_resume_restores_paired_agents(tmp_path):
    (tmp_path / "nesting_joint_c6_final.pt").write_bytes(b"n")
    (tmp_path / "scheduling_joint_c6.zip").write_bytes(b"s")
    write_phase3_pair_metadata(tmp_path, 6)
    nest, sched, phase, round_id = resolve_resume_checkpoints(
        tmp_path, "phase3", 6)
    assert Path(nest).name == "nesting_joint_c6_final.pt"
    assert Path(sched).name == "scheduling_joint_c6.zip"
    assert (phase, round_id) == ("phase3", 6)


@pytest.mark.parametrize(
    ("phase", "nesting_updates", "scheduling_updates"),
    [("phase1", 1, 0), ("phase2", 0, 1), ("phase3", 1, 1)],
)
def test_resume_phase_semantics(
        tmp_path, phase, nesting_updates, scheduling_updates):
    events = []
    nesting = _NestingSpy(events)
    terminal = _TerminalSpy(events)
    scheduling = _SchedulingSpy(events) if phase != "phase1" else None
    env = _VersionedSchedulingEnv(nesting) if scheduling else None
    run_resume_cycle(
        phase, nesting, terminal, scheduling, env, 1, tmp_path, 1)
    assert nesting.version == nesting_updates
    assert (scheduling.learn_calls if scheduling else 0) == scheduling_updates


def test_invalid_resume_phase_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="phase1.*phase2.*phase3"):
        build_resume_nesting_env("phase4")
    with pytest.raises(ValueError, match="phase1.*phase2.*phase3"):
        resolve_resume_checkpoints(tmp_path, "phase4", 0)


def test_start_fresh_block_discards_stale_sb3_observation():
    model = _SchedulingSpy([])
    env = _VersionedSchedulingEnv(_NestingSpy([]))
    assert model._last_obs is not None
    start_fresh_scheduling_block(model, env)
    assert model._last_obs is None
