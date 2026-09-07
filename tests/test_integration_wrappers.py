import numpy as np
import pytest

from core.instance import generate_instance
from core.scheduling_problem import NestingTerminalResult, NestedPlateResult


def _result(seed=55):
    instance = generate_instance(seed=seed, num_parts=2, plate_size=(100, 100), num_machines=1)
    parts = tuple((0.0, 0.0, float(p["w"]), float(p["h"]), int(p["order_id"]), False)
                  for p in instance.parts)
    return NestingTerminalResult(instance, (NestedPlateResult(0, parts),))


class NestingPolicy:
    def predict(self, obs, action_masks=None, deterministic=True):
        return 0, None


def test_provider_handles_repeated_resets_and_keeps_scheduling_env_independent():
    import gymnasium as gym
    from sb3_contrib.common.wrappers import ActionMasker
    from envs.scheduling_env import SchedulingEnv
    from integration.scheduling_problem_provider import SchedulingProblemProviderWrapper

    class Upstream(gym.Env):
        def reset(self, *, seed=None, options=None):
            return np.zeros(1), {}
        def step(self, action):
            return np.zeros(1), 0.0, True, False, {"terminal_result": _result()}
        def _get_action_mask(self):
            return np.ones(1, dtype=bool)

    base = SchedulingEnv(num_machines=1, max_tasks=4)
    provider = SchedulingProblemProviderWrapper(base, Upstream(), NestingPolicy())
    provider.reset(seed=5)
    first = list(base.task_pool)
    provider.reset(seed=5)
    assert base.task_pool == first
    assert provider.last_episode_seed == 5
    provider.reset()
    assert provider.last_episode_seed != 5
    assert not hasattr(base, "nesting_env")
    masked = ActionMasker(provider, lambda env: env.get_wrapper_attr("_get_action_mask")())
    masked.reset(seed=5)
    assert masked.action_masks().any()


def test_terminal_wrapper_uses_fresh_env_and_fails_fast():
    import gymnasium as gym
    from envs.scheduling_env import SchedulingEnv
    from integration.scheduling_terminal_reward import (
        SchedulingPolicyRolloutError, SchedulingTerminalRewardWrapper,
    )

    class TerminalBase(gym.Env):
        def __init__(self):
            super().__init__()
            self.w_terminal = 10.0
            self.cost_metrics = {}
        def reset(self, *, seed=None, options=None):
            return np.zeros(1), {}
        def step(self, action):
            return np.zeros(1), 0.0, True, False, {"terminal_result": _result()}
        def _compute_metrics(self):
            return self.cost_metrics.copy()

    live_training_env = SchedulingEnv(num_machines=1, max_tasks=4)
    wrapper = SchedulingTerminalRewardWrapper(TerminalBase(), NestingPolicy())
    _, reward_1, done, _, _ = wrapper.step(0)
    ema_after_first = wrapper.reward_transform._cost_ema
    _, reward_2, _, _, _ = wrapper.step(0)
    assert done and reward_1 == pytest.approx(reward_2)
    assert wrapper.reward_transform._cost_ema == ema_after_first
    assert wrapper.last_evaluator_id != id(live_training_env)

    class FailingPolicy:
        def predict(self, *args, **kwargs):
            raise RuntimeError("policy failure")
    with pytest.raises(SchedulingPolicyRolloutError):
        SchedulingTerminalRewardWrapper(TerminalBase(), FailingPolicy()).step(0)
