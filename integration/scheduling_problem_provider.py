"""SB3 reset provider that creates explicit scheduling problems externally."""

import gymnasium as gym
import numpy as np

from core.cooperative_context import build_nesting_context, build_scheduling_context
from core.instance import generate_instance
from core.scheduling_problem import build_scheduling_problem


class SchedulingProblemProviderWrapper(gym.Wrapper):
    """Own upstream nesting rollout without exposing it to SchedulingEnv."""

    def __init__(self, env, nesting_env, nesting_policy, instance_factory=None):
        super().__init__(env)
        self.nesting_env = nesting_env
        self.nesting_policy = nesting_policy
        self.instance_factory = instance_factory or (
            lambda seed: generate_instance(seed=seed, num_machines=self.env.unwrapped.num_machines)
        )
        self._provider_rng = np.random.default_rng()
        self.last_episode_seed = None

    def _get_action_mask(self):
        return self.env.unwrapped._get_action_mask()

    def reset(self, *, seed=None, options=None):
        # Do not call gym.Wrapper.reset(): it forwards a bare reset to the
        # SchedulingEnv before this provider has built its explicit problem.
        if seed is not None:
            self._provider_rng = np.random.default_rng(seed)
            episode_seed = int(seed)
        else:
            episode_seed = int(self._provider_rng.integers(0, 2**32 - 1))
        self.last_episode_seed = episode_seed

        instance = self.instance_factory(episode_seed)
        scheduling_context = build_scheduling_context(instance)
        obs, _ = self.nesting_env.reset(
            seed=episode_seed,
            options={"instance": instance, "scheduling_context": scheduling_context},
        )
        if hasattr(self.nesting_policy, "reset_cache"):
            self.nesting_policy.reset_cache()
        done = False
        terminal_info = None
        while not done:
            try:
                mask = self.nesting_env.action_masks()
            except AttributeError:
                mask = self.nesting_env.unwrapped._get_action_mask()
            action, _ = self.nesting_policy.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, terminal_info = self.nesting_env.step(action)
            done = terminated or truncated
        if not terminal_info or "terminal_result" not in terminal_info:
            raise RuntimeError("Upstream nesting rollout ended without terminal_result")
        result = terminal_info["terminal_result"]
        problem = build_scheduling_problem(result.instance, result.plates)
        nesting_context = build_nesting_context(result.instance, result.plates)
        return self.env.reset(
            seed=episode_seed,
            options={"problem": problem, "nesting_context": nesting_context},
        )
