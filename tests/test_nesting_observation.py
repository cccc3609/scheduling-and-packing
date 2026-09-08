import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from core.nesting_observation import (
    NESTING_CONTEXT_DIM,
    NESTING_GLOBAL_DIM,
    NESTING_PART_DIM,
    NESTING_SKYLINE_DIM,
    NestingObservationLayout,
    NestingPartFeature,
)
from envs.packing_envs import NestingSchedulingEnv
from evaluate_batch import run_rl_episode as run_batch_episode
from evaluate_generalization import run_rl_episode as run_generalization_episode
from models.pointer_extractor import (
    LEGACY_NESTING_CHECKPOINT_ERROR,
    NestingModel,
    PartEncoder,
    PointerActorHead,
    StepDecoder,
    load_nesting_state_dict_strict,
)
from models.pointer_policy import PointerActorCriticPolicy, PointerFeatureExtractor
from train_dual import NestingModelPredictor, NestingPPO


def _small_layout(max_parts=4):
    return NestingObservationLayout(max_parts=max_parts)


def _tokens(layout, real=2):
    tokens = torch.zeros(1, layout.max_parts, layout.part_dim)
    for index in range(real):
        tokens[0, index, :4] = torch.tensor(
            [0.1 + index * 0.1, 0.2, 0.02 + index * 0.01, 0.5])
        tokens[0, index, layout.valid_index] = 1.0
    return tokens


def _first_valid_action(env):
    return int(np.flatnonzero(env.unwrapped._get_action_mask())[0])


class _TwoPartEnv(NestingSchedulingEnv):
    def reset(self, *, seed=None, options=None):
        return super().reset(
            seed=7 if seed is None else seed,
            options={"num_parts": 2, "plate_size": (200, 200)},
        )


class _ThreePartEnv(NestingSchedulingEnv):
    def reset(self, *, seed=None, options=None):
        return super().reset(
            seed=11 if seed is None else seed,
            options={"num_parts": 3, "plate_size": (200, 200)},
        )


class _CountingModel(NestingModel):
    def __init__(self, layout):
        super().__init__(layout=layout, embed_dim=16, n_heads=4, n_enc_layers=1)
        self.decision_inputs = []

    def forward_decision(self, part_feats, state_feat, action_mask=None):
        self.decision_inputs.append(part_feats.detach().cpu().clone())
        return super().forward_decision(part_feats, state_feat, action_mask)


def test_nesting_observation_schema_matches_layout():
    layout = NestingObservationLayout()
    env = NestingSchedulingEnv()
    obs, _ = env.reset(seed=1, options={"num_parts": 3})
    assert layout.part_dim == NESTING_PART_DIM == 6
    assert layout.state_dim == NESTING_GLOBAL_DIM + NESTING_SKYLINE_DIM + NESTING_CONTEXT_DIM == 57
    assert layout.part_block_dim == 720
    assert layout.obs_dim == 777
    assert layout.part_slice == slice(0, 720)
    assert layout.global_slice == slice(720, 741)
    assert layout.skyline_slice == slice(741, 761)
    assert layout.context_slice == slice(761, 777)
    assert layout.state_slice == slice(720, 777)
    assert obs.shape == env.observation_space.shape == (layout.obs_dim,)


def test_nondefault_skyline_layout_propagates_to_env_and_model():
    layout = NestingObservationLayout(max_parts=4, skyline_dim=12)
    env = NestingSchedulingEnv(observation_layout=layout)
    obs, _ = env.reset(
        seed=12, options={"num_parts": 2, "plate_size": (200, 200)})
    model = NestingModel(
        layout=env.layout, embed_dim=16, n_heads=4, n_enc_layers=1).eval()
    extractor = PointerFeatureExtractor(
        env.observation_space, layout=env.layout,
        embed_dim=16, n_heads=4, n_layers=1).eval()
    obs_tensor = torch.as_tensor(obs).unsqueeze(0)
    tokens, context = extractor._get_tokens_and_context(obs_tensor)
    mask = torch.as_tensor(env._get_action_mask()).unsqueeze(0)
    logits, value = model.forward_decision(
        torch.as_tensor(env.get_part_feats()).unsqueeze(0),
        torch.as_tensor(env.get_state_feat()).unsqueeze(0), mask)
    assert env.layout is layout
    assert model.layout is env.layout
    assert extractor.layout is env.layout
    assert env.layout.skyline_dim == 12
    assert obs.shape == (env.layout.obs_dim,)
    assert obs[env.layout.skyline_slice].shape == (12,)
    assert tokens.shape[:2] == (1, layout.max_parts)
    assert context.shape == (1, 16)
    assert logits.shape == (1, env.action_space.n)
    assert value.shape == (1, 1)


def test_context_dim_has_single_authority():
    env = NestingSchedulingEnv()
    assert env.COMM_DIM_IN == env.layout.context_dim == NESTING_CONTEXT_DIM
    incompatible = NestingObservationLayout(context_dim=8)
    with pytest.raises(ValueError, match="context is fixed"):
        NestingSchedulingEnv(observation_layout=incompatible)


def test_observation_width_is_strict():
    layout = _small_layout()
    wrong = gym.spaces.Box(-1.0, 1.0, shape=(layout.obs_dim - 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected nesting observation width"):
        PointerFeatureExtractor(wrong, layout=layout, embed_dim=16, n_heads=4, n_layers=1)


def test_padding_token_is_zero_and_invalid():
    env = NestingSchedulingEnv()
    env.reset(seed=2, options={"num_parts": 2})
    feats = env.get_part_feats()
    assert np.all(feats[2:] == 0.0)
    assert np.all(feats[:2, env.observation_layout.valid_index] == 1.0)
    assert np.all(feats[2:, env.observation_layout.valid_index] == 0.0)


def test_validity_uses_only_valid_feature():
    layout = _small_layout()
    encoder = PartEncoder(layout=layout, embed_dim=16, n_heads=4, n_layers=1).eval()
    tokens = torch.ones(1, layout.max_parts, layout.part_dim)
    tokens[..., layout.valid_index] = 0.0
    valid = encoder.get_valid_mask(tokens)
    encoded = encoder(tokens)
    assert not valid.any()
    assert torch.equal(encoded, torch.zeros_like(encoded))


def test_packed_real_part_remains_valid():
    env = NestingSchedulingEnv()
    env.reset(seed=3, options={"num_parts": 2})
    action = _first_valid_action(env)
    part_index, _, _ = env.decode_action(action)
    env.step(action)
    token = env.get_part_feats()[part_index]
    assert token[env.observation_layout.valid_index] == 1.0
    assert token[env.observation_layout.packed_index] == 1.0


def test_packed_flag_updates_after_successful_step():
    env = NestingSchedulingEnv()
    env.reset(seed=4, options={"num_parts": 2})
    action = _first_valid_action(env)
    part_index, _, _ = env.decode_action(action)
    before = env.get_part_feats().copy()
    env.step(action)
    after = env.get_part_feats()
    assert before[part_index, env.observation_layout.packed_index] == 0.0
    assert after[part_index, env.observation_layout.packed_index] == 1.0


def test_policy_sees_updated_packed_feature_next_step():
    env = NestingSchedulingEnv()
    env.reset(seed=5, options={"num_parts": 2})
    layout = env.observation_layout
    model = _CountingModel(layout).eval()
    pf0 = torch.as_tensor(env.get_part_feats()).unsqueeze(0)
    sf0 = torch.as_tensor(env.get_state_feat()).unsqueeze(0)
    mask0 = torch.as_tensor(env._get_action_mask()).unsqueeze(0)
    model.forward_decision(pf0, sf0, mask0)
    action = _first_valid_action(env)
    part_index, _, _ = env.decode_action(action)
    env.step(action)
    pf1 = torch.as_tensor(env.get_part_feats()).unsqueeze(0)
    sf1 = torch.as_tensor(env.get_state_feat()).unsqueeze(0)
    mask1 = torch.as_tensor(env._get_action_mask()).unsqueeze(0)
    model.forward_decision(pf1, sf1, mask1)
    assert model.decision_inputs[0][0, part_index, layout.packed_index] == 0
    assert model.decision_inputs[1][0, part_index, layout.packed_index] == 1


def test_part_encoder_masks_padding_tokens():
    torch.manual_seed(1)
    layout = _small_layout()
    encoder = PartEncoder(layout=layout, embed_dim=16, n_heads=4, n_layers=1).eval()
    clean = _tokens(layout, real=2)
    polluted = clean.clone()
    polluted[:, 2:, :layout.valid_index] = 999.0
    out_clean = encoder(clean)
    out_polluted = encoder(polluted)
    assert torch.allclose(out_clean[:, :2], out_polluted[:, :2], atol=1e-6)
    assert torch.equal(out_polluted[:, 2:], torch.zeros_like(out_polluted[:, 2:]))


def test_step_decoder_masks_padding_tokens():
    torch.manual_seed(2)
    layout = _small_layout()
    decoder = StepDecoder(layout=layout, embed_dim=16, n_heads=4).eval()
    state = torch.randn(1, layout.state_dim)
    valid = torch.tensor([[True, True, False, False]])
    clean_h = torch.randn(1, layout.max_parts, 16)
    polluted_h = clean_h.clone()
    polluted_h[:, 2:] = 1e5
    clean_context, _ = decoder(state, clean_h, valid)
    polluted_context, _ = decoder(state, polluted_h, valid)
    assert torch.allclose(clean_context, polluted_context, atol=1e-6)


def test_polluted_invalid_token_cannot_affect_output():
    torch.manual_seed(3)
    layout = _small_layout()
    model = NestingModel(layout=layout, embed_dim=16, n_heads=4, n_enc_layers=1).eval()
    clean = _tokens(layout, real=2)
    polluted = clean.clone()
    polluted[:, 2:, :layout.valid_index] = torch.randn_like(
        polluted[:, 2:, :layout.valid_index]) * 1000
    state = torch.randn(1, layout.state_dim)
    mask = torch.ones(1, layout.max_parts * 6, dtype=torch.bool)
    logits_a, value_a = model.forward_decision(clean, state, mask)
    logits_b, value_b = model.forward_decision(polluted, state, mask)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)
    assert torch.allclose(value_a, value_b, atol=1e-6)


def test_padding_isolation_through_actor_and_critic():
    torch.manual_seed(31)
    layout = _small_layout()
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(layout.obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Discrete(layout.max_parts * 6)
    policy = PointerActorCriticPolicy(
        observation_space, action_space, lambda _: 3e-4,
        layout=layout, embed_dim=16, n_heads=4, n_layers=1,
    ).eval()
    clean_tokens = _tokens(layout, real=2)
    polluted_tokens = clean_tokens.clone()
    polluted_tokens[:, 2:, :layout.valid_index] = 1e4
    state = torch.randn(1, layout.state_dim)
    clean_obs = torch.cat((clean_tokens.flatten(1), state), dim=1)
    polluted_obs = torch.cat((polluted_tokens.flatten(1), state), dim=1)
    clean_h, clean_context = policy.features_extractor._get_tokens_and_context(clean_obs)
    polluted_h, polluted_context = policy.features_extractor._get_tokens_and_context(polluted_obs)
    clean_logits = policy.pointer_head(clean_h, clean_context)
    polluted_logits = policy.pointer_head(polluted_h, polluted_context)
    assert torch.allclose(clean_logits, polluted_logits, atol=1e-6)
    assert torch.allclose(
        policy.value_net(clean_context), policy.value_net(polluted_context), atol=1e-6)


def test_all_padding_attention_is_finite_and_zero_context():
    layout = _small_layout()
    encoder = PartEncoder(layout=layout, embed_dim=16, n_heads=4, n_layers=1).eval()
    decoder = StepDecoder(layout=layout, embed_dim=16, n_heads=4).eval()
    tokens = torch.zeros(2, layout.max_parts, layout.part_dim)
    valid = encoder.get_valid_mask(tokens)
    encoded = encoder(tokens)
    context, value = decoder(torch.randn(2, layout.state_dim), encoded, valid)
    assert torch.isfinite(encoded).all() and torch.isfinite(context).all() and torch.isfinite(value).all()
    assert torch.equal(encoded, torch.zeros_like(encoded))
    assert torch.equal(context, torch.zeros_like(context))


def test_rollout_buffer_stores_pre_action_part_snapshot():
    env = _ThreePartEnv()
    layout = env.observation_layout
    model = _CountingModel(layout)
    ppo = NestingPPO(env, model, n_steps=2, batch_size=2, n_epochs=1)
    rollout = ppo.collect_rollout()
    stored = ppo.last_rollout_buffer["part_feats"][0]
    live = env.get_part_feats()
    assert stored[:, layout.packed_index].sum() == 0
    assert live[:, layout.packed_index].sum() == 2
    assert stored[:, layout.packed_index].sum() == 0
    model.decision_inputs.clear()
    ppo.update(*rollout[:-1])
    update_packed_counts = model.decision_inputs[0][..., layout.packed_index].sum(dim=1)
    assert sorted(update_packed_counts.tolist()) == [0.0, 1.0]


def test_no_dynamic_part_embedding_reused_across_env_step():
    env = NestingSchedulingEnv()
    env.reset(seed=8, options={"num_parts": 2})
    layout = env.observation_layout
    model = _CountingModel(layout).eval()

    class _PPO:
        device = "cpu"

        def __init__(self):
            self.model = model

        @staticmethod
        def _to_tensor(value):
            return torch.as_tensor(value, dtype=torch.float32)

    predictor = NestingModelPredictor(_PPO(), env)
    action, _ = predictor.predict(None, deterministic=True)
    env.step(action)
    predictor.predict(None, deterministic=True)
    assert len(model.decision_inputs) == 2
    assert not torch.equal(model.decision_inputs[0], model.decision_inputs[1])


def test_training_reencodes_parts_each_decision():
    env = _TwoPartEnv()
    model = _CountingModel(env.observation_layout)
    ppo = NestingPPO(env, model, n_steps=1, batch_size=1, n_epochs=1)
    ppo.collect_rollout()
    assert len(model.decision_inputs) == 2  # action plus fresh bootstrap state
    assert not torch.equal(model.decision_inputs[0], model.decision_inputs[1])


def test_provider_reencodes_parts_each_decision():
    test_no_dynamic_part_embedding_reused_across_env_step()


@pytest.mark.parametrize("runner", [run_batch_episode, run_generalization_episode])
def test_evaluation_reencodes_parts_each_decision(runner):
    model_env = NestingSchedulingEnv()
    model = _CountingModel(model_env.observation_layout).eval()
    if runner is run_batch_episode:
        runner(model_env, model, seed=9, options={"num_parts": 2})
    else:
        runner(model, None, "cpu", 9, 2, (200, 200), evaluation_mode="edd")
    assert len(model.decision_inputs) == 2
    assert not torch.equal(model.decision_inputs[0], model.decision_inputs[1])


def test_part_encoder_permutation_equivariance():
    torch.manual_seed(4)
    layout = _small_layout()
    encoder = PartEncoder(layout=layout, embed_dim=16, n_heads=4, n_layers=1).eval()
    tokens = _tokens(layout, real=4)
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(
        encoder(tokens[:, permutation]), encoder(tokens)[:, permutation], atol=1e-6)


def test_decoder_context_permutation_invariance():
    torch.manual_seed(5)
    layout = _small_layout()
    encoder = PartEncoder(layout=layout, embed_dim=16, n_heads=4, n_layers=1).eval()
    decoder = StepDecoder(layout=layout, embed_dim=16, n_heads=4).eval()
    tokens = _tokens(layout, real=4)
    permutation = torch.tensor([2, 0, 3, 1])
    state = torch.randn(1, layout.state_dim)
    valid = encoder.get_valid_mask(tokens)
    h = encoder(tokens)
    context, _ = decoder(state, h, valid)
    perm_context, _ = decoder(state, h[:, permutation], valid[:, permutation])
    assert torch.allclose(context, perm_context, atol=1e-6)


def test_action_blocks_follow_part_permutation():
    torch.manual_seed(6)
    actor = PointerActorHead(embed_dim=16, n_actions_per_part=6).eval()
    h = torch.randn(1, 4, 16)
    context = torch.randn(1, 16)
    permutation = torch.tensor([2, 0, 3, 1])
    logits = actor(h, context).reshape(1, 4, 6)
    permuted = actor(h[:, permutation], context).reshape(1, 4, 6)
    assert torch.allclose(permuted, logits[:, permutation], atol=1e-6)


def test_legacy_checkpoint_shape_is_rejected():
    layout = _small_layout()
    model = NestingModel(layout=layout, embed_dim=16, n_heads=4, n_enc_layers=1)
    state = copy.deepcopy(model.state_dict())
    key = "encoder.item_embed.0.weight"
    state[key] = state[key][:, :5]
    with pytest.raises(ValueError, match="5-D token / 657-D observation schema"):
        load_nesting_state_dict_strict(model, state)
    assert "retraining" in LEGACY_NESTING_CHECKPOINT_ERROR
