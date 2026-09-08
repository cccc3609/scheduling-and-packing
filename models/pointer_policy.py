"""
models/pointer_policy.py  —  基于 NestingModel 的 SB3 兼容策略

修复：
  - 原代码导入不存在的 PointerFeatureExtractor → 改为使用 PartEncoder + StepDecoder
  - 保留 SB3 MaskableActorCriticPolicy 接口兼容性
  - 保留 get_attention_weights 可视化接口
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from models.pointer_extractor import PartEncoder, StepDecoder, PointerActorHead
from core.nesting_observation import (
    LEGACY_NESTING_SCHEMA_ERROR, NestingObservationLayout,
)

_NEG_INF = -1e9


class PointerFeatureExtractor(nn.Module):
    """
    将展平 obs 解析为 (tokens, context) 的桥接层。
    obs 结构: [part_feats_flat(N*part_feat_dim), state_feat(state_feat_dim)]
    """

    def __init__(self, observation_space, features_dim=128,
                 item_dim=None, global_prefix_dim=0,
                 embed_dim=128, n_heads=4, n_layers=2,
                 max_parts=None, state_feat_dim=None, layout=None):
        super().__init__()
        self.layout = layout or NestingObservationLayout()
        item_dim = self.layout.part_dim if item_dim is None else item_dim
        max_parts = self.layout.max_parts if max_parts is None else max_parts
        state_feat_dim = self.layout.state_dim if state_feat_dim is None else state_feat_dim
        if (item_dim, max_parts, state_feat_dim) != (
                self.layout.part_dim, self.layout.max_parts, self.layout.state_dim):
            raise ValueError("PointerFeatureExtractor dimensions must match its layout")
        if observation_space.shape != (self.layout.obs_dim,):
            self.layout.validate_observation_width(observation_space.shape[-1])
        self.item_dim = self.layout.part_dim
        self.max_parts = self.layout.max_parts
        self.state_feat_dim = self.layout.state_dim
        self.embed_dim = embed_dim

        self.part_encoder = PartEncoder(
            part_feat_dim=item_dim, embed_dim=embed_dim,
            n_heads=n_heads, n_layers=n_layers, layout=self.layout
        )
        self.step_decoder = StepDecoder(
            state_feat_dim=state_feat_dim, embed_dim=embed_dim,
            n_heads=n_heads, layout=self.layout
        )
        # features_dim 用于 SB3 兼容
        self._features_dim = features_dim

    @property
    def features_dim(self):
        return self._features_dim

    def _get_tokens_and_context(self, obs: torch.Tensor):
        """
        解析展平 obs → tokens [B, N, D], context [B, D]
        """
        if obs.dim() != 2:
            raise ValueError("Nesting observations must have shape [batch, obs_dim]")
        self.layout.validate_observation_width(obs.shape[-1])
        B = obs.shape[0]
        part_feats = obs[:, self.layout.part_slice].reshape(
            B, self.layout.max_parts, self.layout.part_dim)
        state_feat = obs[:, self.layout.state_slice]

        H = self.part_encoder(part_feats)                 # [B, N, D]
        valid_mask = self.part_encoder.get_valid_mask(part_feats)
        context, _ = self.step_decoder(state_feat, H, valid_mask)  # [B, D]
        return H, context

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, context = self._get_tokens_and_context(obs)
        return context


class PointerActorCriticPolicy(MaskableActorCriticPolicy):
    """
    替换 MaskablePPO 默认的 MlpPolicy。
    继承 MaskableActorCriticPolicy，满足 MaskablePPO 的类型检查。
    手动在 logits 层面应用掩码，避免 apply_masking 的 Simplex 校验问题。
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        item_dim: int = None,
        global_prefix_dim: int = 0,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_parts: int = None,
        n_actions_per_part: int = 6,
        state_feat_dim: int = None,
        layout: NestingObservationLayout = None,
        **kwargs
    ):
        self.layout = layout or NestingObservationLayout()
        item_dim = self.layout.part_dim if item_dim is None else item_dim
        n_parts = self.layout.max_parts if n_parts is None else n_parts
        state_feat_dim = self.layout.state_dim if state_feat_dim is None else state_feat_dim
        if (item_dim, n_parts, state_feat_dim) != (
                self.layout.part_dim, self.layout.max_parts, self.layout.state_dim):
            if item_dim == 5:
                raise ValueError(LEGACY_NESTING_SCHEMA_ERROR)
            raise ValueError("Pointer policy dimensions must match its observation layout")
        if observation_space.shape != (self.layout.obs_dim,):
            self.layout.validate_observation_width(observation_space.shape[-1])
        expected_actions = self.layout.max_parts * n_actions_per_part
        if action_space.n != expected_actions:
            raise ValueError(
                f"Expected {expected_actions} nesting actions, got {action_space.n}")
        self.item_dim = item_dim
        self.global_prefix_dim = global_prefix_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_parts = n_parts
        self.n_actions_per_part = n_actions_per_part
        self.state_feat_dim = state_feat_dim

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self.features_extractor = PointerFeatureExtractor(
            observation_space=self.observation_space,
            features_dim=self.embed_dim,
            item_dim=self.item_dim,
            global_prefix_dim=self.global_prefix_dim,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_parts=self.n_parts,
            state_feat_dim=self.state_feat_dim,
            layout=self.layout,
        )

        self.pointer_head = PointerActorHead(
            embed_dim=self.embed_dim,
            n_actions_per_part=self.n_actions_per_part,
        )

        self.value_net = nn.Linear(self.embed_dim, 1)

        self.action_dist = MaskableCategoricalDistribution(self.action_space.n)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    # ── 掩码应用辅助 ─────────────────────────────────────────────────────────

    def _apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        action_masks: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        在 logits 层面直接屏蔽非法动作。
        sb3_contrib 约定：mask 中 True=合法，False=非法。
        保证至少有一个合法动作，防止全屏蔽导致 nan。
        """
        if action_masks is None:
            return logits

        if isinstance(action_masks, np.ndarray):
            mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=logits.device)
        else:
            mask_tensor = action_masks.to(dtype=torch.bool, device=logits.device)

        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).expand_as(logits)
        elif mask_tensor.dim() == 2 and mask_tensor.shape[0] != logits.shape[0]:
            mask_tensor = mask_tensor.expand_as(logits)

        # 安全兜底：全 False 时放开所有
        all_masked = ~mask_tensor.any(dim=-1, keepdim=True)
        mask_tensor = mask_tensor | all_masked.expand_as(mask_tensor)

        logits = logits.masked_fill(~mask_tensor, _NEG_INF)
        return logits

    # ── 核心前向方法 ─────────────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, context = self.features_extractor._get_tokens_and_context(obs)
        action_logits = self.pointer_head(tokens, context)
        action_logits = self._apply_mask_to_logits(action_logits, action_masks)

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(context)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        tokens, context = self.features_extractor._get_tokens_and_context(obs)
        action_logits = self.pointer_head(tokens, context)
        action_logits = self._apply_mask_to_logits(action_logits, action_masks)

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(context)
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        _, context = self.features_extractor._get_tokens_and_context(obs)
        return self.value_net(context)

    def get_distribution(
        self,
        obs: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ):
        tokens, context = self.features_extractor._get_tokens_and_context(obs)
        action_logits = self.pointer_head(tokens, context)
        action_logits = self._apply_mask_to_logits(action_logits, action_masks)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    # ── 可视化接口 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_attention_weights(self, obs: torch.Tensor) -> np.ndarray:
        """返回各零件的 pointer attention 权重 [N] numpy array。"""
        tokens, context = self.features_extractor._get_tokens_and_context(obs)
        B, N, D = tokens.shape
        t1 = self.pointer_head.W1(tokens)
        t2 = self.pointer_head.W2(context).unsqueeze(1).expand_as(t1)
        scores = self.pointer_head.v(torch.tanh(t1 + t2)).squeeze(-1)
        scores = self.pointer_head.clip_C * torch.tanh(scores)
        weights = torch.softmax(scores, dim=-1)
        return weights.cpu().numpy()
