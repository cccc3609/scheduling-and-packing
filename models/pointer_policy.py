import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from models.pointer_extractor import PointerFeatureExtractor, PointerActorHead

# apply_masking 里 True=合法、False=非法（sb3_contrib 的约定）
_NEG_INF = -1e9


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
        item_dim: int = 67,
        global_prefix_dim: int = 0,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_parts: int = 120,
        n_actions_per_part: int = 6,
        **kwargs
    ):
        self.item_dim = item_dim
        self.global_prefix_dim = global_prefix_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_parts = n_parts
        self.n_actions_per_part = n_actions_per_part

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
        )

        self.pointer_head = PointerActorHead(
            embed_dim=self.embed_dim,
            n_parts=self.n_parts,
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

        # 处理 batch 维度
        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).expand_as(logits)
        elif mask_tensor.dim() == 2 and mask_tensor.shape[0] != logits.shape[0]:
            mask_tensor = mask_tensor.expand_as(logits)

        # 安全检查：如果某行全是 False，放开所有动作（避免 softmax nan）
        all_masked = ~mask_tensor.any(dim=-1, keepdim=True)  # [B, 1]
        mask_tensor = mask_tensor | all_masked.expand_as(mask_tensor)

        # False 位置填充极小值
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