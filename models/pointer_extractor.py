"""
models/pointer_extractor.py  —  Encoder-Decoder 解耦的 Pointer Network

Patch 5 correctness:
  - validity comes only from the canonical VALID token feature;
  - self-attention and cross-attention both mask padding;
  - dynamic part embeddings are recomputed for every decision.
"""

import torch
import torch.nn as nn
from typing import Tuple

from core.nesting_observation import (
    LEGACY_NESTING_SCHEMA_ERROR, NestingObservationLayout,
)


LEGACY_NESTING_CHECKPOINT_ERROR = LEGACY_NESTING_SCHEMA_ERROR


def load_nesting_state_dict_strict(model: nn.Module, state_dict) -> None:
    """Load a Patch-5 nesting checkpoint without partial compatibility tricks."""
    first_weight = state_dict.get("encoder.item_embed.0.weight")
    if first_weight is not None and first_weight.shape[1] != model.layout.part_dim:
        raise ValueError(LEGACY_NESTING_CHECKPOINT_ERROR)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Nesting checkpoint is incompatible with the current strict Patch 5 model schema"
        ) from exc


class PartEncoder(nn.Module):
    """
    Part encoder with explicit validity and all-padding protection.
    输入: [B, N, part_feat_dim]
    输出: [B, N, embed_dim]
    """
    def __init__(self, part_feat_dim=None, embed_dim=128, n_heads=4, n_layers=2,
                 layout=None):
        super().__init__()
        self.layout = layout or NestingObservationLayout()
        part_feat_dim = self.layout.part_dim if part_feat_dim is None else part_feat_dim
        if part_feat_dim != self.layout.part_dim:
            raise ValueError(
                f"part_feat_dim={part_feat_dim} does not match layout.part_dim="
                f"{self.layout.part_dim}")
        self.part_feat_dim = part_feat_dim
        self.embed_dim = embed_dim
        self.item_embed = nn.Sequential(
            nn.Linear(part_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def get_valid_mask(self, part_feats: torch.Tensor) -> torch.Tensor:
        if part_feats.dim() != 3:
            raise ValueError("part_feats must have shape [batch, parts, features]")
        if part_feats.shape[-1] != self.layout.part_dim:
            raise ValueError(
                f"Expected part feature width {self.layout.part_dim}, "
                f"got {part_feats.shape[-1]}")
        if part_feats.shape[1] != self.layout.max_parts:
            raise ValueError(
                f"Expected {self.layout.max_parts} part slots, got {part_feats.shape[1]}")
        return part_feats[..., self.layout.valid_index] > 0.5

    @staticmethod
    def _safe_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~valid_mask
        all_padding = ~valid_mask.any(dim=-1)
        if all_padding.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padding, 0] = False
        return padding_mask

    def forward(self, part_feats: torch.Tensor) -> torch.Tensor:
        valid_mask = self.get_valid_mask(part_feats)
        safe_padding_mask = self._safe_padding_mask(valid_mask)
        x = self.item_embed(part_feats)
        encoded = self.transformer(x, src_key_padding_mask=safe_padding_mask)
        return encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)


class StepDecoder(nn.Module):
    """
    每步 Decoder。接收当前动态状态，输出 context 和 value 估计。
    输入: state_feat [B, state_feat_dim], H [B, N, embed_dim]
    输出: context [B, embed_dim], value [B, 1]
    """
    def __init__(self, state_feat_dim=None, embed_dim=128, n_heads=4, layout=None):
        super().__init__()
        self.layout = layout or NestingObservationLayout()
        state_feat_dim = self.layout.state_dim if state_feat_dim is None else state_feat_dim
        if state_feat_dim != self.layout.state_dim:
            raise ValueError(
                f"state_feat_dim={state_feat_dim} does not match layout.state_dim="
                f"{self.layout.state_dim}")
        self.state_proj = nn.Sequential(
            nn.Linear(state_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=0.0, batch_first=True
        )
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, state_feat: torch.Tensor, H: torch.Tensor,
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_feat.dim() != 2 or state_feat.shape[-1] != self.layout.state_dim:
            raise ValueError(
                f"state_feat must have shape [batch, {self.layout.state_dim}]")
        if H.dim() != 3 or H.shape[1] != self.layout.max_parts:
            raise ValueError(
                f"H must contain exactly {self.layout.max_parts} part slots")
        if valid_mask.shape != H.shape[:2]:
            raise ValueError("valid_mask must match the batch and part axes of H")
        valid_mask = valid_mask.to(dtype=torch.bool, device=H.device)
        safe_padding_mask = PartEncoder._safe_padding_mask(valid_mask)
        has_valid_part = valid_mask.any(dim=-1, keepdim=True)
        query = self.state_proj(state_feat).unsqueeze(1)   # [B, 1, D]
        context, _ = self.cross_attn(
            query, H, H, key_padding_mask=safe_padding_mask)
        context = context.squeeze(1)                        # [B, D]
        context = context * has_valid_part.to(context.dtype)
        return context, self.value_head(context)            # [B, D], [B, 1]


class PointerActorHead(nn.Module):
    """
    score_i = v^T · tanh(W1·H_i + W2·context)
    展开到 [B, N × n_actions_per_part]
    """
    def __init__(self, embed_dim=128, n_actions_per_part=6, clip_C=10.0):
        super().__init__()
        self.n_actions_per_part = n_actions_per_part
        self.clip_C = clip_C
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v  = nn.Linear(embed_dim, 1, bias=False)
        self.action_bias = nn.Parameter(torch.zeros(n_actions_per_part))

    def forward(self, H, context, action_mask=None):
        B, N, D = H.shape
        t1 = self.W1(H)
        t2 = self.W2(context).unsqueeze(1).expand_as(t1)
        scores = self.clip_C * torch.tanh(self.v(torch.tanh(t1 + t2)).squeeze(-1))  # [B, N]

        logits = (scores.unsqueeze(-1).expand(B, N, self.n_actions_per_part)
                  + self.action_bias.view(1, 1, -1)).reshape(B, N * self.n_actions_per_part)

        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool, device=logits.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand_as(logits)
            if mask.shape != logits.shape:
                raise ValueError(
                    f"action_mask must have shape {tuple(logits.shape)}, "
                    f"got {tuple(mask.shape)}")
            # Existing terminal-mask behavior; Patch 5 does not change action semantics.
            all_masked = ~mask.any(dim=-1, keepdim=True)
            mask = mask | all_masked.expand_as(mask)
            logits = logits.masked_fill(~mask, -1e9)
        return logits


class NestingModel(nn.Module):
    """
    完整排样 Actor-Critic，供自定义 PPO 训练循环使用。

    使用流程：
        # episode 开始
        logits, value = model.forward_decision(part_feats, state_feat, mask)
        dist  = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    """
    def __init__(self, part_feat_dim=None, state_feat_dim=None,
                 embed_dim=128, n_heads=4, n_enc_layers=2,
                 n_actions_per_part=6, max_parts=None, layout=None):
        super().__init__()
        self.layout = layout or NestingObservationLayout()
        part_feat_dim = self.layout.part_dim if part_feat_dim is None else part_feat_dim
        state_feat_dim = self.layout.state_dim if state_feat_dim is None else state_feat_dim
        max_parts = self.layout.max_parts if max_parts is None else max_parts
        if (part_feat_dim, state_feat_dim, max_parts) != (
                self.layout.part_dim, self.layout.state_dim, self.layout.max_parts):
            raise ValueError("NestingModel dimensions must match its observation layout")
        self.embed_dim = embed_dim
        self.n_actions_per_part = n_actions_per_part
        self.max_parts = max_parts

        self.encoder = PartEncoder(
            part_feat_dim, embed_dim, n_heads, n_enc_layers, layout=self.layout)
        self.decoder = StepDecoder(
            state_feat_dim, embed_dim, n_heads, layout=self.layout)
        self.actor   = PointerActorHead(embed_dim, n_actions_per_part)

    def encode_parts(self, part_feats: torch.Tensor) -> torch.Tensor:
        """Encode the current 6-D part-token snapshot."""
        return self.encoder(part_feats)

    def decode_step(self, state_feat, H, action_mask=None, valid_mask=None):
        """→ logits [B, N*K], value [B, 1]"""
        if valid_mask is None:
            raise ValueError("decode_step requires the original part valid_mask")
        context, value = self.decoder(state_feat, H, valid_mask)
        logits = self.actor(H, context, action_mask)
        return logits, value

    def forward_decision(self, part_feats, state_feat, action_mask=None):
        """Encode and decode one decision from the same current snapshot."""
        valid_mask = self.encoder.get_valid_mask(part_feats)
        H = self.encoder(part_feats)
        return self.decode_step(
            state_feat, H, action_mask=action_mask, valid_mask=valid_mask)

    @torch.no_grad()
    def get_attention_weights(self, state_feat, H, valid_mask):
        context, _ = self.decoder(state_feat, H, valid_mask)
        B, N, D = H.shape
        t1 = self.actor.W1(H)
        t2 = self.actor.W2(context).unsqueeze(1).expand_as(t1)
        scores = self.actor.clip_C * torch.tanh(self.actor.v(torch.tanh(t1 + t2)).squeeze(-1))
        scores = scores.masked_fill(~valid_mask, -1e9)
        all_padding = ~valid_mask.any(dim=-1, keepdim=True)
        scores = torch.where(all_padding, torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid_mask.to(weights.dtype)
        return weights.cpu().numpy()
