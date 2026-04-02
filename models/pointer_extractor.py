"""
models/pointer_extractor.py  —  Encoder-Decoder 解耦的 Pointer Network

改进：
  - PartEncoder  : episode 开始时只跑一次，生成零件静态 embedding H [N, D]
  - StepDecoder  : 每步用当前状态生成 query，对 H 做 Cross-Attention
  - 计算量 O(T·N²) → O(N²) + O(T·N)
  - 无 padding，消除 norm_first 的 PyTorch warning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PartEncoder(nn.Module):
    """
    零件静态 Encoder。episode reset 时调用一次，输出缓存。
    输入: [B, N, part_feat_dim]
    输出: [B, N, embed_dim]
    """
    def __init__(self, part_feat_dim=5, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
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

    def forward(self, part_feats: torch.Tensor) -> torch.Tensor:
        x = self.item_embed(part_feats)
        return self.transformer(x)


class StepDecoder(nn.Module):
    """
    每步 Decoder。接收当前动态状态，输出 context 和 value 估计。
    输入: state_feat [B, state_feat_dim], H [B, N, embed_dim]
    输出: context [B, embed_dim], value [B, 1]
    """
    def __init__(self, state_feat_dim=57, embed_dim=128, n_heads=4):
        super().__init__()
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

    def forward(self, state_feat: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.state_proj(state_feat).unsqueeze(1)   # [B, 1, D]
        context, _ = self.cross_attn(query, H, H)          # [B, 1, D]
        context = context.squeeze(1)                        # [B, D]
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
            # 安全兜底：全 False 时放开所有
            all_masked = ~mask.any(dim=-1, keepdim=True)
            mask = mask | all_masked.expand_as(mask)
            logits = logits.masked_fill(~mask, -1e9)
        return logits


class NestingModel(nn.Module):
    """
    完整排样 Actor-Critic，供自定义 PPO 训练循环使用。

    使用流程：
        # episode 开始
        H = model.encode_parts(part_feats)          # [B, N, D]，只跑一次

        # 每步
        logits, value = model.decode_step(state_feat, H, mask)
        dist  = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    """
    def __init__(self, part_feat_dim=5, state_feat_dim=57,
                 embed_dim=128, n_heads=4, n_enc_layers=2,
                 n_actions_per_part=6, max_parts=120):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_actions_per_part = n_actions_per_part
        self.max_parts = max_parts

        self.encoder = PartEncoder(part_feat_dim, embed_dim, n_heads, n_enc_layers)
        self.decoder = StepDecoder(state_feat_dim, embed_dim, n_heads)
        self.actor   = PointerActorHead(embed_dim, n_actions_per_part)

    def encode_parts(self, part_feats: torch.Tensor) -> torch.Tensor:
        """part_feats: [B, N, 5] → H: [B, N, D]"""
        return self.encoder(part_feats)

    def decode_step(self, state_feat, H, action_mask=None):
        """→ logits [B, N*K], value [B, 1]"""
        context, value = self.decoder(state_feat, H)
        logits = self.actor(H, context, action_mask)
        return logits, value

    @torch.no_grad()
    def get_attention_weights(self, state_feat, H):
        context, _ = self.decoder(state_feat, H)
        B, N, D = H.shape
        t1 = self.actor.W1(H)
        t2 = self.actor.W2(context).unsqueeze(1).expand_as(t1)
        scores = self.actor.clip_C * torch.tanh(self.actor.v(torch.tanh(t1 + t2)).squeeze(-1))
        return torch.softmax(scores, dim=-1).cpu().numpy()