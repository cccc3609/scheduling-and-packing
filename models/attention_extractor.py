import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 item_dim: int = 10, global_prefix_dim: int = 0):
        super().__init__(observation_space, features_dim)

        self.item_dim = item_dim
        self.global_prefix_dim = global_prefix_dim

        total_input_dim = observation_space.shape[0]
        self.seq_input_dim = total_input_dim - global_prefix_dim

        # 1. 列表项编码器
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 2. 全局前缀编码器
        if self.global_prefix_dim > 0:
            self.prefix_encoder = nn.Sequential(
                nn.Linear(self.global_prefix_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            )
            concat_dim = 128 + 64
        else:
            self.prefix_encoder = None
            concat_dim = 128

        # 3. 最终映射层
        self.final_fc = nn.Linear(concat_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if torch.isnan(observations).any() or torch.isinf(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0, posinf=5.0, neginf=-5.0)

        batch_size = observations.shape[0]

        global_feat = None
        if self.global_prefix_dim > 0:
            prefix_input = observations[:, :self.global_prefix_dim]
            global_feat = self.prefix_encoder(prefix_input)
            seq_input = observations[:, self.global_prefix_dim:]
        else:
            seq_input = observations

        x = seq_input.view(batch_size, -1, self.item_dim)
        mask = (torch.sum(torch.abs(x), dim=2, keepdim=True) > 1e-6).float()

        embeddings = self.item_encoder(x)
        masked_embeddings = embeddings * mask

        num_valid = torch.sum(mask, dim=1)
        num_valid = torch.clamp(num_valid, min=1e-5)

        seq_context = torch.sum(masked_embeddings, dim=1) / num_valid

        if global_feat is not None:
            final_input = torch.cat([global_feat, seq_context], dim=1)
        else:
            final_input = seq_context

            # 🟢 新增：出口清洗
            # 防止 Global Pooling 因为除以极小值产生过大的特征
            final_output = self.final_fc(final_input)

            # 限制特征层的输出范围，防止传给 PPO 的 Logits 爆炸
            final_output = torch.clamp(final_output, -10.0, 10.0)

        return self.final_fc(final_input)