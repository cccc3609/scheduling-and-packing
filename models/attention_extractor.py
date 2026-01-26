import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    数值稳定增强版 Attention 特征提取器
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 item_dim: int = 10, global_prefix_dim: int = 0):
        super().__init__(observation_space, features_dim)

        self.item_dim = item_dim
        self.global_prefix_dim = global_prefix_dim

        total_input_dim = observation_space.shape[0]
        self.seq_input_dim = total_input_dim - global_prefix_dim

        if self.seq_input_dim % self.item_dim != 0:
            raise ValueError(f"Dim mismatch: {self.seq_input_dim} % {self.item_dim} != 0")

        self.max_items = self.seq_input_dim // self.item_dim

        # 1. 列表项编码器 (Item Encoder)
        # 使用 Tanh 替代 ReLU 以防止数值爆炸
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),  # 🟢 改为 Tanh
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()  # 🟢 改为 Tanh
        )

        # 2. 全局前缀编码器
        if self.global_prefix_dim > 0:
            self.prefix_encoder = nn.Sequential(
                nn.Linear(self.global_prefix_dim, 64),
                nn.LayerNorm(64),
                nn.Tanh()  # 🟢 改为 Tanh
            )
            concat_dim = 128 + 64
        else:
            self.prefix_encoder = None
            concat_dim = 128

        # 3. 最终映射层
        self.final_fc = nn.Linear(concat_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 🟢 防御 1: 入口清洗 (防止环境传来的 NaN)
        if torch.isnan(observations).any() or torch.isinf(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size = observations.shape[0]

        # === A. 处理全局前缀 ===
        global_feat = None
        if self.global_prefix_dim > 0:
            prefix_input = observations[:, :self.global_prefix_dim]
            global_feat = self.prefix_encoder(prefix_input)
            seq_input = observations[:, self.global_prefix_dim:]
        else:
            seq_input = observations

        # === B. 处理序列 ===
        x = seq_input.view(batch_size, -1, self.item_dim)

        # Mask 生成
        mask = (torch.sum(torch.abs(x), dim=2, keepdim=True) > 1e-6).float()

        embeddings = self.item_encoder(x)
        masked_embeddings = embeddings * mask

        # Global Pooling (Sum)
        num_valid = torch.sum(mask, dim=1)
        # 🟢 防御 2: 分母绝对不能太小
        num_valid = torch.clamp(num_valid, min=1.0)

        seq_context = torch.sum(masked_embeddings, dim=1) / num_valid

        # === C. 融合 ===
        if global_feat is not None:
            final_input = torch.cat([global_feat, seq_context], dim=1)
        else:
            final_input = seq_context

        output = self.final_fc(final_input)

        # 🟢 防御 3: 出口截断 (防止 Logits 爆炸)
        # 这一步至关重要，它保证了传给 PPO Policy 的值永远在 [-10, 10] 之间
        # 这样 Softmax 就绝对不会算出 NaN
        output = torch.clamp(output, -10.0, 10.0)

        return output