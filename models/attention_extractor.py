import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    升级版特征提取器：
    使用 Transformer Encoder 处理变长零件序列。
    具备 Self-Attention 机制，能捕捉零件间的几何互补关系。
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 item_dim: int = 42, global_prefix_dim: int = 0):
        super().__init__(observation_space, features_dim)

        self.item_dim = item_dim
        self.global_prefix_dim = global_prefix_dim

        total_input_dim = observation_space.shape[0]
        self.seq_input_dim = total_input_dim - global_prefix_dim

        self.max_items = self.seq_input_dim // self.item_dim
        self.embed_dim = 128  # Transformer 的隐藏层维度

        # 1. 零件嵌入层 (Input Embedding)
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # 2. 全局前缀编码器 (处理 Global Feats)
        if self.global_prefix_dim > 0:
            self.prefix_encoder = nn.Sequential(
                nn.Linear(self.global_prefix_dim, 64),
                nn.ReLU()
            )
            self.fusion_dim = self.embed_dim + 64
        else:
            self.prefix_encoder = None
            self.fusion_dim = self.embed_dim

        # 3. 🟢 Transformer Encoder (核心升级)
        # nhead=4, 2层。让零件之间进行"对话"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,  # [Batch, Seq, Feat]
            norm_first=True  # Pre-LN 训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. 最终输出层
        self.final_fc = nn.Linear(self.fusion_dim, features_dim)
        self.final_ln = nn.LayerNorm(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 数值清洗
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, 0.0)

        batch_size = observations.shape[0]

        # === A. 全局前缀处理 ===
        global_feat = None
        if self.global_prefix_dim > 0:
            prefix_input = observations[:, :self.global_prefix_dim]
            global_feat = self.prefix_encoder(prefix_input)
            seq_input = observations[:, self.global_prefix_dim:]
        else:
            seq_input = observations

        # === B. 序列处理 ===
        # [Batch, Max_Items, Item_Dim]
        x = seq_input.view(batch_size, -1, self.item_dim)

        # 生成 Padding Mask (True 表示该位置是 Padding，需要被忽略)
        # 注意：PyTorch Transformer 的 src_key_padding_mask 逻辑是 True 代表忽略
        # 我们检测全0特征
        is_padding = (torch.sum(torch.abs(x), dim=2) < 1e-6)  # [Batch, Max_Items]

        # Embedding
        embeddings = self.item_encoder(x)  # [Batch, Max_Items, 128]

        # 🟢 Transformer Self-Attention
        # 这里发生了魔法：零件之间互相"看"到了对方
        trans_out = self.transformer(embeddings, src_key_padding_mask=is_padding)

        # === C. 全局聚合 ===
        # 使用 Mask 进行 Mean Pooling
        # 反转 mask (0.0 表示 padding, 1.0 表示 valid)
        valid_mask = (~is_padding).float().unsqueeze(-1)  # [Batch, Max_Items, 1]

        masked_out = trans_out * valid_mask
        num_valid = torch.sum(valid_mask, dim=1)
        num_valid = torch.clamp(num_valid, min=1.0)

        # 聚合出"当前的排样局势"
        seq_context = torch.sum(masked_out, dim=1) / num_valid  # [Batch, 128]

        # === D. 融合输出 ===
        if global_feat is not None:
            final_input = torch.cat([global_feat, seq_context], dim=1)
        else:
            final_input = seq_context

        out = self.final_fc(final_input)
        # 最后的数值稳定
        return self.final_ln(out)