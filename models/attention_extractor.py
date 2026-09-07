import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from core.scheduling_observation import SchedulingObservationLayout


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """Encode fixed global context and a masked scheduling-task sequence."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        layout: SchedulingObservationLayout = None,
    ):
        super().__init__(observation_space, features_dim)
        if not isinstance(layout, SchedulingObservationLayout):
            raise TypeError("AttentionFeatureExtractor requires SchedulingObservationLayout")
        if observation_space.shape != (layout.obs_dim,):
            raise ValueError(
                "Scheduling observation space does not match layout: "
                f"expected {(layout.obs_dim,)}, got {observation_space.shape}"
            )

        self.layout = layout
        self.item_dim = layout.task_dim
        self.global_prefix_dim = layout.global_prefix_dim
        self.max_items = layout.max_tasks
        self.embed_dim = 128  # Transformer 的隐藏层维度

        # 1. 零件嵌入层 (Input Embedding)
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.task_position_embedding = nn.Embedding(
            self.max_items, self.embed_dim)

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

    def _split_observation(self, observations: torch.Tensor):
        if observations.shape[-1] != self.layout.obs_dim:
            raise ValueError(
                f"Expected observation width {self.layout.obs_dim}, "
                f"got {observations.shape[-1]}"
            )
        prefix_input = observations[:, :self.global_prefix_dim]
        task_input = observations[:, self.layout.task_slice]
        task_tokens = task_input.reshape(
            observations.shape[0], self.max_items, self.item_dim)
        return prefix_input, task_tokens

    def _valid_task_mask(self, task_tokens: torch.Tensor) -> torch.Tensor:
        return task_tokens[..., self.layout.task_valid_index] > 0.5

    def _encode_task_sequence(self, task_tokens: torch.Tensor) -> torch.Tensor:
        valid_task_mask = self._valid_task_mask(task_tokens)
        original_padding_mask = ~valid_task_mask

        embeddings = self.item_encoder(task_tokens)
        positions = torch.arange(
            self.max_items, device=task_tokens.device)
        position_embeddings = self.task_position_embedding(positions).unsqueeze(0)
        embeddings = embeddings + position_embeddings * valid_task_mask.unsqueeze(-1)

        safe_padding_mask = original_padding_mask.clone()
        all_padding_rows = safe_padding_mask.all(dim=1)
        if all_padding_rows.any():
            safe_padding_mask[all_padding_rows, 0] = False

        trans_out = self.transformer(
            embeddings, src_key_padding_mask=safe_padding_mask)

        valid_weights = valid_task_mask.to(trans_out.dtype).unsqueeze(-1)
        masked_out = trans_out * valid_weights
        num_valid = valid_weights.sum(dim=1).clamp(min=1.0)
        return masked_out.sum(dim=1) / num_valid

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 数值清洗
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, 0.0)

        prefix_input, task_tokens = self._split_observation(observations)
        global_feat = self.prefix_encoder(prefix_input)
        seq_context = self._encode_task_sequence(task_tokens)

        # === D. 融合输出 ===
        final_input = torch.cat([global_feat, seq_context], dim=1)

        out = self.final_fc(final_input)
        # 最后的数值稳定
        return self.final_ln(out)
