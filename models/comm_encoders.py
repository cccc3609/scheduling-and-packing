"""
models/comm_encoders.py  —  可训练通信编码器

协同优化改进：
  SchedulingIntentEncoder / NestingResultEncoder 增加 forward() 方法，
     参数可纳入对应 agent 的优化器，通信向量可被梯度更新
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Communication encoders
# ─────────────────────────────────────────────────────────────────────────────

class SchedulingIntentEncoder(nn.Module):
    """
    Scheduling agent 在 episode 开始时，看到所有订单信息，
    输出 16 维"调度意图向量" m_s，传给 Nesting agent。

    改进：
    - 新增 forward() 方法，支持有梯度的前向传播
    - 参数可纳入 Scheduling agent 的优化器
    - encode_orders() 保留为无梯度的推理接口
    """

    def __init__(self, input_dim: int = 12, comm_dim: int = 16):
        super().__init__()
        self.comm_dim = comm_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),
            nn.Tanh(),
        )

    def _extract_features(self, orders: dict, parts_pool: list,
                          num_machines: int, cutting_speed: float) -> list:
        """提取 12 维手工特征"""
        if not orders:
            return [0.0] * self.input_dim

        dues = [o['due_date'] for o in orders.values()]
        total_area = sum(p['area'] for p in parts_pool)
        total_proc = sum(2 * (p['w'] + p['h']) for p in parts_pool) / max(0.1, cutting_speed)
        num_orders = len(orders)

        features = [
            np.mean(dues) / max(1.0, total_proc),
            np.std(dues) / max(1.0, total_proc),
            np.min(dues) / max(1.0, total_proc),
            np.max(dues) / max(1.0, total_proc),
            num_machines / max(1.0, num_orders),
            total_proc / max(1.0, num_machines * np.max(dues)),
            len([d for d in dues if d < total_proc]) / num_orders,
            total_area / max(1.0, num_orders),
            np.std([len([p for p in parts_pool if p['order_id'] == oid])
                    for oid in orders]) / max(1.0, num_orders),
            num_orders / 10.0,
            total_proc / max(1.0, np.min(dues)),
            np.mean([o.get('finished_time', 0.0) for o in orders.values()]),
        ]
        return features

    def forward(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        """有梯度的前向传播（训练时）"""
        return self.encoder(feat_tensor)

    def encode_orders(self, orders: dict, parts_pool: list,
                      num_machines: int, cutting_speed: float) -> np.ndarray:
        """无梯度的推理接口"""
        features = self._extract_features(orders, parts_pool, num_machines, cutting_speed)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_s = self.encoder(feat_tensor).squeeze(0).numpy()
        return m_s.astype(np.float32)


class NestingResultEncoder(nn.Module):
    """
    Nesting agent 排样完成后，输出 8 维"排样结果摘要" m_n。

    改进：
    - 新增 forward() 支持梯度传播
    - 参数可纳入 Nesting agent 的优化器
    """

    def __init__(self, input_dim: int = 8, comm_dim: int = 8):
        super().__init__()
        self.comm_dim = comm_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, comm_dim),
            nn.Tanh(),
        )

    def _extract_features(self, history_plates: list, orders: dict,
                          parts_pool: list, plate_area: float) -> list:
        """提取 8 维手工特征"""
        if not history_plates:
            return [0.0] * self.input_dim

        utils = [p.utilization for p in history_plates]
        num_plates = len(history_plates)

        due_stds = []
        frag_ratios = []
        for plate in history_plates:
            if not plate.placed_parts:
                continue
            oids = [int(p[4]) for p in plate.placed_parts]
            dues = [orders[o]['due_date'] for o in oids if o in orders]
            if len(dues) > 1:
                due_stds.append(np.std(dues))
            for oid in set(oids):
                total = len([p for p in parts_pool if p['order_id'] == oid])
                on_this = oids.count(oid)
                frag_ratios.append(1.0 - on_this / max(1, total))

        total_part_area = sum(p['area'] for p in parts_pool)
        expected_plates = max(1.0, total_part_area / max(1.0, plate_area))

        features = [
            num_plates / max(1.0, expected_plates),
            np.mean(utils),
            np.std(utils) if len(utils) > 1 else 0.0,
            np.mean(due_stds) if due_stds else 0.0,
            np.std(due_stds) if len(due_stds) > 1 else 0.0,
            np.mean(frag_ratios) if frag_ratios else 0.0,
            min(utils) if utils else 0.0,
            num_plates / 10.0,
        ]
        return features

    def forward(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        """有梯度的前向传播"""
        return self.encoder(feat_tensor)

    def encode_result(self, history_plates: list, orders: dict,
                      parts_pool: list, plate_area: float) -> np.ndarray:
        """无梯度的推理接口"""
        features = self._extract_features(history_plates, orders, parts_pool, plate_area)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_n = self.encoder(feat_tensor).squeeze(0).numpy()
        return m_n.astype(np.float32)
