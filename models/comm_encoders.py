import torch
import torch.nn as nn
import numpy as np
import copy


class SchedulingIntentEncoder(nn.Module):
    """
    Scheduling agent 在 episode 开始时，看到所有订单信息，
    输出 16 维"调度意图向量" m_s，传给 Nesting agent。

    输入特征（per-order 统计，压缩到固定维度）：
        - 订单交期的分布特征（mean/std/min/max）
        - 机器数量与订单数量的比值（容量压力）
        - 各订单面积占比（大小件分布）
        - 预计总加工时间 vs 最紧订单交期（时间压力比）
    """

    def __init__(self, input_dim: int = 12, comm_dim: int = 16):
        super().__init__()
        self.comm_dim = comm_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),
            nn.Tanh(),  # 输出限制在 [-1, 1]，数值稳定
        )

    def encode_orders(self, orders: dict, parts_pool: list,
                      num_machines: int, cutting_speed: float) -> np.ndarray:
        """
        从订单信息构造输入特征，返回 numpy 向量。
        在 SchedulingEnv.reset() 中调用。
        """
        if not orders:
            return np.zeros(self.comm_dim, dtype=np.float32)

        dues = [o['due_date'] for o in orders.values()]
        total_area = sum(p['area'] for p in parts_pool)
        total_proc = sum(2 * (p['w'] + p['h']) for p in parts_pool) / max(0.1, cutting_speed)
        num_orders = len(orders)

        # 12 维输入特征
        features = [
            np.mean(dues) / max(1.0, total_proc),  # 平均交期 / 总加工时间
            np.std(dues) / max(1.0, total_proc),  # 交期离散度
            np.min(dues) / max(1.0, total_proc),  # 最紧交期压力
            np.max(dues) / max(1.0, total_proc),  # 最宽松交期
            num_machines / max(1.0, num_orders),  # 机器/订单比（容量松紧）
            total_proc / max(1.0, num_machines * np.max(dues)),  # 负载率
            len([d for d in dues if d < total_proc]) / num_orders,  # 已经很紧的订单比例
            total_area / max(1.0, num_orders),  # 平均订单面积（相对值）
            np.std([len([p for p in parts_pool if p['order_id'] == oid])
                    for oid in orders]) / max(1.0, num_orders),  # 订单大小不均匀度
            num_orders / 10.0,  # 订单数量（归一化）
            total_proc / max(1.0, np.min(dues)),  # 时间压力比
            np.mean([o.get('finished_time', 0.0) for o in orders.values()]),  # 当前完成度
        ]
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_s = self.encoder(feat_tensor).squeeze(0).numpy()
        return m_s.astype(np.float32)


class NestingResultEncoder(nn.Module):
    """
    Nesting agent 排样完成后，输出 8 维"排样结果摘要" m_n，
    传给 Scheduling agent 作为参考。

    输入特征：
        - 实际用板数 / 预期用板数
        - 平均利用率 & 利用率方差
        - 每张板的交期分散度（急缓混排程度）
        - 订单碎片化指数（一个订单分布在几张板上）
    """

    def __init__(self, input_dim: int = 8, comm_dim: int = 8):
        super().__init__()
        self.comm_dim = comm_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, comm_dim),
            nn.Tanh(),
        )

    def encode_result(self, history_plates: list, orders: dict,
                      parts_pool: list, plate_area: float) -> np.ndarray:
        """
        从排样结果构造摘要特征，返回 numpy 向量。
        在排样 episode 结束时（nesting terminated）调用。
        """
        if not history_plates:
            return np.zeros(self.comm_dim, dtype=np.float32)

        utils = [p.utilization for p in history_plates]
        num_plates = len(history_plates)

        # 每张板的交期标准差均值（急缓混排程度）
        due_stds = []
        frag_ratios = []
        for plate in history_plates:
            if not plate.placed_parts:
                continue
            oids = [int(p[4]) for p in plate.placed_parts]
            dues = [orders[o]['due_date'] for o in oids if o in orders]
            if len(dues) > 1:
                due_stds.append(np.std(dues))
            # 碎片化：订单零件分布在多张板上
            for oid in set(oids):
                total = len([p for p in parts_pool if p['order_id'] == oid])
                on_this = oids.count(oid)
                frag_ratios.append(1.0 - on_this / max(1, total))

        total_part_area = sum(p['area'] for p in parts_pool)
        expected_plates = max(1.0, total_part_area / max(1.0, plate_area))

        features = [
            num_plates / max(1.0, expected_plates),  # 实际/预期用板比
            np.mean(utils),  # 平均利用率
            np.std(utils) if len(utils) > 1 else 0.0,  # 利用率方差
            np.mean(due_stds) if due_stds else 0.0,  # 平均交期混排度
            np.std(due_stds) if len(due_stds) > 1 else 0.0,  # 混排度方差
            np.mean(frag_ratios) if frag_ratios else 0.0,  # 平均碎片化
            min(utils) if utils else 0.0,  # 最低利用率板（问题板）
            num_plates / 10.0,  # 板数（归一化）
        ]
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_n = self.encoder(feat_tensor).squeeze(0).numpy()
        return m_n.astype(np.float32)