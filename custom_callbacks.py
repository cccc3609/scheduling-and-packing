import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    负责将环境中结算的业务指标提取出来，画到 TensorBoard 上的自定义回调
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 遍历所有的并行环境（即使你目前只有一个 DummyVecEnv，SB3 底层也是个列表）
        for i in range(len(self.locals.get("infos", []))):
            info = self.locals["infos"][i]

            # 只有当一个 Episode (一局) 结束时，环境才会抛出这些统计信息
            if "episode_metrics" in info:
                metrics = info["episode_metrics"]
                cost_m = self.training_env.get_attr("cost_metrics")[i]

                # ==== 1. 物理业务指标 (Physical) ====
                # 记录这局排完后，真实的板材利用率
                self.logger.record("Physical/Utilization", cost_m.get("utilization", 0.0))
                # 记录这局总共用了多少张板子
                self.logger.record("Physical/Plate_Count", cost_m.get("plate_count", 0))
                # 记录这局拖期订单的总延迟时间
                self.logger.record("Physical/Total_Delay_Time", cost_m.get("total_delay", 0.0))
                # 记录迟到的订单数量
                self.logger.record("Physical/Late_Orders_Count", metrics.get("late_orders_count", 0))

                # ==== 2. 经济成本指标 (Economy) ====
                # 记录这局的总花销（越低越好）
                self.logger.record("Economy/Total_Cost", cost_m.get("cost_total", 0.0))
                # 记录这局的材料费
                self.logger.record("Economy/Material_Cost", cost_m.get("cost_material", 0.0))
                # 记录这局的交期罚金 (JIT Cost，这个是我们消融实验的核心观测点！)
                self.logger.record("Economy/JIT_Cost", cost_m.get("cost_jit", 0.0))

                # ==== 3. 强化学习基础奖励 ====
                # 记录模型最后算出的归一化得分
                self.logger.record("Rewards/Scaled_Reward", cost_m.get("norm_reward", 0.0))

        return True


class SnapshotCallback(BaseCallback):
    """
    占位防报错：定时画图的回调
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(SnapshotCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        # 这里你可以接入你的 plot_nesting 逻辑，如果觉得慢，可以直接 pass
        return True