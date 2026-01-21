import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    一个“全量记录型” TensorBoard Callback：
    - 保留你已有的 Physical / Economy / Rewards
    - 额外透传 SB3 / MaskablePPO 的全部训练指标
    """

    def _on_step(self) -> bool:
        # =====================================================
        # A. 环境 episode 级指标（你原来就有的）
        # =====================================================
        for info in self.locals.get("infos", []):
            if "episode_metrics" not in info:
                continue

            m = info["episode_metrics"]

            # ---------- Physical ----------
            if 'utilization' in m:
                self.logger.record("Physical/Utilization", m['utilization'])
            elif 'raw_avg_utilization' in m:
                self.logger.record("Physical/Utilization", m['raw_avg_utilization'])

            self.logger.record("Physical/Plate_Count", m.get('plate_count', 0))
            self.logger.record("Physical/Late_Count", m.get('late_orders_count', 0))

            if 'total_delay' in m:
                self.logger.record("Physical/Total_Delay_Time", m['total_delay'])

            # ---------- Economy ----------
            if 'cost_total' in m:
                self.logger.record("Economy/Total_Cost", m['cost_total'])
            if 'cost_material' in m:
                self.logger.record("Economy/Material_Cost", m['cost_material'])
            if 'cost_jit' in m:
                self.logger.record("Economy/JIT_Cost", m['cost_jit'])

            # ---------- Rewards ----------
            if 'norm_reward' in m:
                self.logger.record("Rewards/Scaled_Reward", m['norm_reward'])
            elif 'norm_util_score' in m:
                self.logger.record("Rewards/Norm_Util", m['norm_util_score'])
                self.logger.record("Rewards/Norm_JIT", m['norm_jit_score'])

        # =====================================================
        # B. PPO / MaskablePPO 训练级指标（新增）
        # =====================================================
        if self.model is not None and hasattr(self.model, "logger"):
            # SB3 内部 logger 里已经有这些值
            # 我们只负责“转存”
            for key, value in self.model.logger.name_to_value.items():
                # 只转存 train/ rollout/ 等训练相关指标
                if key.startswith("train/") or key.startswith("rollout/"):
                    self.logger.record(key, value)

        return True


class SnapshotCallback(BaseCallback):
    """
    你原来的快照回调，原样保留
    """
    def __init__(self, freq, log_dir):
        super().__init__()
        self.freq = freq
        self.path = os.path.join(log_dir, "snapshots")
        os.makedirs(self.path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            try:
                env = self.training_env.envs[0].unwrapped
                if hasattr(env, 'history_plates') and env.history_plates:
                    self._save(env.history_plates, self.num_timesteps)
            except Exception as e:
                print(f"Snapshot Error: {e}")
        return True

    def _save(self, plates, step):
        plates = sorted(plates, key=lambda x: x.utilization, reverse=True)[:4]
        if not plates:
            return

        fig, ax = plt.subplots(1, len(plates), figsize=(12, 3))
        if len(plates) == 1:
            ax = [ax]

        for i, a in enumerate(ax):
            p = plates[i]
            a.add_patch(
                patches.Rectangle((0, 0), p.width, p.height, fc='#f0f0f0', ec='black')
            )
            for pt in p.placed_parts:
                x, y, w, h, oid = pt[:5]
                a.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        fc=plt.cm.tab20(int(oid) % 20),
                        ec='black'
                    )
                )
            a.set_title(f"Util: {p.utilization:.2%}")
            a.axis('off')

        plt.savefig(f"{self.path}/step_{step}.png")
        plt.close(fig)
