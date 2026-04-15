"""
手动加载 Scheduling Policy 权重，绕开 MaskablePPO.load() 在 Windows 上的崩溃问题。
"""
import zipfile, io, torch, torch.nn as nn
import numpy as np
from models.attention_extractor import AttentionFeatureExtractor
from stable_baselines3.common.distributions import CategoricalDistribution


class SchedulingPolicyInference:
    """
    轻量推理封装，只保留 predict() 接口，
    行为与 MaskablePPO.predict() 完全一致。
    """

    def __init__(self, zip_path: str, obs_dim: int, action_dim: int,
                 device: str = "cpu"):
        self.device     = device
        self.action_dim = action_dim

        # ── 重建 policy 网络结构 ──
        # 与 train_dual.py 的 make_sched_policy_kwargs() 完全对应
        self.features_extractor = AttentionFeatureExtractor(
            observation_space=_fake_space(obs_dim),
            features_dim=256,
            item_dim=4,
            global_prefix_dim=14,
        ).to(device)

        # mlp_extractor: pi=[256,256], vf=[256,256]
        self.mlp_pi = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        ).to(device)
        self.mlp_vf = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        ).to(device)
        self.action_net = nn.Linear(256, action_dim).to(device)
        self.value_net  = nn.Linear(256, 1).to(device)

        # ── 加载权重 ──
        self._load_weights(zip_path)

        self.features_extractor.eval()
        self.mlp_pi.eval()
        self.mlp_vf.eval()
        self.action_net.eval()
        self.value_net.eval()

    def _load_weights(self, zip_path: str):
        with zipfile.ZipFile(zip_path + ".zip", 'r') as zf:
            with zf.open("policy.pth") as f:
                sd = torch.load(io.BytesIO(f.read()), map_location=self.device)

        # SB3 保存的 key 前缀映射
        fe_sd  = {k.replace("features_extractor.", ""): v
                  for k, v in sd.items() if k.startswith("features_extractor.")}
        pi_sd  = {k.replace("mlp_extractor.policy_net.", ""): v
                  for k, v in sd.items() if k.startswith("mlp_extractor.policy_net.")}
        vf_sd  = {k.replace("mlp_extractor.value_net.", ""): v
                  for k, v in sd.items() if k.startswith("mlp_extractor.value_net.")}
        act_sd = {k.replace("action_net.", ""): v
                  for k, v in sd.items() if k.startswith("action_net.")}
        val_sd = {k.replace("value_net.", ""): v
                  for k, v in sd.items() if k.startswith("value_net.")}

        self.features_extractor.load_state_dict(fe_sd, strict=True)
        if pi_sd:  self.mlp_pi.load_state_dict(pi_sd,   strict=False)
        if vf_sd:  self.mlp_vf.load_state_dict(vf_sd,   strict=False)
        if act_sd: self.action_net.load_state_dict(act_sd, strict=True)
        if val_sd: self.value_net.load_state_dict(val_sd,  strict=True)
        print(f"  [SchedulingPolicy] 权重加载完成，共 {len(sd)} 个 key")

    @torch.no_grad()
    def predict(self, obs: np.ndarray,
                action_masks: np.ndarray = None,
                deterministic: bool = True):
        """
        与 MaskablePPO.predict() 签名兼容。
        返回 (action, None)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        feat   = self.features_extractor(obs_t)      # [1, 256]
        logits = self.action_net(self.mlp_pi(feat))  # [1, action_dim]

        if action_masks is not None:
            mask = torch.as_tensor(action_masks, dtype=torch.bool,
                                   device=self.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            all_masked = ~mask.any(dim=-1, keepdim=True)
            mask = mask | all_masked.expand_as(mask)
            logits = logits.masked_fill(~mask, -1e9)

        if deterministic:
            action = int(logits.argmax(dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(
                logits=logits).sample().item())
        return action, None


def _fake_space(dim: int):
    """构造一个假的 observation_space 供 AttentionFeatureExtractor 初始化用。"""
    import gymnasium as gym
    import numpy as np
    return gym.spaces.Box(low=-np.inf, high=np.inf,
                          shape=(dim,), dtype=np.float32)


def load_scheduling_policy(zip_prefix: str,
                            obs_dim: int = None,
                            action_dim: int = None,
                            device: str = "cpu") -> SchedulingPolicyInference:
    """
    对外接口。
    obs_dim    : scheduling obs 维度 = 3 + 120*4 + 3 + 8 = 494
    action_dim : 120 * 3 = 360
    """
    from config import MAX_SCHED_TASKS_CAPACITY
    if obs_dim    is None: obs_dim    = 3 + MAX_SCHED_TASKS_CAPACITY * 4 + 3 + 8
    if action_dim is None: action_dim = MAX_SCHED_TASKS_CAPACITY * 3
    return SchedulingPolicyInference(zip_prefix, obs_dim, action_dim, device)