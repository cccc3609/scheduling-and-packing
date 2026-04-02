"""
train_dual.py  —  自定义 PPO 训练循环，完全绕开 SB3

架构：
  Nesting  : NestingModel（PartEncoder + StepDecoder + PointerActorHead）
             Encoder-Decoder 解耦：每局只跑一次 Encoder，每步只跑轻量 Decoder
  Scheduling: MaskablePPO（sb3_contrib）+ AttentionFeatureExtractor
              调度侧动作空间较小，继续使用 SB3 简化工程

训练流程：
  Phase 1: Nesting 预热（自定义 PPO，无调度 partner）
  Phase 2: Scheduling 适应（SB3 MaskablePPO，nesting partner = Phase1 模型）
  Phase 3: 联合微调（交替更新）
"""

import os, shutil, datetime, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

# SB3 只用于 Scheduling 侧
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from models.pointer_extractor import NestingModel
from models.attention_extractor import AttentionFeatureExtractor
from config import TRAIN_CONFIG, MAX_PARTS_CAPACITY

try:
    from custom_callbacks import TensorboardCallback, SnapshotCallback
    HAS_CALLBACKS = True
except ImportError:
    HAS_CALLBACKS = False

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
PART_FEAT_DIM  = NestingSchedulingEnv.PART_FEAT_DIM   # 5
STATE_FEAT_DIM = NestingSchedulingEnv.STATE_FEAT_DIM  # 57
MAX_PARTS      = MAX_PARTS_CAPACITY                    # 120
N_ACTIONS_PER  = 6                                     # 2旋转 × 3策略

# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()

def exponential_lr(start_lr, end_lr=1e-5):
    def func(progress_remaining):
        p = 1.0 - progress_remaining
        return start_lr * (end_lr / start_lr) ** p
    return func

def setup_experiment():
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base    = f"./experiments/exp_{ts}"
    log_n   = f"{base}/logs/nesting/"
    log_s   = f"{base}/logs/scheduling/"
    save_dir = f"{base}/models/"
    for d in [log_n, log_s, save_dir]:
        os.makedirs(d, exist_ok=True)
    print(f"Experiment initialized: {base}")
    return log_n, log_s, save_dir


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 PPO — Nesting 侧
# ─────────────────────────────────────────────────────────────────────────────

class NestingPPO:
    """
    专为 NestingModel 设计的 PPO 训练器。
    核心优化：Encoder 每局只跑一次，Decoder 每步只做轻量 Cross-Attention。
    """

    def __init__(
        self,
        env: NestingSchedulingEnv,
        model: NestingModel,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 4,
        device: str = "cpu",
    ):
        self.env         = env
        self.model       = model.to(device)
        self.device      = device
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_eps    = clip_eps
        self.ent_coef    = ent_coef
        self.vf_coef     = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps     = n_steps
        self.batch_size  = batch_size
        self.n_epochs    = n_epochs
        self.optimizer   = optim.Adam(model.parameters(), lr=lr)
        self.total_steps = 0

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def collect_rollout(self):
        """
        收集 n_steps 个转移样本。
        关键：每个 episode 开始时只跑一次 PartEncoder，后续每步只跑 StepDecoder。
        """
        buf_part_feats  = []   # [T, N, 5]  — 零件静态特征（episode 级）
        buf_state_feats = []   # [T, 57]    — 当前动态状态
        buf_actions     = []   # [T]
        buf_log_probs   = []   # [T]
        buf_values      = []   # [T]
        buf_rewards     = []   # [T]
        buf_dones       = []   # [T]
        buf_masks       = []   # [T, 720]

        obs, info = self.env.reset()
        part_feats_np = self.env.get_part_feats()   # [120, 5]，episode 开始只取一次
        H = None  # 延迟到第一步计算（确保在 no_grad 外）

        ep_rewards = []
        ep_r = 0.0

        for _ in range(self.n_steps):
            state_feat_np = self.env.get_state_feat()   # [57]，每步更新
            action_mask   = self.env.unwrapped._get_action_mask()  # [720] bool

            pf_t  = self._to_tensor(part_feats_np).unsqueeze(0)    # [1, 120, 5]
            sf_t  = self._to_tensor(state_feat_np).unsqueeze(0)    # [1, 57]
            msk_t = self._to_tensor(action_mask).bool().unsqueeze(0)  # [1, 720]

            with torch.no_grad():
                # Encoder 只在 episode 开始时跑（H is None 或 episode 刚 reset）
                if H is None:
                    H = self.model.encode_parts(pf_t)              # [1, 120, 128]

                # 只跑轻量 Decoder
                logits, value = self.model.decode_step(sf_t, H, msk_t)
                dist   = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            buf_part_feats.append(part_feats_np.copy())
            buf_state_feats.append(state_feat_np.copy())
            buf_actions.append(int(action.item()))
            buf_log_probs.append(float(log_prob.item()))
            buf_values.append(float(value.item()))
            buf_masks.append(action_mask.copy())

            obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            done = terminated or truncated
            buf_rewards.append(float(reward))
            buf_dones.append(float(done))
            ep_r += reward
            self.total_steps += 1

            if done:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                obs, info = self.env.reset()
                part_feats_np = self.env.get_part_feats()  # 新 episode，更新零件特征
                H = None                                    # 清空 H，下步重新 encode

        # 计算最后一步的 bootstrap value
        with torch.no_grad():
            sf_last = self._to_tensor(self.env.get_state_feat()).unsqueeze(0)
            pf_last = self._to_tensor(part_feats_np).unsqueeze(0)
            if H is None:
                H = self.model.encode_parts(pf_last)
            _, last_value = self.model.decode_step(sf_last, H)
        last_val = float(last_value.item()) * (1.0 - buf_dones[-1])

        # GAE
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_val if t == self.n_steps - 1 else buf_values[t + 1]
            delta = buf_rewards[t] + self.gamma * next_val * (1 - buf_dones[t]) - buf_values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - buf_dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(buf_values, dtype=np.float32)

        mean_ep_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        return (buf_part_feats, buf_state_feats, buf_actions, buf_log_probs,
                buf_values, buf_masks, advantages, returns, mean_ep_r)

    def update(self, buf_part_feats, buf_state_feats, buf_actions, buf_log_probs,
               buf_values, buf_masks, advantages, returns):
        T = len(buf_actions)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns,    dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        all_losses = []
        idx = np.arange(T)

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.batch_size):
                b = idx[start: start + self.batch_size]

                pf  = torch.stack([torch.as_tensor(buf_part_feats[i],  dtype=torch.float32)
                                   for i in b]).to(self.device)   # [B, 120, 5]
                sf  = torch.stack([torch.as_tensor(buf_state_feats[i], dtype=torch.float32)
                                   for i in b]).to(self.device)   # [B, 57]
                msk = torch.stack([torch.as_tensor(buf_masks[i], dtype=torch.bool)
                                   for i in b]).to(self.device)   # [B, 720]
                old_lp = torch.as_tensor([buf_log_probs[i] for i in b],
                                         dtype=torch.float32, device=self.device)
                act    = torch.as_tensor([buf_actions[i] for i in b],
                                         dtype=torch.long, device=self.device)

                # Encoder 在训练时对 batch 完整跑（梯度需要流过 Encoder）
                H      = self.model.encode_parts(pf)              # [B, 120, 128]
                logits, value = self.model.decode_step(sf, H, msk)

                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(act)
                entropy = dist.entropy().mean()

                ratio   = torch.exp(new_lp - old_lp)
                adv_b   = adv_t[b]
                pg_loss = -torch.min(
                    ratio * adv_b,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                ).mean()

                vf_loss = F.mse_loss(value.squeeze(-1), ret_t[b])
                loss    = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                all_losses.append(float(loss.item()))

        return float(np.mean(all_losses))

    def learn(self, total_timesteps: int, save_path: str = None, tag: str = "nesting"):
        steps_done = 0
        cycle = 0
        while steps_done < total_timesteps:
            cycle += 1
            (pf, sf, acts, lps, vals, masks, adv, ret, mean_r) = self.collect_rollout()
            loss = self.update(pf, sf, acts, lps, vals, masks, adv, ret)
            steps_done += self.n_steps
            print(f"[{tag}] cycle={cycle:4d}  steps={steps_done:7d}  "
                  f"mean_ep_r={mean_r:7.2f}  loss={loss:.4f}")
            if save_path and cycle % 10 == 0:
                torch.save(self.model.state_dict(), f"{save_path}/{tag}_c{cycle}.pt")
        if save_path:
            torch.save(self.model.state_dict(), f"{save_path}/{tag}_final.pt")

    def predict(self, env: NestingSchedulingEnv, deterministic: bool = True):
        """
        用于 SchedulingEnv.reset() 中驱动 nesting rollout。
        返回 action（int）。
        """
        part_feats = env.get_part_feats()
        state_feat = env.get_state_feat()
        mask       = env.unwrapped._get_action_mask()

        pf  = self._to_tensor(part_feats).unsqueeze(0)
        sf  = self._to_tensor(state_feat).unsqueeze(0)
        msk = self._to_tensor(mask).bool().unsqueeze(0)

        with torch.no_grad():
            H = self.model.encode_parts(pf)
            logits, _ = self.model.decode_step(sf, H, msk)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Scheduling 侧 — 仍用 SB3，但修正 item_dim / global_prefix_dim
# ─────────────────────────────────────────────────────────────────────────────
# scheduling obs 结构：
#   m_feat(3) + upstream(3) + nesting_result(8) = 14  ← global_prefix
#   tasks: max_tasks × 4                              ← 序列部分，item_dim=4

def make_sched_policy_kwargs():
    from config import MAX_SCHED_TASKS_CAPACITY
    return dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            item_dim=4,            # 修复：每个 task 特征维度是 4
            global_prefix_dim=14,  # 修复：machines(3)+upstream(3)+nesting_result(8)
        ),
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )


# ─────────────────────────────────────────────────────────────────────────────
# NestingPPO 包装成 SB3-like predict 接口供 SchedulingEnv 使用
# ─────────────────────────────────────────────────────────────────────────────

class NestingModelPredictor:
    """
    让 SchedulingEnv 能像调用 SB3 model 一样调用 NestingPPO。
    SchedulingEnv.reset() 里会调用:
        a, _ = self.nesting_model.predict(obs, action_masks=m, deterministic=True)
    这里把这个接口适配到 NestingPPO.predict()。
    """

    def __init__(self, ppo: NestingPPO, env: NestingSchedulingEnv):
        self.ppo = ppo
        self.env = env

    def predict(self, obs, action_masks=None, deterministic=True):
        action = self.ppo.predict(self.env, deterministic=deterministic)
        return action, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log_n, log_s, save_dir = setup_experiment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    lr_start = TRAIN_CONFIG.get('lr_start', 3e-4)
    lr_end   = TRAIN_CONFIG.get('lr_end', 1e-5)
    steps    = TRAIN_CONFIG['steps_per_cycle']

    # ── 环境 ──────────────────────────────────────────────────────────────────
    nest_env  = NestingSchedulingEnv()
    sched_env = SchedulingEnv()
    sched_env_masked = ActionMasker(sched_env, mask_fn)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    nesting_model = NestingModel(
        part_feat_dim=PART_FEAT_DIM,
        state_feat_dim=STATE_FEAT_DIM,
        embed_dim=128,
        n_heads=4,
        n_enc_layers=2,
        n_actions_per_part=N_ACTIONS_PER,
        max_parts=MAX_PARTS,
    )

    nesting_ppo = NestingPPO(
        env=nest_env,
        model=nesting_model,
        lr=lr_start,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        device=device,
    )

    pk_sched = make_sched_policy_kwargs()
    print("Init Scheduling PPO (SB3)...")
    sched_model = MaskablePPO(
        "MlpPolicy", sched_env_masked, policy_kwargs=pk_sched,
        learning_rate=exponential_lr(lr_start, lr_end),
        n_steps=8192, batch_size=2048, gamma=0.99,
        ent_coef=0.05, tensorboard_log=log_s, verbose=1,
        max_grad_norm=0.1, clip_range=0.1
    )

    # ── Phase 1：Nesting 预热 ─────────────────────────────────────────────────
    print(f"\n{'='*50}\nPhase 1: Nesting Warm-up\n{'='*50}")
    nest_env.set_scheduling_partner(None)
    nesting_ppo.learn(steps * 10, save_path=save_dir, tag="nesting_phase1")

    # ── Phase 2：Scheduling 适应 ──────────────────────────────────────────────
    print(f"\n{'='*50}\nPhase 2: Scheduling Adaptation\n{'='*50}")
    # 给 SchedulingEnv 装载 nesting predictor
    predictor = NestingModelPredictor(nesting_ppo, nest_env)
    sched_env.set_nesting_partner(ActionMasker(nest_env, mask_fn), predictor)
    sched_model.learn(steps * 10, reset_num_timesteps=False)
    sched_model.save(f"{save_dir}/scheduling_phase2")

    # ── Phase 3：联合微调 ─────────────────────────────────────────────────────
    print(f"\n{'='*50}\nPhase 3: Joint Fine-tuning\n{'='*50}")
    fine_tune_cycles = max(1, TRAIN_CONFIG['total_cycles'] - 20)

    for c in range(fine_tune_cycles):
        print(f"\n===== Joint Cycle {c + 1} =====")

        # Nesting：接收调度 partner（SB3 model）
        nest_env.set_scheduling_partner(sched_model)
        nesting_ppo.learn(steps, save_path=save_dir, tag=f"nesting_joint_c{c+1}")

        # Scheduling：接收更新后的 nesting predictor
        predictor = NestingModelPredictor(nesting_ppo, nest_env)
        sched_env.set_nesting_partner(ActionMasker(nest_env, mask_fn), predictor)
        sched_model.learn(steps, reset_num_timesteps=False)
        sched_model.save(f"{save_dir}/scheduling_joint_c{c+1}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()