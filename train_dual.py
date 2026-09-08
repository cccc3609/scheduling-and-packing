"""
train_dual.py  —  协同优化训练循环

架构：
  Nesting  : NestingModel（自定义 PPO）
  Scheduling: MaskablePPO（sb3_contrib）+ AttentionFeatureExtractor

Patch 2：环境边界和终局调度 rollout 由 integration wrappers 协调。

训练流程：
  Phase 1: Nesting 预热（EDD fallback，通信向量全零）
  Phase 2: Scheduling 适应（nesting 冻结，scheduling 学习调度策略）
  Phase 3: 联合微调（交替更新，联合终局奖励，通信梯度打通）
"""

import os, shutil, datetime, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_problem_provider import SchedulingProblemProviderWrapper
from integration.scheduling_terminal_reward import SchedulingTerminalRewardWrapper
from models.pointer_extractor import NestingModel
from models.attention_extractor import AttentionFeatureExtractor
from config import TRAIN_CONFIG
from core.scheduling_observation import SchedulingObservationLayout

try:
    from custom_callbacks import TensorboardCallback, SnapshotCallback
    HAS_CALLBACKS = True
except ImportError:
    HAS_CALLBACKS = False

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
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

    终局调度 reward 由外部 wrapper 在 transition 返回前注入。
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
        self.total_steps = 0
        self.last_rollout_buffer = None

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def set_lr(self, new_lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def collect_rollout(self):
        buf_part_feats  = []
        buf_state_feats = []
        buf_actions     = []
        buf_log_probs   = []
        buf_values      = []
        buf_rewards     = []
        buf_dones       = []
        buf_masks       = []

        obs, info = self.env.reset()
        ep_rewards = []
        ep_r = 0.0

        for _ in range(self.n_steps):
            part_feats_np = self.env.get_part_feats()
            state_feat_np = self.env.get_state_feat()
            action_mask   = self.env.unwrapped._get_action_mask()

            pf_t  = self._to_tensor(part_feats_np).unsqueeze(0)
            sf_t  = self._to_tensor(state_feat_np).unsqueeze(0)
            msk_t = self._to_tensor(action_mask).bool().unsqueeze(0)

            with torch.no_grad():
                logits, value = self.model.forward_decision(pf_t, sf_t, msk_t)
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

        # bootstrap
        with torch.no_grad():
            sf_last = self._to_tensor(self.env.get_state_feat()).unsqueeze(0)
            pf_last = self._to_tensor(self.env.get_part_feats()).unsqueeze(0)
            _, last_value = self.model.forward_decision(pf_last, sf_last)
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

        self.last_rollout_buffer = {
            "part_feats": [item.copy() for item in buf_part_feats],
            "state_feats": [item.copy() for item in buf_state_feats],
            "action_masks": [item.copy() for item in buf_masks],
            "actions": list(buf_actions),
            "old_log_probs": list(buf_log_probs),
            "values": list(buf_values),
            "rewards": list(buf_rewards),
            "dones": list(buf_dones),
        }

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
                                   for i in b]).to(self.device)
                sf  = torch.stack([torch.as_tensor(buf_state_feats[i], dtype=torch.float32)
                                   for i in b]).to(self.device)
                msk = torch.stack([torch.as_tensor(buf_masks[i], dtype=torch.bool)
                                   for i in b]).to(self.device)
                old_lp = torch.as_tensor([buf_log_probs[i] for i in b],
                                         dtype=torch.float32, device=self.device)
                act    = torch.as_tensor([buf_actions[i] for i in b],
                                         dtype=torch.long, device=self.device)

                logits, value = self.model.forward_decision(pf, sf, msk)

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
        part_feats = env.get_part_feats()
        state_feat = env.get_state_feat()
        mask       = env.unwrapped._get_action_mask()

        pf  = self._to_tensor(part_feats).unsqueeze(0)
        sf  = self._to_tensor(state_feat).unsqueeze(0)
        msk = self._to_tensor(mask).bool().unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model.forward_decision(pf, sf, msk)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Scheduling 侧
# ─────────────────────────────────────────────────────────────────────────────

def make_sched_policy_kwargs(layout: SchedulingObservationLayout):
    return dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            layout=layout,
        ),
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )


# ─────────────────────────────────────────────────────────────────────────────
# NestingModelPredictor — current-decision encoding
# ─────────────────────────────────────────────────────────────────────────────

class NestingModelPredictor:
    """
    SB3-like predict interface. Dynamic part tokens are encoded every call.
    """

    def __init__(self, ppo: NestingPPO, env: NestingSchedulingEnv):
        self.ppo = ppo
        self.env = env
        if self.ppo.model.layout != self.env.unwrapped.layout:
            raise ValueError("Nesting predictor model and environment layouts must match")

    def reset_cache(self):
        """Compatibility no-op: Patch 5 forbids an episode-level H cache."""

    def predict(self, obs, action_masks=None, deterministic=True):
        state_feat = self.env.get_state_feat()
        mask       = self.env.unwrapped._get_action_mask()
        part_feats = self.env.get_part_feats()

        pf  = self.ppo._to_tensor(part_feats).unsqueeze(0)
        sf  = self.ppo._to_tensor(state_feat).unsqueeze(0)
        msk = self.ppo._to_tensor(mask).bool().unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.ppo.model.forward_decision(pf, sf, msk)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())
        return action, None


# ─────────────────────────────────────────────────────────────────────────────
# Main — 三阶段训练，Phase 2+ 启用联合终局奖励
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log_n, log_s, save_dir = setup_experiment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    lr_start = TRAIN_CONFIG.get('lr_start', 3e-4)
    lr_end   = TRAIN_CONFIG.get('lr_end', 1e-5)
    steps    = TRAIN_CONFIG['steps_per_cycle']

    # ── 环境 ──
    nest_base = NestingSchedulingEnv()
    nesting_layout = nest_base.layout

    # ── 模型 ──
    nesting_model = NestingModel(
        part_feat_dim=nesting_layout.part_dim,
        state_feat_dim=nesting_layout.state_dim,
        embed_dim=128,
        n_heads=4,
        n_enc_layers=2,
        n_actions_per_part=N_ACTIONS_PER,
        max_parts=nesting_layout.max_parts,
        layout=nesting_layout,
    )

    nesting_ppo = NestingPPO(
        env=nest_base,
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

    terminal_nest_env = SchedulingTerminalRewardWrapper(
        nest_base, evaluation_mode="edd")
    nesting_ppo.env = terminal_nest_env
    provider_nest_env = NestingSchedulingEnv(
        observation_layout=nesting_layout)
    provider_predictor = NestingModelPredictor(nesting_ppo, provider_nest_env)
    sched_env = SchedulingEnv()
    sched_provider = SchedulingProblemProviderWrapper(
        sched_env, provider_nest_env, provider_predictor)
    sched_env_masked = ActionMasker(sched_provider, mask_fn)

    pk_sched = make_sched_policy_kwargs(sched_env.observation_layout)
    print("Init Scheduling PPO (SB3)...")
    sched_model = MaskablePPO(
        "MlpPolicy", sched_env_masked, policy_kwargs=pk_sched,
        learning_rate=exponential_lr(lr_start, lr_end),
        n_steps=8192, batch_size=2048, gamma=0.99,
        ent_coef=0.05, tensorboard_log=log_s, verbose=1,
        max_grad_norm=0.1, clip_range=0.1
    )
    # ── Phase 1：Nesting 预热 ─────────────────────────────────────────────
    # 显式使用 EDD terminal evaluator；这不是 policy rollout 的异常 fallback。
    print(f"\n{'='*60}\nPhase 1: Nesting Warm-up (explicit EDD terminal evaluation)\n{'='*60}")
    nesting_ppo.learn(steps * 10, save_path=save_dir, tag="nesting_phase1")

    # ── Phase 2：Scheduling 适应 ──────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Scheduling Adaptation\n{'='*60}")
    sched_model.learn(steps * 10, reset_num_timesteps=False)
    sched_model.save(f"{save_dir}/scheduling_phase2")

    # Phase 3 使用当前 scheduling policy；保留同一个 wrapper 和 EMA transform。
    terminal_nest_env.set_evaluator("policy", scheduling_policy=sched_model)

    # ── Phase 3：联合微调（核心改进）─────────────────────────────────────
    # 启用联合终局奖励：Nesting 终局时调用真实 Scheduling agent
    print(f"\n{'='*60}\nPhase 3: Joint Fine-tuning (real scheduling for terminal reward)\n{'='*60}")
    fine_tune_cycles = max(1, TRAIN_CONFIG['total_cycles'] - 20)

    for c in range(fine_tune_cycles):
        print(f"\n===== Joint Cycle {c + 1}/{fine_tune_cycles} =====")

        # 余弦退火
        progress = c / max(1, fine_tune_cycles - 1)
        cos_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * progress))
        nesting_ppo.set_lr(cos_lr)
        print(f"  Nesting LR: {cos_lr:.6f}")

        # ── Nesting 训练：启用联合终局奖励 ──
        nesting_ppo.learn(steps, save_path=save_dir, tag=f"nesting_joint_c{c+1}")

        # ── Scheduling 训练 ──
        sched_model.learn(steps, reset_num_timesteps=False)
        sched_model.save(f"{save_dir}/scheduling_joint_c{c+1}")

    # 保存最终模型
    torch.save(nesting_model.state_dict(), f"{save_dir}/nesting_final.pt")
    sched_model.save(f"{save_dir}/scheduling_final")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
