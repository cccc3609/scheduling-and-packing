import os
import shutil
import datetime
import torch.nn as nn
from typing import Callable
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from custom_callbacks import TensorboardCallback, SnapshotCallback
from config import TRAIN_CONFIG
from models.attention_extractor import AttentionFeatureExtractor


def mask_fn(env): return env.get_wrapper_attr("_get_action_mask")()


def exponential_schedule(start_lr: float, end_lr: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        current_progress = 1.0 - progress_remaining
        return start_lr * (end_lr / start_lr) ** current_progress

    return func


def setup_experiment():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    base_dir = f"./experiments/{exp_name}"
    log_n = f"{base_dir}/logs/nesting/"
    log_s = f"{base_dir}/logs/scheduling/"
    save_dir = f"{base_dir}/models/"
    code_dir = f"{base_dir}/code_backup/"
    for d in [log_n, log_s, save_dir, code_dir]: os.makedirs(d, exist_ok=True)

    # 简单备份
    if os.path.exists("train_dual.py"): shutil.copy("train_dual.py", code_dir)
    return log_n, log_s, save_dir


def main():
    log_n, log_s, save_dir = setup_experiment()

    nest_env = NestingSchedulingEnv()
    nest_env = ActionMasker(nest_env, mask_fn)
    sched_env = SchedulingEnv()
    sched_env = ActionMasker(sched_env, mask_fn)

    pk_nest = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, item_dim=22, global_prefix_dim=0),
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])
    )

    pk_sched = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, item_dim=4, global_prefix_dim=3),
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    # 🟢 调低初始学习率至 3e-4 (更稳定)
    lr_start = 3e-4
    lr_end = 1e-5

    print(f"Init Nesting PPO (LR={lr_start})...")
    nest_model = MaskablePPO(
        "MlpPolicy", nest_env, policy_kwargs=pk_nest,
        learning_rate=exponential_schedule(lr_start, lr_end),
        n_steps=4096, batch_size=512, gamma=0.99, ent_coef=0.05,
        tensorboard_log=log_n, verbose=1,
        max_grad_norm=0.5  # 梯度裁剪
    )

    print(f"Init Scheduling PPO (LR={lr_start})...")
    sched_model = MaskablePPO(
        "MlpPolicy", sched_env, policy_kwargs=pk_sched,
        learning_rate=exponential_schedule(lr_start, lr_end),
        n_steps=4096, batch_size=512, gamma=0.99, ent_coef=0.05,
        tensorboard_log=log_s, verbose=1,
        max_grad_norm=0.5,  # 梯度裁剪
        clip_range=0.1  # 🟢 更保守的 PPO Clip，防止参数突变
    )

    nest_env.unwrapped.set_scheduling_partner(sched_model)
    sched_env.unwrapped.set_nesting_partner(nest_env, nest_model)

    cb = CallbackList([
        CheckpointCallback(50000, save_dir, name_prefix="nest"),
        TensorboardCallback(),
        SnapshotCallback(20000, log_n)
    ])

    cycles = TRAIN_CONFIG['total_cycles']
    steps = TRAIN_CONFIG['steps_per_cycle']
    print(f"🚀 Start Training... (Cycles: {cycles}, Steps: {steps})")

    for c in range(cycles):
        print(f"\n===== Cycle {c + 1}/{cycles} =====")
        print(f">>> Nesting training")
        nest_env.unwrapped.set_scheduling_partner(sched_model)
        nest_model.learn(steps, reset_num_timesteps=False, callback=cb)
        nest_model.save(f"{save_dir}/nesting_c{c + 1}")

        print(">>> Scheduling training")
        sched_env.unwrapped.set_nesting_partner(nest_env, nest_model)
        sched_model.learn(steps, reset_num_timesteps=False)
        sched_model.save(f"{save_dir}/scheduling_c{c + 1}")


if __name__ == "__main__":
    main()