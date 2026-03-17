import os
import shutil
import datetime
import math
import torch.nn as nn
from typing import Callable

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from custom_callbacks import TensorboardCallback, SnapshotCallback
from config import TRAIN_CONFIG
from models.attention_extractor import AttentionFeatureExtractor


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


# 学习率调度器
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

    # 创建目录
    for d in [log_n, log_s, save_dir, code_dir]:
        os.makedirs(d, exist_ok=True)

    # 备份根目录脚本
    files_to_backup = [
        "train_dual.py",
        "custom_callbacks.py",
        "config.py",
        "evaluate_generalization.py",
        "packing_envs.py",
        "scheduling_envs.py",
        "evaluate_batch.py",
        "test_result.py"
    ]

    for f in files_to_backup:
        if os.path.exists(f):
            shutil.copy(f, code_dir)
            print(f"Backed up file: {f}")

    # 备份
    folders_to_backup = ["envs", "heuristic", "models"]
    for folder in folders_to_backup:
        src = folder
        dst = os.path.join(code_dir, folder)
        if os.path.exists(src):
            # 如果目标目录存在，先删除（
            if os.path.exists(dst):
                shutil.rmtree(dst)

            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            print(f"Backed up folder: {src}")

    print(f" Experiment initialized: {base_dir}")
    return log_n, log_s, save_dir


def main():
    log_n, log_s, save_dir = setup_experiment()

    nest_env = NestingSchedulingEnv()
    nest_env = ActionMasker(nest_env, mask_fn)
    sched_env = SchedulingEnv()
    sched_env = ActionMasker(sched_env, mask_fn)

    pk_nest = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256,item_dim=42, global_prefix_dim=0),
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])
    )

    pk_sched = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, item_dim=4, global_prefix_dim=3),
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )


    lr_start = TRAIN_CONFIG.get('lr_start', 3e-4)
    lr_end = TRAIN_CONFIG.get('lr_end', 1e-5)

    print("Init Nesting PPO (Tanh Activated)...")
    nest_model = MaskablePPO("MlpPolicy", nest_env, policy_kwargs=pk_nest,
                             learning_rate=exponential_schedule(lr_start, lr_end),
                             n_steps=4096, batch_size=512, gamma=0.99, ent_coef=0.01,
                             tensorboard_log=log_n, verbose=1, max_grad_norm=0.5)

    print("Init Scheduling PPO (Tanh Activated)...")
    sched_model = MaskablePPO("MlpPolicy", sched_env, policy_kwargs=pk_sched,
                              learning_rate=exponential_schedule(lr_start, lr_end),
                              n_steps=4096, batch_size=512, gamma=0.99, ent_coef=0.05,
                              tensorboard_log=log_s, verbose=1,
                              max_grad_norm=0.1, clip_range=0.1)

    nest_env.unwrapped.set_scheduling_partner(sched_model)
    sched_env.unwrapped.set_nesting_partner(nest_env, nest_model)

    cb = CallbackList(
        [CheckpointCallback(100000, save_dir, 'nest'), TensorboardCallback(), SnapshotCallback(50000, log_n)])

    cycles = TRAIN_CONFIG['total_cycles']
    steps = TRAIN_CONFIG['steps_per_cycle']

    print(f"Start Training... (Cycles: {cycles}, Steps: {steps})")
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