import os
import shutil
import datetime
import math
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
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ValidationCallback(BaseCallback):
    """
    带动作掩码的自定义边训边测回调机制
    用于在独立的验证环境上测试表现，并自动保存 Best Model
    """

    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, save_path="./", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # 每隔 eval_freq 步，暂停训练，进行验证
        if self.n_calls % self.eval_freq == 0:
            mean_reward, mean_util, mean_jit = self._evaluate()

            # 将验证结果写入 TensorBoard 的独立面板
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_utilization", mean_util)
            self.logger.record("eval/mean_jit_cost", mean_jit)

            # 保存历史最佳模型
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(
                        f"\n🔥 [Validation] 发现新突破! 平均验证奖励从 {self.best_mean_reward:.2f} 提升至 {mean_reward:.2f}")
                    print(f"   -> 验证集利用率: {mean_util:.2%}, 验证集 JIT 成本: {mean_jit:.2f}")
                self.best_mean_reward = mean_reward
                self.model.save(f"{self.save_path}/best_model")

        return True

    def _evaluate(self):
        ep_rewards = []
        ep_utils = []
        ep_jits = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_r = 0.0

            while not done:
                # 动态获取当前合法动作掩码
                mask = self.eval_env.get_wrapper_attr("_get_action_mask")()
                # 验证时必须使用 deterministic=True (确定性策略，剥离探索噪声)
                action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_r += reward
                done = terminated or truncated

                if done:
                    metrics = self.eval_env.get_wrapper_attr("cost_metrics")
                    ep_utils.append(metrics.get('utilization', 0.0))
                    ep_jits.append(metrics.get('cost_jit', 0.0))

            ep_rewards.append(ep_r)

        return np.mean(ep_rewards), np.mean(ep_utils), np.mean(ep_jits)
def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()

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
    files_to_backup = ["train_dual.py", "custom_callbacks.py", "config.py", "evaluate_generalization.py", "packing_envs.py", "scheduling_envs.py", "evaluate_batch.py", "test_result.py"]
    for f in files_to_backup:
        if os.path.exists(f): shutil.copy(f, code_dir)

    for folder in ["envs", "heuristic", "models"]:
        src, dst = folder, os.path.join(code_dir, folder)
        if os.path.exists(src):
            if os.path.exists(dst): shutil.rmtree(dst)
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
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
        features_extractor_kwargs=dict(features_dim=256, item_dim=46, global_prefix_dim=0),
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])
    )

    # 【修改点】：机器3维 + 上游感知3维 = 6维全局信息
    pk_sched = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, item_dim=4, global_prefix_dim=6),
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    lr_start = TRAIN_CONFIG.get('lr_start', 3e-4)
    lr_end = TRAIN_CONFIG.get('lr_end', 1e-5)

    print("Init Nesting PPO (Tanh Activated)...")
    nest_model = MaskablePPO("MlpPolicy", nest_env, policy_kwargs=pk_nest, learning_rate=exponential_schedule(lr_start, lr_end), n_steps=8192, batch_size=2048, gamma=0.99, ent_coef=0.02, tensorboard_log=log_n, verbose=1, max_grad_norm=0.5)

    print("Init Scheduling PPO (Tanh Activated)...")
    sched_model = MaskablePPO("MlpPolicy", sched_env, policy_kwargs=pk_sched, learning_rate=exponential_schedule(lr_start, lr_end), n_steps=8192, batch_size=2048, gamma=0.99, ent_coef=0.05, tensorboard_log=log_s, verbose=1, max_grad_norm=0.1, clip_range=0.1)

    # === [新增：构建独立的检测（验证）环境] ===
    eval_nest_env = NestingSchedulingEnv()
    eval_nest_env = ActionMasker(eval_nest_env, mask_fn)
    # 给排样验证环境装载调度器作为陪练
    eval_nest_env.unwrapped.set_scheduling_partner(sched_model)

    # 初始化验证拦截器
    eval_callback = ValidationCallback(
        eval_env=eval_nest_env,
        eval_freq=15000,  # 每 1.5 万步在验证集上考一次试
        n_eval_episodes=5,  # 每次考试跑 5 局取平均
        save_path=save_dir  # 最佳模型会保存在这里
    )

    # 把 eval_callback 塞进你原有的回调列表里
    cb = CallbackList([
        CheckpointCallback(100000, save_dir, 'nest'),
        TensorboardCallback(),
        eval_callback  # <--- 验证功能在此处生效
    ])

    steps = TRAIN_CONFIG['steps_per_cycle']

    # 【修改点】：引入“课程式冻结”分段训练法 (Curriculum Freezing)
    print(f"\n{'='*50}\nPhase 1: Nesting Warm-up (排样预热期)\n{'='*50}")
    nest_env.unwrapped.set_scheduling_partner(None)
    nest_model.learn(steps * 10, reset_num_timesteps=False, callback=cb)
    nest_model.save(f"{save_dir}/nesting_phase1")

    print(f"\n{'='*50}\nPhase 2: Scheduling Adaptation (调度适应期)\n{'='*50}")
    sched_env.unwrapped.set_nesting_partner(nest_env, nest_model)
    sched_model.learn(steps * 10, reset_num_timesteps=False)
    sched_model.save(f"{save_dir}/scheduling_phase2")

    print(f"\n{'='*50}\nPhase 3: Joint Fine-tuning (联合微调期)\n{'='*50}")
    fine_tune_cycles = TRAIN_CONFIG['total_cycles'] - 20
    for c in range(max(1, fine_tune_cycles)):
        print(f"\n===== Joint Cycle {c + 1} =====")
        nest_env.unwrapped.set_scheduling_partner(sched_model)
        nest_model.learn(steps, reset_num_timesteps=False, callback=cb)
        nest_model.save(f"{save_dir}/nesting_joint_c{c + 1}")

        sched_env.unwrapped.set_nesting_partner(nest_env, nest_model)
        sched_model.learn(steps, reset_num_timesteps=False)
        sched_model.save(f"{save_dir}/scheduling_joint_c{c + 1}")

if __name__ == "__main__":
    main()