import os
import shutil
import datetime
from typing import Callable
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_problem_provider import SchedulingProblemProviderWrapper
from integration.scheduling_terminal_reward import SchedulingTerminalRewardWrapper
from custom_callbacks import TensorboardCallback, SnapshotCallback
from config import TRAIN_CONFIG
from core.nesting_observation import LEGACY_NESTING_SCHEMA_ERROR

# ================= 配置区域 (请修改这里) =================
# 1. 上次中断的实验文件夹路径
PREV_EXP_DIR = "./experiments/exp_20260117_170916_resumed_from_c18"

# 2. 从第几轮开始续训
START_CYCLE = 22

# 3. 总共要跑多少轮
TOTAL_CYCLES = 50


# =======================================================

def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def build_resume_nesting_env():
    """Restore the pre-Patch-2 resume terminal semantics via explicit EDD."""
    nest_base = NestingSchedulingEnv()
    nest_terminal_env = SchedulingTerminalRewardWrapper(
        nest_base, evaluation_mode="edd")
    return ActionMasker(nest_terminal_env, mask_fn)


# 指数衰减调度器 (与 train_dual.py 保持一致)
def exponential_schedule(start_lr: float, end_lr: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        current_progress = 1.0 - progress_remaining
        return start_lr * (end_lr / start_lr) ** current_progress

    return func


def setup_resume_experiment():
    """初始化新的实验目录，用于存放续训的日志"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件夹名字带上 resumed 标记
    exp_name = f"exp_{timestamp}_resumed_from_c{START_CYCLE - 1}"

    base_dir = f"./experiments/{exp_name}"
    log_n = f"{base_dir}/logs/nesting/"
    log_s = f"{base_dir}/logs/scheduling/"
    save_dir = f"{base_dir}/models/"
    code_dir = f"{base_dir}/code_backup/"

    for d in [log_n, log_s, save_dir, code_dir]:
        os.makedirs(d, exist_ok=True)

    # 备份当前代码 (确保这次续训用的代码逻辑被记录)
    files_to_backup = [
        "train_dual.py", "train_resume.py", "custom_callbacks.py",
        "config.py", "visualize_results.py", "plot_training_metrics.py"
    ]
    for f in files_to_backup:
        if os.path.exists(f):
            shutil.copy(f, code_dir)

    for folder in ["envs", "heuristic", "models"]:
        if os.path.exists(folder):
            shutil.copytree(folder, f"{code_dir}/{folder}", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))

    print(f"📦 续训实验目录已创建: {base_dir}")
    return log_n, log_s, save_dir


def main():
    # 1. 检查旧模型是否存在
    prev_model_dir = os.path.join(PREV_EXP_DIR, "models")
    last_completed_cycle = START_CYCLE - 1

    nest_path = os.path.join(prev_model_dir, f"nesting_c{last_completed_cycle}.zip")
    sched_path = os.path.join(prev_model_dir, f"scheduling_c{last_completed_cycle}.zip")

    if not os.path.exists(nest_path) or not os.path.exists(sched_path):
        print(f"❌ 错误：找不到第 {last_completed_cycle} 轮的模型文件！")
        print(f"   检查路径: {nest_path}")
        return

    # 2. 初始化新实验路径
    log_n, log_s, save_dir = setup_resume_experiment()

    # 3. 初始化环境 (使用修复后的最新代码)
    # 这里的环境已经包含了最新的防崩逻辑
    print("⏳ 初始化环境...")
    nest_env = build_resume_nesting_env()

    sched_base = SchedulingEnv()

    # 4. 加载旧模型
    print(f"🔥 加载旧模型 (Cycle {last_completed_cycle})...")

    # 重新定义 LR Schedule，防止加载旧模型时由 pickle 问题导致报错
    lr_start = TRAIN_CONFIG.get('lr_start', 1e-3)
    lr_end = TRAIN_CONFIG.get('lr_end', 1e-5)

    custom_objects = {
        "learning_rate": exponential_schedule(lr_start, lr_end),
        "lr_schedule": exponential_schedule(lr_start, lr_end),
        # 也可以在这里覆盖其他参数，比如 clip_range
        "clip_range": 0.1,
        "max_grad_norm": 0.3
    }

    # 加载 Nesting
    try:
        nest_model = MaskablePPO.load(
            nest_path,
            env=nest_env,  # 绑定新环境
            custom_objects=custom_objects,
            tensorboard_log=log_n,  # 指向新日志目录
            print_system_info=True
        )
    except ValueError as exc:
        if "Observation spaces do not match" in str(exc):
            raise ValueError(LEGACY_NESTING_SCHEMA_ERROR) from exc
        raise

    sched_env = ActionMasker(
        SchedulingProblemProviderWrapper(sched_base, nest_env, nest_model), mask_fn)

    # 加载 Scheduling
    sched_model = MaskablePPO.load(
        sched_path,
        env=sched_env,
        custom_objects=custom_objects,
        tensorboard_log=log_s,
        # 🟢 新增：强制 CPU 运行以获得更详细的报错（如果有），并且重置优化器状态
        device="cpu",
        force_reset=True
    )
    # 5. 回调
    cb = CallbackList([
        CheckpointCallback(50000, save_dir, name_prefix="nest"),
        TensorboardCallback(),
        SnapshotCallback(20000, log_n)
    ])

    # 7. 续训循环
    steps = TRAIN_CONFIG['steps_per_cycle']
    print(f"🚀 开始续训: Cycle {START_CYCLE} -> {TOTAL_CYCLES}")

    for c in range(START_CYCLE, TOTAL_CYCLES + 1):
        print(f"\n===== Cycle {c}/{TOTAL_CYCLES} (Resumed) =====")

        # 训练 Nesting
        print(">>> Training Nesting...")
        nest_model.learn(steps, reset_num_timesteps=False, callback=cb)
        nest_model.save(f"{save_dir}/nesting_c{c}")

        # 训练 Scheduling
        print(">>> Training Scheduling...")
        sched_model.learn(steps, reset_num_timesteps=False)
        sched_model.save(f"{save_dir}/scheduling_c{c}")

    print("✅ 续训全部完成！")


if __name__ == "__main__":
    main()
