import os
import shutil
import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_problem_provider import SchedulingProblemProviderWrapper
from integration.scheduling_terminal_reward import (
    SchedulingTerminalRewardWrapper,
    configure_terminal_evaluator_for_phase,
)
from config import TRAIN_CONFIG
from models.pointer_extractor import NestingModel
from train_dual import (
    N_ACTIONS_PER,
    NestingModelPredictor,
    NestingPPO,
    load_phase3_pair_metadata,
    set_phase3_nesting_lr,
    start_fresh_scheduling_block,
    train_phase3_round,
)

# ================= 配置区域 (请修改这里) =================
# 1. 上次中断的实验文件夹路径
PREV_EXP_DIR = "./experiments/exp_20260117_170916_resumed_from_c18"

# 2. 从第几轮开始续训
START_CYCLE = 22

# 3. 总共要跑多少轮
TOTAL_CYCLES = max(1, TRAIN_CONFIG["total_cycles"] - 20)

# Explicit lifecycle selection; do not infer the evaluator from checkpoint files.
RESUME_PHASE = "phase3"
VALID_RESUME_PHASES = frozenset({"phase1", "phase2", "phase3"})


# =======================================================

def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def build_resume_nesting_env(resume_phase="phase1"):
    """Build the resume nesting env with an explicit lifecycle phase."""
    if resume_phase not in VALID_RESUME_PHASES:
        raise ValueError("resume_phase must be 'phase1', 'phase2', or 'phase3'")
    nest_base = NestingSchedulingEnv()
    nest_terminal_env = SchedulingTerminalRewardWrapper(
        nest_base, evaluation_mode="edd",
        scheduling_env_factory=SchedulingEnv)
    nest_terminal_env.resume_phase = resume_phase
    return ActionMasker(nest_terminal_env, mask_fn)


def configure_resume_terminal_evaluator(nesting_env, resume_phase, scheduling_model=None):
    """Apply the explicitly selected evaluator after checkpoints are loaded."""
    terminal_wrapper = nesting_env.env
    evaluator_phase = "phase3" if resume_phase == "phase3" else "phase1"
    return configure_terminal_evaluator_for_phase(
        terminal_wrapper, evaluator_phase, scheduling_policy=scheduling_model)


def build_resume_nesting_trainer(terminal_env, device="cpu"):
    """Recreate the current custom Nesting trainer, not the legacy SB3 agent."""
    layout = terminal_env.unwrapped.layout
    model = NestingModel(
        part_feat_dim=layout.part_dim,
        state_feat_dim=layout.state_dim,
        embed_dim=128,
        n_heads=4,
        n_enc_layers=2,
        n_actions_per_part=N_ACTIONS_PER,
        max_parts=layout.max_parts,
        layout=layout,
    )
    return NestingPPO(
        env=terminal_env,
        model=model,
        lr=TRAIN_CONFIG.get("lr_start", 3e-4),
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


def build_isolated_scheduling_training(nesting_ppo):
    """Use current Nesting weights with provider-owned mutable environment state."""
    training_nesting_env = nesting_ppo.env.unwrapped
    provider_nesting_env = NestingSchedulingEnv(
        observation_layout=training_nesting_env.layout)
    if provider_nesting_env is training_nesting_env:
        raise RuntimeError("Scheduling provider must not reuse the Nesting training env")
    provider_predictor = NestingModelPredictor(nesting_ppo, provider_nesting_env)
    scheduling_base = SchedulingEnv()
    provider = SchedulingProblemProviderWrapper(
        scheduling_base, provider_nesting_env, provider_predictor)
    return ActionMasker(provider, mask_fn), provider_nesting_env


def run_resume_cycle(phase, nesting_ppo, terminal_env, scheduling_model,
                     scheduling_env, steps, save_dir, round_id):
    """Run only the agent updates belonging to the explicitly selected phase."""
    if phase not in VALID_RESUME_PHASES:
        raise ValueError("phase must be 'phase1', 'phase2', or 'phase3'")
    if phase == "phase1":
        terminal_env.set_evaluator("edd")
        nesting_ppo.learn(
            steps, save_path=save_dir, tag=f"nesting_phase1_resumed_c{round_id}",
            checkpoint_phase="phase1", checkpoint_round=0)
        return
    if scheduling_model is None or scheduling_env is None:
        raise ValueError(f"{phase} resume requires a scheduling model and env")
    if phase == "phase2":
        start_fresh_scheduling_block(scheduling_model, scheduling_env)
        scheduling_model.learn(steps, reset_num_timesteps=False)
        scheduling_model.save(
            f"{save_dir}/scheduling_phase2_resumed_c{round_id}")
        return
    train_phase3_round(
        nesting_ppo, terminal_env, scheduling_model, scheduling_env,
        steps, save_dir, round_id)


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


def resolve_resume_checkpoints(model_dir, phase, last_completed_round):
    """Resolve only authoritative current-mainline checkpoint names."""
    if phase not in VALID_RESUME_PHASES:
        raise ValueError("phase must be 'phase1', 'phase2', or 'phase3'")
    if phase == "phase3":
        metadata = load_phase3_pair_metadata(model_dir, last_completed_round)
        return (
            os.path.join(model_dir, metadata["nesting_checkpoint"]),
            os.path.join(model_dir, metadata["scheduling_checkpoint"]),
            "phase3",
            int(last_completed_round),
        )

    nesting_path = os.path.join(model_dir, "nesting_phase1_final.pt")
    scheduling_path = None
    if phase == "phase2":
        scheduling_path = os.path.join(model_dir, "scheduling_phase2.zip")
    required = [nesting_path] + ([scheduling_path] if scheduling_path else [])
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing authoritative {phase} resume checkpoint(s): {missing}")
    return nesting_path, scheduling_path, "phase1", 0


def main():
    if RESUME_PHASE not in VALID_RESUME_PHASES:
        raise ValueError("RESUME_PHASE must be 'phase1', 'phase2', or 'phase3'")
    prev_model_dir = os.path.join(PREV_EXP_DIR, "models")
    last_completed_round = START_CYCLE - 1
    nest_path, sched_path, checkpoint_phase, checkpoint_round = (
        resolve_resume_checkpoints(
            prev_model_dir, RESUME_PHASE, last_completed_round))

    log_n, log_s, save_dir = setup_resume_experiment()

    print("⏳ 初始化环境...")
    wrapped_nest_env = build_resume_nesting_env(RESUME_PHASE)
    terminal_nest_env = wrapped_nest_env.env
    nesting_ppo = build_resume_nesting_trainer(terminal_nest_env, device="cpu")
    nesting_ppo.load_training_checkpoint(
        nest_path, expected_phase=checkpoint_phase,
        expected_round=checkpoint_round, map_location="cpu")

    scheduling_model = scheduling_env = None
    if RESUME_PHASE in {"phase2", "phase3"}:
        scheduling_env, _ = build_isolated_scheduling_training(nesting_ppo)
        scheduling_model = MaskablePPO.load(
            sched_path, env=scheduling_env, tensorboard_log=log_s,
            device="cpu", force_reset=True)
    configure_resume_terminal_evaluator(
        wrapped_nest_env, RESUME_PHASE,
        scheduling_model=scheduling_model)

    steps = TRAIN_CONFIG['steps_per_cycle']
    print(f"🚀 开始续训: Cycle {START_CYCLE} -> {TOTAL_CYCLES}")

    for c in range(START_CYCLE, TOTAL_CYCLES + 1):
        print(f"\n===== Cycle {c}/{TOTAL_CYCLES} (Resumed) =====")
        if RESUME_PHASE == "phase3":
            set_phase3_nesting_lr(
                nesting_ppo, c, TOTAL_CYCLES,
                TRAIN_CONFIG.get("lr_start", 3e-4),
                TRAIN_CONFIG.get("lr_end", 1e-5))
        run_resume_cycle(
            RESUME_PHASE, nesting_ppo, terminal_nest_env,
            scheduling_model, scheduling_env, steps, save_dir, c)

    print("✅ 续训全部完成！")


if __name__ == "__main__":
    main()
