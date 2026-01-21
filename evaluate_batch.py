import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker

# 引入环境
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def find_latest_experiment_models(exp_root="./experiments"):
    """
    自动查找 experiments 文件夹下最新的实验，以及该实验中最新的 Cycle 模型
    """
    if not os.path.exists(exp_root):
        fallback_dir = "./logs_dual/models/"
        if os.path.exists(fallback_dir):
            print(f"⚠️ 未找到 {exp_root}，尝试使用旧版路径: {fallback_dir}")
            return find_latest_in_dir(fallback_dir)
        return None, None, 0

    exp_dirs = glob.glob(os.path.join(exp_root, "exp_*"))
    if not exp_dirs:
        return None, None, 0

    latest_exp = max(exp_dirs, key=os.path.getctime)
    model_dir = os.path.join(latest_exp, "models")

    print(f"📂 锁定最新实验目录: {latest_exp}")
    return find_latest_in_dir(model_dir)


def find_latest_in_dir(model_dir):
    if not os.path.exists(model_dir):
        return None, None, 0

    files = os.listdir(model_dir)
    cycles = []
    for f in files:
        if "nesting_c" in f:
            try:
                c = int(f.split("_c")[1].split(".zip")[0])
                cycles.append(c)
            except:
                pass

    if not cycles: return None, None, 0

    latest = max(cycles)
    nesting_path = os.path.join(model_dir, f"nesting_c{latest}")
    scheduling_path = os.path.join(model_dir, f"scheduling_c{latest}")
    return nesting_path, scheduling_path, latest


def evaluate(num_episodes=50):
    print(f"正在准备批量评估 ({num_episodes} 局)...")

    # 1. 自动查找模型
    n_path, s_path, cycle = find_latest_experiment_models()

    if not n_path:
        print("未找到任何模型文件！请先运行 train_dual.py")
        return

    print(f"加载模型 Cycle: {cycle}")
    print(f"排样模型: {n_path}")
    print(f"调度模型: {s_path}")

    try:
        nesting_model = MaskablePPO.load(n_path)
        scheduling_model = MaskablePPO.load(s_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 准备环境 (参数必须与训练时一致)
    nesting_env = NestingSchedulingEnv(max_parts=80, plate_size=(200, 200))
    nesting_env = ActionMasker(nesting_env, mask_fn)

    scheduling_env = SchedulingEnv(num_machines=3, max_tasks=80)
    scheduling_env = ActionMasker(scheduling_env, mask_fn)

    # 3. 注入伙伴
    nesting_env.unwrapped.set_scheduling_partner(scheduling_model)
    scheduling_env.unwrapped.set_nesting_partner(nesting_env, nesting_model)

    # 4. 统计数据容器
    stats = {
        "utilization": [],
        "jit_cost": [],
        "late_count": [],
        "plates_used": []
    }

    # 5. 开始循环测试
    print("🚀 开始测试...")
    for i in tqdm(range(num_episodes)):
        obs, _ = nesting_env.reset()
        done = False

        while not done:
            action_masks = nesting_env.get_wrapper_attr("_get_action_mask")()
            action, _ = nesting_model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = nesting_env.step(action)
            done = terminated or truncated

            # 在每局结束时收集数据
            if done and "episode_metrics" in info:
                m = info["episode_metrics"]

                # 兼容旧键名，防止 KeyError
                util_val = m.get("raw_avg_utilization", m.get("avg_utilization", 0))
                jit_val = m.get("raw_avg_jit_cost", m.get("total_tardiness", 0))
                late_val = m.get("late_orders_count", 0)
                plate_val = m.get("plate_count", 0)

                stats["utilization"].append(util_val)
                stats["jit_cost"].append(jit_val)
                stats["late_count"].append(late_val)
                stats["plates_used"].append(plate_val)

    # 6. 生成分析报告
    df = pd.DataFrame(stats)

    print("\n" + "=" * 60)
    print(f"📊 测试报告 (基于 {num_episodes} 个随机 Episode | Cycle {cycle})")
    print("=" * 60)
    print(f"{'指标 (Metric)':<30} | {'均值 (Mean)':<10} | {'标准差 (Std)':<10} | {'极值 (Min/Max)'}")
    print("-" * 80)

    print(
        f"{'平均利用率 (Utilization)':<30} | {df['utilization'].mean():.2%}     | {df['utilization'].std():.2%}     | Max: {df['utilization'].max():.2%}")
    print(
        f"{'平均JIT成本 (Avg JIT Cost)':<30} | {df['jit_cost'].mean():.2f}       | {df['jit_cost'].std():.2f}       | Min: {df['jit_cost'].min():.2f}")
    print(
        f"{'迟到订单数 (Late Orders)':<30} | {df['late_count'].mean():.2f}       | {df['late_count'].std():.2f}       | Max: {df['late_count'].max()}")
    print(
        f"{'消耗板材数 (Plates Used)':<30} | {df['plates_used'].mean():.2f}       | {df['plates_used'].std():.2f}       | Min: {df['plates_used'].min()}")
    print("=" * 60)

    # 保存结果
    csv_name = f"eval_result_c{cycle}.csv"
    df.to_csv(csv_name, index_label="Episode_ID")
    print(f"💾 详细数据已保存至: {csv_name}")


if __name__ == "__main__":
    evaluate(num_episodes=50)