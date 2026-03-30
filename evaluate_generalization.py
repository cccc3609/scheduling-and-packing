import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from models.attention_extractor import AttentionFeatureExtractor
from config import TEST_SCENARIOS

# 引入之前的绘图函数 (复用 visualize_results.py)
# 确保 visualize_results.py 在同一目录下
try:
    from test_result import plot_nesting, plot_gantt, plot_jit_analysis, get_colors
except ImportError:
    print("❌ 错误：找不到 visualize_results.py，请确保该文件在根目录下。")
    exit()


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def find_latest_experiment_models(exp_root="./experiments"):
    """查找最新实验的模型路径"""
    if not os.path.exists(exp_root):
        # 兼容旧路径
        if os.path.exists("./logs_dual/models"):
            return find_latest_in_dir("./logs_dual/models"), "./logs_dual/"
        return None, None, None

    exp_dirs = glob.glob(os.path.join(exp_root, "exp_*"))
    if not exp_dirs: return None, None, None

    latest_exp = max(exp_dirs, key=os.path.getctime)
    model_dir = os.path.join(latest_exp, "models")
    return find_latest_in_dir(model_dir), latest_exp


def find_latest_in_dir(model_dir):
    if not os.path.exists(model_dir): return None, None
    files = os.listdir(model_dir)
    cycles = []

    # 优先寻找联合微调 (Phase 3) 的最新模型
    for f in files:
        if "nesting_joint_c" in f:
            try:
                cycles.append(int(f.split("_joint_c")[1].split(".zip")[0]))
            except:
                pass

    if cycles:
        latest = max(cycles)
        return os.path.join(model_dir, f"nesting_joint_c{latest}"), os.path.join(model_dir,
                                                                                 f"scheduling_joint_c{latest}")

    # 如果没找到 Phase 3，尝试加载 Phase 1 和 Phase 2 的独立模型
    if "nesting_phase1.zip" in files and "scheduling_phase2.zip" in files:
        print("💡 未检测到联合微调模型，降级使用 Phase 1 & 2 预热模型。")
        return os.path.join(model_dir, "nesting_phase1"), os.path.join(model_dir, "scheduling_phase2")

    return None, None


def run_evaluation():
    # 1. 寻找模型
    (nest_path, sched_path), exp_dir = find_latest_experiment_models()
    if not nest_path:
        print("❌ 未找到模型文件，请先训练。")
        return

    print(f"📂 锁定实验目录: {exp_dir}")
    print(f"🔥 加载模型: {os.path.basename(nest_path)}")

    # 2. 准备输出目录
    eval_report_dir = os.path.join(exp_dir, "comprehensive_report")
    os.makedirs(eval_report_dir, exist_ok=True)

    print(f"📂 评估报告将保存在: {eval_report_dir}")

    # 3. 初始化环境
    nest_env = NestingSchedulingEnv()  # 参数由 reset options 控制
    nest_env = ActionMasker(nest_env, mask_fn)

    sched_env = SchedulingEnv()
    sched_env = ActionMasker(sched_env, mask_fn)

    # 4. 加载并注入模型
    try:
        nest_model = MaskablePPO.load(nest_path)
        sched_model = MaskablePPO.load(sched_path)
        nest_env.unwrapped.set_scheduling_partner(sched_model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    summary_data = []

    # 5. 遍历测试场景
    print(f"\n🚀 开始执行综合测试 (共 {len(TEST_SCENARIOS)} 个场景)...")

    for scenario in TEST_SCENARIOS:
        scene_name = scenario["name"]
        num_parts = scenario["num_parts"]
        plate_size = scenario["plate_size"]

        print(f"\n🧪 Testing: {scene_name} | Parts: {num_parts} | Size: {plate_size}")

        # 为每个场景创建图片保存文件夹
        # 处理文件名中的空格和特殊字符
        safe_name = scene_name.replace(" ", "_").replace("(", "").replace(")", "")
        scene_img_dir = os.path.join(eval_report_dir, safe_name)
        os.makedirs(scene_img_dir, exist_ok=True)

        metrics = {
            "utilization": [], "total_cost": [], "mat_cost": [], "jit_cost": [],
            "late_rate": [], "plate_count": []
        }

        # 每个场景跑 20 局取平均
        NUM_EPISODES = 20

        for i in tqdm(range(NUM_EPISODES), desc=f"Scenario: {safe_name}"):
            # 🟢 动态重置环境参数
            obs, _ = nest_env.reset(seed=2000 + i, options={
                "num_parts": num_parts,
                "plate_size": plate_size
            })

            done = False
            while not done:
                mask = nest_env.get_wrapper_attr("_get_action_mask")()
                action, _ = nest_model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, _, info = nest_env.step(action)
                done = terminated

                if done:
                    # 收集数据
                    cost_m = nest_env.unwrapped.cost_metrics
                    eps_m = info["episode_metrics"]

                    metrics["utilization"].append(cost_m.get("utilization", 0))
                    metrics["total_cost"].append(cost_m.get("cost_total", 0))
                    metrics["mat_cost"].append(cost_m.get("cost_material", 0))
                    metrics["jit_cost"].append(cost_m.get("cost_jit", 0))

                    # 迟到率
                    late_cnt = eps_m.get("late_orders_count", 0)
                    total_orders = len(nest_env.unwrapped.orders)
                    metrics["late_rate"].append(late_cnt / total_orders if total_orders else 0)

                    metrics["plate_count"].append(cost_m.get("plate_count", 0))

        # === 记录该场景的统计结果 ===
        summary = {
            "Scenario": scene_name,
            "Parts": num_parts,
            "Plate Size": str(plate_size),
            "Avg Util": np.mean(metrics["utilization"]),
            "Avg Cost": np.mean(metrics["total_cost"]),
            "Mat Cost": np.mean(metrics["mat_cost"]),
            "JIT Cost": np.mean(metrics["jit_cost"]),
            "Late Rate": np.mean(metrics["late_rate"]),
            "Avg Plates": np.mean(metrics["plate_count"])
        }
        summary_data.append(summary)

        # === 为该场景生成一套可视化图表 (取最后一局的数据) ===
        raw_env = nest_env.unwrapped
        colors = get_colors(len(raw_env.orders) + 5)

        # 1. 排样图
        plot_nesting(raw_env.history_plates, colors, save_dir=scene_img_dir, file_prefix="sample")
        # 2. 甘特图
        plot_gantt(raw_env.scheduler_state_machine.log, raw_env.orders, colors, save_dir=scene_img_dir,
                   file_prefix="sample")
        # 3. JIT 分析
        plot_jit_analysis(raw_env.orders, raw_env.cost_metrics, save_dir=scene_img_dir, file_prefix="sample")

    # 6. 生成并保存汇总报表
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(eval_report_dir, "final_summary_report.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print("📊 FINAL COMPREHENSIVE EVALUATION REPORT")
    print("=" * 100)
    # 格式化输出
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 100)
    print(f"✅ 所有详细图表和数据已保存在: {eval_report_dir}")


if __name__ == "__main__":
    run_evaluation()