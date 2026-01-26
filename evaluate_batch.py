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
from models.attention_extractor import AttentionFeatureExtractor  # 必须引入防止加载报错
from config import MAX_PARTS_CAPACITY

# 引入可视化工具
from visualize_results import plot_nesting, plot_gantt, plot_jit_analysis, get_colors


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


# === 1. 定义测试场景 (Scenarios) ===
TEST_SCENARIOS = [
    # 场景 A: 标准基准 (训练时的典型情况)
    {
        "name": "Standard_50",
        "desc": "标准压力 (50零件, 200x200板)",
        "num_parts": 50,
        "plate_size": (200, 200)
    },
    # 场景 B: 小批量急单 (考验 JIT 响应速度)
    {
        "name": "Small_30",
        "desc": "低负载 (30零件, 200x200板)",
        "num_parts": 30,
        "plate_size": (200, 200)
    },
    # 场景 C: 大批量压力测试 (考验排样填缝能力和调度抗压)
    {
        "name": "HighLoad_100",
        "desc": "高负载 (100零件, 200x200板)",
        "num_parts": 100,
        "plate_size": (200, 200)
    },
    # 场景 D: 异形板材 (扁长条，考验排样算法泛化性)
    {
        "name": "WidePlate_80",
        "desc": "异形板材 (80零件, 400x150板)",
        "num_parts": 80,
        "plate_size": (400, 150)
    },
    # 场景 E: 巨型板材 (工业大件模拟)
    {
        "name": "Huge_100",
        "desc": "巨型板材 (100零件, 350x350板)",
        "num_parts": 100,
        "plate_size": (350, 350)
    }
]


def find_latest_models(exp_root="./experiments"):
    """健壮的模型查找逻辑"""
    if not os.path.exists(exp_root):
        print(f"❌ 找不到实验目录: {exp_root}")
        return None, None, None

    exp_dirs = glob.glob(os.path.join(exp_root, "exp_*"))
    if not exp_dirs:
        print("❌ 没有找到实验记录")
        return None, None, None

    # 按时间倒序
    exp_dirs.sort(key=os.path.getctime, reverse=True)

    for latest_exp in exp_dirs:
        model_dir = os.path.join(latest_exp, "models")
        if not os.path.exists(model_dir): continue

        nest_cycles = set()
        for f in os.listdir(model_dir):
            if f.startswith("nesting_c") and f.endswith(".zip"):
                try:
                    nest_cycles.add(int(f.split("_c")[1].split(".zip")[0]))
                except:
                    pass

        sched_cycles = set()
        for f in os.listdir(model_dir):
            if f.startswith("scheduling_c") and f.endswith(".zip"):
                try:
                    sched_cycles.add(int(f.split("_c")[1].split(".zip")[0]))
                except:
                    pass

        valid = nest_cycles.intersection(sched_cycles)
        if valid:
            c = max(valid)
            print(f"📂 锁定实验: {os.path.basename(latest_exp)}")
            print(f"🔥 加载 Cycle {c} 模型...")
            return (os.path.join(model_dir, f"nesting_c{c}"),
                    os.path.join(model_dir, f"scheduling_c{c}"),
                    latest_exp)

    return None, None, None


def run_evaluation(episodes_per_scenario=20):
    # 1. 加载模型
    nest_path, sched_path, exp_dir = find_latest_models()
    if not nest_path: return

    try:
        nest_model = MaskablePPO.load(nest_path)
        sched_model = MaskablePPO.load(sched_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 准备输出目录
    report_dir = os.path.join(exp_dir, "evaluation_reports")
    os.makedirs(report_dir, exist_ok=True)
    print(f"📄 报告将保存在: {report_dir}")

    # 3. 初始化环境
    # 注意：max_parts_capacity 必须 >= 测试中的最大零件数 (这里是100)
    # 使用 config 中的 MAX_PARTS_CAPACITY (通常是120)
    env_capacity = max(MAX_PARTS_CAPACITY, 120)

    nest_env = NestingSchedulingEnv()
    nest_env.max_capacity = env_capacity  # 强制覆盖容量以适应测试
    nest_env = ActionMasker(nest_env, mask_fn)

    sched_env = SchedulingEnv()
    sched_env.max_tasks = env_capacity
    sched_env = ActionMasker(sched_env, mask_fn)

    # 注入伙伴
    nest_env.unwrapped.set_scheduling_partner(sched_model)
    sched_env.unwrapped.set_nesting_partner(nest_env, nest_model)

    summary_data = []

    # 4. 遍历场景测试
    for scenario in TEST_SCENARIOS:
        scene_name = scenario['name']
        print(f"\n🧪 测试场景: {scene_name} | {scenario['desc']}")

        # 创建该场景的图片文件夹
        scene_dir = os.path.join(report_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        metrics = {
            "utilization": [], "total_delay": [], "plate_count": [],
            "late_orders": [], "mat_cost": [], "jit_cost": []
        }

        # 循环跑 N 局
        for i in tqdm(range(episodes_per_scenario)):
            # 🟢 动态重置环境参数
            obs, _ = nest_env.reset(seed=1000 + i, options={
                "num_parts": scenario['num_parts'],
                "plate_size": scenario['plate_size']
            })

            done = False
            while not done:
                mask = get_action_masks(nest_env)
                action, _ = nest_model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, _, info = nest_env.step(action)
                done = terminated

                if done:
                    # 收集数据
                    m = info.get('episode_metrics', {})
                    cost_m = nest_env.unwrapped.cost_metrics

                    # 兼容不同版本的键名
                    util = m.get('raw_avg_utilization', m.get('avg_utilization', 0))
                    delay = cost_m.get('total_delay', 0)
                    late = m.get('late_orders_count', 0)
                    plates = m.get('plate_count', 0)

                    metrics["utilization"].append(util)
                    metrics["total_delay"].append(delay)
                    metrics["plate_count"].append(plates)
                    metrics["late_orders"].append(late)
                    metrics["mat_cost"].append(cost_m.get('cost_material', 0))
                    metrics["jit_cost"].append(cost_m.get('cost_jit', 0))

        # === 记录该场景平均表现 ===
        summary_data.append({
            "Scenario": scene_name,
            "Avg Util": np.mean(metrics["utilization"]),
            "Avg Plates": np.mean(metrics["plate_count"]),
            "Avg Delay Time": np.mean(metrics["total_delay"]),
            "Avg Late Orders": np.mean(metrics["late_orders"]),
            "Cost Ratio (JIT/Mat)": np.mean(metrics["jit_cost"]) / (np.mean(metrics["mat_cost"]) + 1e-6)
        })

        # === 为该场景的最后一局生成可视化图表 ===
        raw_env = nest_env.unwrapped
        colors = get_colors(len(raw_env.orders) + 5)

        plot_nesting(raw_env.history_plates, colors, save_dir=scene_dir, file_prefix="sample")
        plot_gantt(raw_env.scheduler_state_machine.log, raw_env.orders, colors, save_dir=scene_dir,
                   file_prefix="sample")
        plot_jit_analysis(raw_env.orders, raw_env.cost_metrics, save_dir=scene_dir, file_prefix="sample")

    # 5. 输出汇总报表
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(report_dir, "final_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("📊 综合评估报告 (Summary Report)")
    print("=" * 80)
    # 格式化输出
    print(df.to_string(index=False, float_format="{:.2f}".format))
    print("=" * 80)
    print(f"✅ 测试完成，详细数据与图表已保存至: {report_dir}")


def get_action_masks(env):
    return env.get_wrapper_attr("_get_action_mask")()


if __name__ == "__main__":
    run_evaluation(episodes_per_scenario=20)