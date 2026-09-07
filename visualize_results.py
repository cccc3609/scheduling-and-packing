import os
import glob
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from models.attention_extractor import AttentionFeatureExtractor  # 🟢 必须引入，否则加载模型报错

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def get_colors(n):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]


def find_latest_experiment_models(exp_root="./experiments"):
    if not os.path.exists(exp_root):
        print(f"❌ 找不到实验目录: {exp_root}")
        return None, None, None

    exp_dirs = glob.glob(os.path.join(exp_root, "exp_*"))
    if not exp_dirs:
        print("❌ 没有找到实验记录")
        return None, None, None

    latest_exp = max(exp_dirs, key=os.path.getctime)
    print(f"📂 锁定最新实验: {latest_exp}")

    model_dir = os.path.join(latest_exp, "models")
    cycles = []
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if "nesting_c" in f:
                try:
                    cycles.append(int(f.split("_c")[1].split(".zip")[0]))
                except:
                    pass

    if not cycles:
        print("❌ 该实验下没有模型文件")
        return None, None, None

    latest_c = max(cycles)
    print(f"🔥 加载第 {latest_c} 轮模型...")
    nest_path = os.path.join(model_dir, f"nesting_c{latest_c}")
    sched_path = os.path.join(model_dir, f"scheduling_c{latest_c}")
    return nest_path, sched_path, latest_exp


def plot_nesting(plates, order_colors, save_dir="."):
    total_plates = len(plates)
    if total_plates == 0: return

    cols = 4
    rows = math.ceil(total_plates / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(f"Final Nesting Result: {total_plates} Plates", y=0.99, fontsize=16)

    if rows * cols == 1: axes = np.array([axes])
    axes = axes.flatten()

    avg_util = np.mean([p.utilization for p in plates])
    print(f"📊 平均利用率: {avg_util:.2%}")

    for i, ax in enumerate(axes):
        if i < total_plates:
            p = plates[i]
            ax.add_patch(patches.Rectangle((0, 0), p.width, p.height, fc='white', ec='black', lw=2))
            for part in p.placed_parts:
                x, y, w, h, oid = part[:5]
                is_rot = part[5] if len(part) > 5 else False
                c = order_colors[int(oid) % len(order_colors)]
                ax.add_patch(patches.Rectangle((x, y), w, h, fc=c, alpha=0.85, ec='black', lw=1))
                if w > (p.width * 0.05) and h > (p.height * 0.05):
                    txt = f"{oid}" + ("R" if is_rot else "")
                    ax.text(x + w / 2, y + h / 2, txt, color='white', ha='center', va='center', fontsize=8,
                            fontweight='bold')
            ax.set_xlim(0, p.width);
            ax.set_ylim(0, p.height)
            ax.set_title(f"Plate {i} (Util: {p.utilization:.1%})", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "result_nesting.png"), dpi=150)
    plt.show()


def plot_gantt(logs, orders, order_colors, save_dir="."):
    if not logs: return
    machine_ids = sorted(list(set(t['machine_id'] for t in logs)))
    num_machines = len(machine_ids)
    max_time = max([t['end'] for t in logs])

    fig, ax = plt.subplots(figsize=(16, 8))
    for t in logs:
        mid, start, duration = t['machine_id'], t['start'], t['end'] - t['start']
        ax.broken_barh([(start, duration)], (mid * 10, 8), fc='#87CEEB', ec='black')
        ax.text(start + duration / 2, mid * 10 + 4, f"P{t['plate_idx']}", ha='center', va='center', fontsize=8)

    sorted_orders = sorted(orders.items(), key=lambda x: x[0])
    for oid, info in sorted_orders:
        due, fin = info['due_date'], info['finished_time']
        c = order_colors[int(oid) % len(order_colors)]
        ax.axvline(x=due, color=c, ls='--', alpha=0.6, lw=1.5)
        ax.text(due, 32 + (int(oid) % 3) * 3, f"Ord{oid}", color=c, rotation=90, fontsize=9, fontweight='bold')
        if fin > due:
            y_pos = -2 - (int(oid) % 5) * 2
            ax.hlines(y=y_pos, xmin=due, xmax=fin, colors='red', lw=2)
            ax.text(fin, y_pos, "Late", color='red', fontsize=6, va='center')

    ax.set_yticks([i * 10 + 4 for i in machine_ids])
    ax.set_yticklabels([f"Machine {i}" for i in machine_ids])
    ax.set_xlabel("Time");
    ax.set_title("Production Schedule Gantt Chart")
    ax.grid(True, axis='x', ls=':', alpha=0.3)
    ax.set_xlim(0, max_time * 1.1);
    ax.set_ylim(-15, num_machines * 10 + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "result_gantt.png"), dpi=150)
    plt.show()


def plot_jit_analysis(orders, save_dir="."):
    if not orders: return
    order_ids, deviations, colors = [], [], []
    for oid, info in sorted(orders.items()):
        diff = info['finished_time'] - info['due_date']
        order_ids.append(f"Ord{oid}")
        deviations.append(diff)
        colors.append('#FF6B6B' if diff > 0 else '#51CF66')

    fig = plt.figure(figsize=(12, 6))
    plt.bar(order_ids, deviations, color=colors, edgecolor='black')
    plt.axhline(0, color='black', lw=1)
    plt.ylabel("Time Deviation (Finish - Due)")
    plt.title("JIT Deviation Analysis (Red=Late, Green=Early)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "result_jit_analysis.png"), dpi=150)
    plt.show()


def main():
    nest_path, sched_path, exp_dir = find_latest_experiment_models()
    if not nest_path: return

    # 初始化环境 (注意：无需传 max_parts，内部读取 config)
    nest_env = NestingSchedulingEnv()
    nest_env = ActionMasker(nest_env, mask_fn)
    sched_env = SchedulingEnv()
    sched_env = ActionMasker(sched_env, mask_fn)

    print("⏳ 正在加载模型...")
    nest_model = MaskablePPO.load(nest_path)
    sched_model = MaskablePPO.load(sched_path)

    print("🚀 正在生成排样方案...")
    # 🟢 固定种子和参数进行展示
    obs, _ = nest_env.reset(seed=42, options={"num_parts": 50, "plate_size": (200, 200)})
    done = False
    while not done:
        mask = get_action_masks(nest_env)
        action, _ = nest_model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, _, _ = nest_env.step(action)
        done = terminated

    raw_env = nest_env.unwrapped
    plates = raw_env.history_plates
    logs = raw_env.scheduler_state_machine.log
    orders = raw_env.orders
    colors = get_colors(len(orders) + 5)

    print(f"📸 结果将保存至: {exp_dir}")
    plot_nesting(plates, colors, save_dir=exp_dir)
    plot_gantt(logs, orders, colors, save_dir=exp_dir)
    plot_jit_analysis(orders, save_dir=exp_dir)


if __name__ == "__main__":
    main()
