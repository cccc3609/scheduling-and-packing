import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from integration.scheduling_terminal_reward import make_dual_agent_terminal_wrapper
from config import COST_CONFIG
from evaluate_batch import load_nesting_model, run_rl_episode
from integration.evaluation_checkpoint import resolve_latest_phase3_pair
from integration.evaluation_results import (
    build_manifest, case_record, create_run_directory, formal_metrics,
    make_evaluation_case, make_run_id, write_evaluation_results,
)
from models.sched_policy_loader import load_scheduling_policy

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def get_colors(n):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]


def find_latest_experiment_models(exp_root="./experiments"):
    pair = resolve_latest_phase3_pair(exp_root)
    return (str(pair.nesting_checkpoint), str(pair.scheduling_checkpoint),
            str(pair.experiment_dir))


def plot_nesting(plates, order_colors, save_dir=".", file_prefix="result"):
    """绘制排样图 (支持文件前缀 + 修正利用率)"""
    total_plates = len(plates)
    if total_plates == 0: return

    # 计算利用率
    utils = [p.utilization for p in plates]
    avg_util = np.mean(utils)

    # 计算剔除最差板后的修正利用率
    if total_plates > 1:
        adj_utils = sorted(utils)[1:]
        adj_avg = np.mean(adj_utils)
        title_str = f"Avg Util: {avg_util:.1%} | Adj Util (Drop Min): {adj_avg:.1%}"
    else:
        title_str = f"Avg Util: {avg_util:.1%}"

    print(f"Nesting Stats: {title_str}")

    cols = 4
    rows = math.ceil(total_plates / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    # 动态标题
    fig.suptitle(f"Nesting Result ({file_prefix}): {total_plates} Plates\n{title_str}", y=0.99, fontsize=14)

    if rows * cols == 1: axes = np.array([axes])
    axes = axes.flatten()

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
            ax.set_title(f"P{i} Util:{p.utilization:.1%}", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    fname = f"{file_prefix}_nesting.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def plot_gantt(logs, orders, order_colors, save_dir=".", file_prefix="result"):
    """绘制甘特图 (支持文件前缀)"""
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
    ax.set_yticklabels([f"M{i}" for i in machine_ids])
    ax.set_xlabel("Time");
    ax.set_title(f"Gantt Chart ({file_prefix})")
    ax.grid(True, axis='x', ls=':', alpha=0.3)
    ax.set_xlim(0, max_time * 1.1);
    ax.set_ylim(-15, num_machines * 10 + 5)

    plt.tight_layout()
    fname = f"{file_prefix}_gantt.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def plot_jit_analysis(orders, metrics, save_dir=".", file_prefix="result"):
    """绘制JIT分析面板 (无Emoji版)"""
    if not orders: return
    order_ids, deviations, colors = [], [], []
    for oid, info in sorted(orders.items()):
        diff = info['finished_time'] - info['due_date']
        order_ids.append(f"Ord{oid}")
        deviations.append(diff)
        colors.append('#FF6B6B' if diff > 0 else '#51CF66')

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Cost & JIT Analysis ({file_prefix})", fontsize=18, y=0.95)

    # 1. 偏差图
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.bar(order_ids, deviations, color=colors, edgecolor='black')
    ax1.axhline(0, color='black', lw=1)
    ax1.set_ylabel("Time Deviation")
    ax1.set_title("Per-Order Deviation", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. 成本饼图
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    if 'cost_material' in metrics:
        costs = [metrics['cost_material'], metrics['cost_jit']]
        labels = [f"Material\n({costs[0]:.0f})", f"JIT Penalty\n({costs[1]:.0f})"]
        ax2.pie(costs, labels=labels, autopct='%1.1f%%', colors=['#4dabf7', '#ff6b6b'], startangle=90)
        ax2.set_title("Total Cost Composition", fontsize=14)
    else:
        ax2.text(0.5, 0.5, "No Cost Data", ha='center')

    # 3. 统计信息 (纯文本)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis('off')

    late_orders = sum(1 for d in deviations if d > 0.1)
    late_rate = late_orders / len(orders)
    mat_rate = COST_CONFIG['cost_material']
    tard_rate = COST_CONFIG['cost_tardiness']

    text_str = (
        f"Performance Summary\n"
        f"--------------------------\n"
        f"Total Orders   : {len(orders)}\n"
        f"Late Orders    : {late_orders} ({late_rate:.1%})\n"
        f"Total Cost     : {metrics.get('cost_total', 0):.2f}\n"
        f"  - Material   : {metrics.get('cost_material', 0):.2f}\n"
        f"  - JIT Penalty: {metrics.get('cost_jit', 0):.2f}\n"
        f"--------------------------\n"
        f"Config Rates:\n"
        f"  Material     : {mat_rate} / area\n"
        f"  Tardiness    : {tard_rate} / area / time"
    )
    ax3.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center')

    plt.tight_layout()
    fname = f"{file_prefix}_analysis.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def main():
    pair = resolve_latest_phase3_pair()
    nest_path = str(pair.nesting_checkpoint)
    sched_path = str(pair.scheduling_checkpoint)

    # Initialize the base environment; bind the policy evaluator after loading.
    nest_base = NestingSchedulingEnv()

    print("Loading Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nest_model = load_nesting_model(nest_path, nest_base.layout, device)
    sched_model = load_scheduling_policy(sched_path, device=device)
    nest_terminal_env = make_dual_agent_terminal_wrapper(
        nest_base, sched_model)

    print("Generating Visualization...")

    # 🟢 固定参数进行展示
    TEST_N = 50
    TEST_W = 200
    TEST_H = 200
    file_prefix = f"N{TEST_N}_Size{TEST_W}x{TEST_H}"
    base_seed = 42
    case = make_evaluation_case(
        0, base_seed, num_parts=TEST_N, plate_size=(TEST_W, TEST_H),
        scenario="visualization", config={"evaluation": "test_result"})
    run_id = make_run_id("test_result", pair, base_seed)
    run_dir = create_run_directory("./evaluation_results", run_id)

    run_rl_episode(
        nest_terminal_env, nest_model, device=device, seed=case.case_seed,
        options={"instance": case.independent_instance()})

    raw_env = nest_base
    final_metrics = raw_env.cost_metrics
    plates = raw_env.history_plates
    logs = raw_env.scheduler_state_machine.log
    orders = raw_env.orders
    colors = get_colors(len(orders) + 5)

    metrics = formal_metrics(
        plates, orders, raw_env.parts_pool, raw_env.plate_w, raw_env.plate_h)
    record = case_record(
        run_id, case, "Dual-Agent RL", evaluation_mode="policy",
        pair=pair, metrics=metrics)
    manifest = build_manifest(
        run_id, "test_result", "policy", base_seed, 1, pair,
        {"num_parts": TEST_N, "plate_size": [TEST_W, TEST_H],
         "num_machines": case.instance.num_machines})
    write_evaluation_results(run_dir, manifest, [record])

    print(f"Results saved to: {run_dir}")

    plot_nesting(plates, colors, save_dir=str(run_dir), file_prefix=file_prefix)
    plot_gantt(logs, orders, colors, save_dir=str(run_dir), file_prefix=file_prefix)
    plot_jit_analysis(orders, final_metrics, save_dir=str(run_dir), file_prefix=file_prefix)


if __name__ == "__main__":
    main()
