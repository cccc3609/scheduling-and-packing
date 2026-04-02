import os
import glob
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from models.attention_extractor import AttentionFeatureExtractor
from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import COST_CONFIG, TRAIN_CONFIG

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()


def get_colors(n):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]


def find_latest_experiment_models(exp_root="./experiments"):
    if not os.path.exists(exp_root): return None, None, None
    exp_dirs = glob.glob(os.path.join(exp_root, "exp_*"))
    if not exp_dirs: return None, None, None
    exp_dirs.sort(key=os.path.getctime, reverse=True)

    for latest_exp in exp_dirs:
        model_dir = os.path.join(latest_exp, "models")
        if not os.path.exists(model_dir): continue
        nest_cycles = set()
        for f in os.listdir(model_dir):
            if f.startswith("nesting_joint_c") and f.endswith(".zip"):
                try:
                    nest_cycles.add(int(f.split("_joint_c")[1].split(".zip")[0]))
                except:
                    pass
        sched_cycles = set()
        for f in os.listdir(model_dir):
            if f.startswith("scheduling_joint_c") and f.endswith(".zip"):
                try:
                    sched_cycles.add(int(f.split("_joint_c")[1].split(".zip")[0]))
                except:
                    pass
        valid_cycles = nest_cycles.intersection(sched_cycles)
        if not valid_cycles: continue
        latest_c = max(valid_cycles)
        print(f"Locked Experiment: {os.path.basename(latest_exp)} | Joint Cycle: {latest_c}")
        return os.path.join(model_dir, f"nesting_joint_c{latest_c}"), os.path.join(model_dir,
                                                                                   f"scheduling_joint_c{latest_c}"), latest_exp
    return None, None, None


def plot_nesting(plates, order_colors, save_dir=".", file_prefix="result"):
    total_plates = len(plates)
    if total_plates == 0: return
    utils = [p.utilization for p in plates]
    avg_util = np.mean(utils)
    title_str = f"Avg Util: {avg_util:.1%}"

    cols = 4
    rows = math.ceil(total_plates / cols)
    fig, axes = plt.subplots(max(1, rows), cols, figsize=(16, 4 * max(1, rows)))
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
    plt.savefig(os.path.join(save_dir, f"{file_prefix}_nesting.png"), dpi=150)
    plt.close()


def plot_gantt(logs, orders, order_colors, save_dir=".", file_prefix="result"):
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
    plt.savefig(os.path.join(save_dir, f"{file_prefix}_gantt.png"), dpi=150)
    plt.close()


def plot_jit_analysis(orders, metrics, save_dir=".", file_prefix="result"):
    if not orders: return
    order_ids, deviations, colors = [], [], []
    for oid, info in sorted(orders.items()):
        diff = info['finished_time'] - info['due_date']
        order_ids.append(f"Ord{oid}")
        deviations.append(diff)
        colors.append('#FF6B6B' if diff > 0 else '#51CF66')

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Cost & JIT Analysis ({file_prefix})", fontsize=18, y=0.95)

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.bar(order_ids, deviations, color=colors, edgecolor='black')
    ax1.axhline(0, color='black', lw=1)
    ax1.set_ylabel("Time Deviation")
    ax1.set_title("Per-Order Deviation (Red=Late, Green=Early)", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = plt.subplot2grid((2, 2), (1, 0))
    if 'cost_material' in metrics:
        costs = [metrics['cost_material'], metrics['cost_jit']]
        labels = [f"Material Waste\n({costs[0]:.0f})", f"JIT Penalty\n({costs[1]:.0f})"]
        ax2.pie(costs, labels=labels, autopct='%1.1f%%', colors=['#4dabf7', '#ff6b6b'], startangle=90)
        ax2.set_title(f"Total Cost: {metrics.get('cost_total', 0):.2f}", fontsize=14)

    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis('off')
    late_orders = sum(1 for d in deviations if d > 0.1)
    text_str = (
        f"Performance Summary\n"
        f"--------------------------\n"
        f"Late Orders    : {late_orders} / {len(orders)}\n"
        f"Total Penalty  : {metrics.get('cost_total', 0):.2f}\n"
        f"Material Waste : {metrics.get('cost_material', 0):.2f}\n"
        f"JIT Penalty    : {metrics.get('cost_jit', 0):.2f}\n"
        f"Avg Utilization: {metrics.get('utilization', 0):.1%}\n"
    )
    ax3.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_prefix}_analysis.png"), dpi=150)
    plt.close()


def run_baseline_on_same_data(parts_pool, orders_raw, plate_w, plate_h):
    import copy
    import numpy as np
    from heuristic.blf_skyline_maxrects import PlateLayoutManager
    from heuristic.scheduler import SchedulerStateMachine
    from config import COST_CONFIG

    parts_pool = copy.deepcopy(parts_pool)
    orders = copy.deepcopy(orders_raw)
    for o in orders.values():
        o['finished_time'] = 0.0

    # 1. FFD 排样
    sorted_parts = sorted(parts_pool, key=lambda x: x['area'], reverse=True)
    active_plates = [PlateLayoutManager(width=plate_w, height=plate_h)]

    for part in sorted_parts:
        placed = False
        for plate in active_plates:
            ok, sx, sy, sw, sh, _ = plate.place_part(part['w'], part['h'], part['order_id'], 1)
            if not ok: ok, sx, sy, sw, sh, _ = plate.place_part(part['w'], part['h'], part['order_id'], 2)
            if ok:
                placed = True
                break
        if not placed:
            new_plate = PlateLayoutManager(width=plate_w, height=plate_h)
            new_plate.place_part(part['w'], part['h'], part['order_id'], 1)
            active_plates.append(new_plate)

    final_plates = [p for p in active_plates if len(p.placed_parts) > 0]

    # 2. EDD 调度
    scheduler = SchedulerStateMachine(num_machines=3)
    tasks = []
    speed = COST_CONFIG['cutting_speed']

    for idx, plate in enumerate(final_plates):
        cut_time = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / speed
        oids = list(set([int(p[4]) for p in plate.placed_parts]))
        min_due = min([orders[o]['due_date'] for o in oids]) if oids else 999.0
        tasks.append({'cut': cut_time, 'due': min_due, 'idx': idx, 'oids': oids, 'done': False})

    tasks.sort(key=lambda x: x['due'])

    for task in tasks:
        mach_times = scheduler.get_state()
        best_mach_idx = np.argmin(mach_times)
        end_t = scheduler.execute_assignment(best_mach_idx, task['cut'], task['idx'])
        for oid in task['oids']:
            orders[oid]['finished_time'] = max(orders[oid]['finished_time'], end_t)

    # 3. 计算对齐的成本
    total_part_area = sum([p['area'] for p in parts_pool])
    consumed_area = len(final_plates) * (plate_w * plate_h)
    utilization = total_part_area / consumed_area if consumed_area > 0 else 0.001

    wasted_area = max(0.0, consumed_area - total_part_area)
    cost_material = wasted_area * COST_CONFIG['cost_material']

    cost_jit = 0.0
    for oid, order in orders.items():
        due, fin = order['due_date'], order['finished_time']
        order_parts = [p for p in parts_pool if p['order_id'] == oid]
        order_value = sum([p['area'] for p in order_parts])
        order_proc_time = max(1.0, sum([2 * (p['w'] + p['h']) for p in order_parts]) / speed)

        diff = fin - due
        ratio = abs(diff) / order_proc_time
        coef = 0.0 if ratio <= 0.025 else ((ratio - 0.025) / 0.05 if ratio <= 0.075 else 1.0)

        if diff > 0:
            cost_jit += order_value * (COST_CONFIG['cost_tardiness'] * coef) * abs(diff)
        else:
            cost_jit += order_value * (COST_CONFIG['cost_earliness'] * coef) * abs(diff)

    metrics = {
        'cost_material': cost_material,
        'cost_jit': cost_jit,
        'cost_total': cost_material + cost_jit,
        'utilization': utilization
    }

    return final_plates, scheduler.log, orders, metrics


def main():
    nest_path, sched_path, exp_dir = find_latest_experiment_models()
    if not nest_path: return

    nest_env = NestingSchedulingEnv()
    nest_env = ActionMasker(nest_env, mask_fn)
    sched_env = SchedulingEnv()
    sched_env = ActionMasker(sched_env, mask_fn)

    print("Loading RL Models...")
    nest_model = MaskablePPO.load(nest_path)
    sched_model = MaskablePPO.load(sched_path)
    nest_env.unwrapped.set_scheduling_partner(sched_model)

    # ==========================================
    # 🔥 核心：生成一套绝对固定的卷子！
    # ==========================================
    TEST_N, TEST_W, TEST_H = 60, 200, 200
    SEED = 1000

    print(f"\nGenerating Data with Seed {SEED}...")
    obs, _ = nest_env.reset(seed=SEED, options={"num_parts": TEST_N, "plate_size": (TEST_W, TEST_H)})

    raw_env = nest_env.unwrapped
    # 深拷贝题目数据，确保 Baseline 和 RL 使用同一张“卷子”
    parts_pool_raw = copy.deepcopy(raw_env.parts_pool)
    orders_raw = copy.deepcopy(raw_env.orders)
    for o in orders_raw.values():
        o['finished_time'] = 0.0

    order_colors = get_colors(len(orders_raw) + 5)

    # ==========================================
    # 比赛选手 1：Dual-Agent RL
    # ==========================================
    print("\nRunning [Player 1]: Dual-Agent RL ...")
    done = False
    while not done:
        mask = get_action_masks(nest_env)
        action, _ = nest_model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, _, info = nest_env.step(action)
        done = terminated

    rl_metrics = raw_env.cost_metrics
    plot_nesting(raw_env.history_plates, order_colors, save_dir=exp_dir, file_prefix="Compare_1_RL")
    plot_gantt(raw_env.scheduler_state_machine.log, raw_env.orders, order_colors, save_dir=exp_dir,
               file_prefix="Compare_1_RL")
    plot_jit_analysis(raw_env.orders, rl_metrics, save_dir=exp_dir, file_prefix="Compare_1_RL")

    # ==========================================
    # 比赛选手 2：传统启发式基线 (FFD + EDD)
    # ==========================================
    print("Running [Player 2]: Traditional Baseline (FFD+EDD) ...")
    base_plates, base_logs, base_orders, base_metrics = run_baseline_on_same_data(parts_pool_raw, orders_raw, TEST_W,
                                                                                  TEST_H)

    plot_nesting(base_plates, order_colors, save_dir=exp_dir, file_prefix="Compare_2_Baseline")
    plot_gantt(base_logs, base_orders, order_colors, save_dir=exp_dir, file_prefix="Compare_2_Baseline")
    plot_jit_analysis(base_orders, base_metrics, save_dir=exp_dir, file_prefix="Compare_2_Baseline")

    # ==========================================
    # 控制台直接打印终极对比结果
    # ==========================================
    print("\n" + "=" * 60)
    print("🏆 A/B TEST RESULTS ON EXACT SAME DATASET (Seed 42) 🏆")
    print("=" * 60)
    print(f"Metrics          | Dual-Agent RL       | Heuristic Baseline ")
    print(f"----------------------------------------------------------")
    print(f"Total Penalty    | {rl_metrics.get('cost_total', 0):>15.2f} | {base_metrics.get('cost_total', 0):>15.2f}")
    print(
        f"Material Waste   | {rl_metrics.get('cost_material', 0):>15.2f} | {base_metrics.get('cost_material', 0):>15.2f}")
    print(f"JIT Penalty      | {rl_metrics.get('cost_jit', 0):>15.2f} | {base_metrics.get('cost_jit', 0):>15.2f}")
    print(f"Avg Utilization  | {rl_metrics.get('utilization', 0):>15.1%} | {base_metrics.get('utilization', 0):>15.1%}")
    print("=" * 60)
    print(f"✅ Images successfully generated in: {exp_dir}")


if __name__ == "__main__":
    main()