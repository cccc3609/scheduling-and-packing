
import os, random, math, copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.sched_policy_loader import load_scheduling_policy

from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_terminal_reward import make_dual_agent_terminal_wrapper
from models.pointer_extractor import NestingModel, load_nesting_state_dict_strict
from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from core.cost import GlobalCostFunction
from core.processing import plate_processing_time
from integration.evaluation_checkpoint import resolve_latest_phase3_pair
from integration.evaluation_results import (
    build_manifest, case_record, create_run_directory, formal_metrics,
    make_evaluation_case, make_run_id, write_evaluation_results,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ── 与 train_dual.py 保持一致的常量 ─────────────────────────────────────────
N_ACTIONS_PER  = 6                                      # 2旋转 × 3策略


# ═══════════════════════════════════════════════════════════════════════════════
# 一、模型查找与加载
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_models(exp_root: str = "./experiments"):
    pair = resolve_latest_phase3_pair(exp_root)
    return (str(pair.nesting_checkpoint), str(pair.scheduling_checkpoint),
            str(pair.experiment_dir))


def load_nesting_model(pt_path: str, layout, device: str = "cpu") -> NestingModel:
    """加载 NestingModel 权重。"""
    model = NestingModel(
        part_feat_dim=layout.part_dim,
        state_feat_dim=layout.state_dim,
        embed_dim=128,
        n_heads=4,
        n_enc_layers=2,
        n_actions_per_part=N_ACTIONS_PER,
        max_parts=layout.max_parts,
        layout=layout,
    ).to(device)
    state_dict = torch.load(pt_path, map_location=device)
    load_nesting_state_dict_strict(model, state_dict)
    model.eval()
    print(f"[INFO] Nesting 模型加载完成: {os.path.basename(pt_path)}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 二、RL 推理（Encoder-Decoder 解耦）
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_rl_episode(
    env: NestingSchedulingEnv,
    model: NestingModel,
    device: str = "cpu",
    seed: int = None,
    options: dict = None,
) -> NestingSchedulingEnv:
    """
    用 NestingModel 跑完一局排样。
    Dynamic part tokens are freshly encoded for every decision.
    返回完成后的环境（含 cost_metrics、history_plates、orders 等）。
    """
    obs, _info = env.reset(seed=seed, options=options or {})
    if model.layout != env.unwrapped.layout:
        raise ValueError("Evaluation model and environment layouts must match")

    done = False
    while not done:
        pf = torch.as_tensor(
            env.get_part_feats(), dtype=torch.float32, device=device
        ).unsqueeze(0)
        sf = torch.as_tensor(
            env.get_state_feat(), dtype=torch.float32, device=device
        ).unsqueeze(0)                     # [1, 57]

        mask = torch.as_tensor(
            env._get_action_mask(), dtype=torch.bool, device=device
        ).unsqueeze(0)                     # [1, 720]

        logits, _ = model.forward_decision(pf, sf, mask)
        action = int(logits.argmax(dim=-1).item())

        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

    return env


# ═══════════════════════════════════════════════════════════════════════════════
# 三、FFD+EDD 传统基线
# ═══════════════════════════════════════════════════════════════════════════════

def run_ffd_edd_baseline(
    parts_pool: list,
    orders_raw: dict,
    plate_w: int,
    plate_h: int,
):
    """
    FFD 排样 + EDD 调度。
    在 deepcopy 的数据上运行，不修改原始数据。
    返回 (final_plates, scheduler_log, orders, metrics)
    """
    parts_pool = copy.deepcopy(parts_pool)
    orders     = copy.deepcopy(orders_raw)
    for o in orders.values():
        o['finished_time'] = 0.0

    # ── FFD 排样 ──
    sorted_parts  = sorted(parts_pool, key=lambda x: x['area'], reverse=True)
    active_plates = [PlateLayoutManager(width=plate_w, height=plate_h)]

    for part in sorted_parts:
        placed = False
        for plate in active_plates:
            ok, *_ = plate.place_part(part['w'], part['h'], part['order_id'], 1)
            if not ok:
                ok, *_ = plate.place_part(part['w'], part['h'], part['order_id'], 2)
            if ok:
                placed = True
                break
        if not placed:
            new_p = PlateLayoutManager(width=plate_w, height=plate_h)
            new_p.place_part(part['w'], part['h'], part['order_id'], 1)
            active_plates.append(new_p)

    final_plates = [p for p in active_plates if p.placed_parts]

    # ── EDD 调度 ──
    scheduler = SchedulerStateMachine(num_machines=3)
    tasks     = []

    for idx, plate in enumerate(final_plates):
        cut  = plate_processing_time(plate.placed_parts)
        oids = list(set(int(p[4]) for p in plate.placed_parts))
        due  = min((orders[o]['due_date'] for o in oids if o in orders),
                   default=999.0)
        tasks.append({'cut': cut, 'due': due, 'idx': idx, 'oids': oids})

    tasks.sort(key=lambda t: t['due'])

    for task in tasks:
        m   = int(np.argmin(scheduler.get_state()))
        end = scheduler.execute_assignment(m, task['cut'], task['idx'])
        for oid in task['oids']:
            if oid in orders:
                orders[oid]['finished_time'] = max(
                    orders[oid]['finished_time'], end)

    # ── 成本计算 ──
    total_area  = sum(p['area'] for p in parts_pool)
    consumed    = len(final_plates) * plate_w * plate_h
    utilization = total_area / consumed if consumed > 0 else 0.001
    cost = GlobalCostFunction().compute(
        final_plates, orders, parts_pool, plate_w, plate_h,
        {oid: order['finished_time'] for oid, order in orders.items()},
    )

    metrics = {
        'cost_material': cost['cost_material'],
        'cost_jit':      cost['cost_jit'],
        'cost_total':    cost['cost_total'],
        'utilization':   utilization,
        'plate_count':   len(final_plates),
        'late_count':    cost['late_count'],
        'total_delay':   cost['total_delay'],
    }
    return final_plates, scheduler.log, orders, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 四、可视化
# ═══════════════════════════════════════════════════════════════════════════════

def _rand_colors(n: int):
    return ["#" + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))
            for _ in range(n)]


def plot_nesting(plates, order_colors, save_dir: str, file_prefix: str):
    if not plates:
        return
    utils = [p.utilization for p in plates]
    cols  = 4
    rows  = math.ceil(len(plates) / cols)
    fig, axes = plt.subplots(max(1, rows), cols,
                             figsize=(16, 4 * max(1, rows)))
    fig.suptitle(
        f"Nesting Result [{file_prefix}]  "
        f"{len(plates)} plates | Avg Util {np.mean(utils):.1%}",
        y=0.99, fontsize=13,
    )
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(plates):
            p = plates[i]
            ax.add_patch(patches.Rectangle(
                (0, 0), p.width, p.height, fc='white', ec='black', lw=2))
            for part in p.placed_parts:
                x, y, w, h, oid = part[:5]
                is_rot = part[5] if len(part) > 5 else False
                c = order_colors[int(oid) % len(order_colors)]
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h, fc=c, alpha=0.85, ec='black', lw=1))
                if w > p.width * 0.05 and h > p.height * 0.05:
                    ax.text(x + w / 2, y + h / 2,
                            f"{oid}" + ("R" if is_rot else ""),
                            color='white', ha='center', va='center',
                            fontsize=8, fontweight='bold')
            ax.set_xlim(0, p.width)
            ax.set_ylim(0, p.height)
            ax.set_title(f"P{i}  {p.utilization:.1%}", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, f"{file_prefix}_nesting.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [图] {os.path.basename(path)}")


def plot_gantt(logs, orders, order_colors, save_dir: str, file_prefix: str):
    if not logs:
        return
    machine_ids = sorted(set(t['machine_id'] for t in logs))
    max_time    = max(t['end'] for t in logs)

    fig, ax = plt.subplots(figsize=(16, 6))
    for t in logs:
        mid = t['machine_id']
        dur = t['end'] - t['start']
        ax.broken_barh([(t['start'], dur)], (mid * 10, 8),
                       fc='#87CEEB', ec='black', lw=0.8)
        ax.text(t['start'] + dur / 2, mid * 10 + 4,
                f"P{t['plate_idx']}", ha='center', va='center', fontsize=7)

    for oid, info in sorted(orders.items()):
        due = info['due_date']
        fin = info['finished_time']
        c   = order_colors[int(oid) % len(order_colors)]
        ax.axvline(x=due, color=c, ls='--', alpha=0.55, lw=1.2)
        ax.text(due, 32 + (int(oid) % 3) * 3,
                f"O{oid}", color=c, rotation=90, fontsize=8)
        if fin > due:
            yp = -2 - (int(oid) % 5) * 2
            ax.hlines(yp, due, fin, colors='red', lw=2)
            ax.text(fin, yp, "Late", color='red', fontsize=6, va='center')

    ax.set_yticks([i * 10 + 4 for i in machine_ids])
    ax.set_yticklabels([f"M{i}" for i in machine_ids])
    ax.set_xlabel("Time (min)")
    ax.set_title(f"Gantt Chart [{file_prefix}]")
    ax.grid(True, axis='x', ls=':', alpha=0.3)
    ax.set_xlim(0, max_time * 1.1)
    ax.set_ylim(-15, len(machine_ids) * 10 + 8)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{file_prefix}_gantt.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [图] {os.path.basename(path)}")


def plot_jit_analysis(orders, metrics, save_dir: str, file_prefix: str):
    if not orders:
        return
    ids, diffs, cols = [], [], []
    for oid, info in sorted(orders.items()):
        d = info['finished_time'] - info['due_date']
        ids.append(f"O{oid}")
        diffs.append(d)
        cols.append('#FF6B6B' if d > 0 else '#51CF66')

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Cost & JIT Analysis [{file_prefix}]", fontsize=16, y=0.97)

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.bar(ids, diffs, color=cols, edgecolor='black', linewidth=0.5)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_ylabel("Completion - Due (min)")
    ax1.set_title("Per-Order Deviation  (Red = Late, Green = Early)", fontsize=12)
    ax1.grid(axis='y', ls='--', alpha=0.4)
    if len(ids) > 20:
        ax1.set_xticks(range(0, len(ids), max(1, len(ids) // 20)))
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    ax2 = plt.subplot2grid((2, 2), (1, 0))
    cm = metrics.get('cost_material', 0)
    cj = metrics.get('cost_jit', 0)
    if cm + cj > 0:
        ax2.pie([cm, cj],
                labels=[f"Material\n{cm:.0f}", f"JIT Penalty\n{cj:.0f}"],
                autopct='%1.1f%%',
                colors=['#4dabf7', '#ff6b6b'],
                startangle=90)
        ax2.set_title(f"Total Cost: {metrics.get('cost_total', 0):.1f}", fontsize=12)

    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis('off')
    late = sum(1 for d in diffs if d > 0.1)
    summary = (
        f"Performance Summary\n"
        f"{'─'*28}\n"
        f"Late Orders     : {late:>4} / {len(orders)}\n"
        f"Total Cost      : {metrics.get('cost_total', 0):>10.2f}\n"
        f"Material Waste  : {metrics.get('cost_material', 0):>10.2f}\n"
        f"JIT Penalty     : {metrics.get('cost_jit', 0):>10.2f}\n"
        f"Avg Utilization : {metrics.get('utilization', 0):>9.1%}\n"
        f"Plates Used     : {metrics.get('plate_count', 0):>4}\n"
    )
    ax3.text(0.05, 0.5, summary, fontsize=13, family='monospace', va='center',
             transform=ax3.transAxes)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{file_prefix}_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [图] {os.path.basename(path)}")


def plot_comparison_bar(rl_m, base_m, save_dir: str):
    """对比柱状图：两个方案在各指标上的对比。"""
    metrics_info = [
        ('cost_total',    'Total Cost',    True),
        ('cost_material', 'Material Waste', True),
        ('cost_jit',      'JIT Penalty',   True),
        ('utilization',   'Utilization',   False),
    ]
    labels = [m[1] for m in metrics_info]
    rl_vals   = []
    base_vals = []

    for key, _, lower_better in metrics_info:
        rv = rl_m.get(key, 0)
        bv = base_m.get(key, 0)
        if key == 'utilization':
            rv *= 100; bv *= 100
        rl_vals.append(rv)
        base_vals.append(bv)

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, rl_vals,   w, label='Dual-Agent RL', color='#4dabf7', edgecolor='black')
    b2 = ax.bar(x + w / 2, base_vals, w, label='FFD+EDD',       color='#ff6b6b', edgecolor='black')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("RL vs FFD+EDD Comparison", fontsize=14)
    ax.legend()
    ax.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [图] {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 五、主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── 1. 严格解析 Patch 7 同轮 checkpoint pair ─────────────────────────────
    pair = resolve_latest_phase3_pair()
    nest_pt = str(pair.nesting_checkpoint)
    sched_zip = str(pair.scheduling_checkpoint)

    rl_env = NestingSchedulingEnv()
    nest_model  = load_nesting_model(nest_pt, rl_env.layout, device)
    sched_model = load_scheduling_policy(sched_zip, device=device)
    print(f"[INFO] Scheduling 模型加载完成: {os.path.basename(sched_zip)}")

    # ── 2. 固定测试数据 ─────────────────────────────────────────────────────────
    TEST_N, TEST_W, TEST_H = 60, 200, 200
    SEED = 1000
    case = make_evaluation_case(
        0, SEED, num_parts=TEST_N, plate_size=(TEST_W, TEST_H),
        scenario="batch", config={"evaluation": "batch"})
    run_id = make_run_id("batch", pair, SEED)
    run_dir = create_run_directory("./evaluation_results", run_id)

    print(f"\n生成测试数据: {TEST_N} 件 | 板材 {TEST_W}×{TEST_H} | Seed={SEED}")

    # ── 3. RL 推理 ─────────────────────────────────────────────────────────────
    print("\n[Player 1] Dual-Agent RL ...")
    rl_evaluator = make_dual_agent_terminal_wrapper(rl_env, sched_model)
    run_rl_episode(
        rl_evaluator, nest_model, device=device,
        seed=SEED,
        options={"instance": case.independent_instance()},
    )
    rl_formal = formal_metrics(
        rl_env.history_plates, rl_env.orders, rl_env.parts_pool,
        rl_env.plate_w, rl_env.plate_h)
    rl_metrics = {
        'cost_material': rl_formal['Material_Cost'],
        'cost_jit': rl_formal['JIT_Cost'],
        'cost_total': rl_formal['Total_Cost'],
        'utilization': rl_formal['Utilization'],
        'plate_count': rl_formal['Plate_Count'],
    }

    # 保存测试数据快照（确保基线和 RL 使用完全相同的订单）
    baseline_instance = case.independent_instance()
    parts_pool_snap = baseline_instance.parts
    orders_snap = baseline_instance.orders

    order_colors = _rand_colors(len(orders_snap) + 5)

    print("  生成 RL 可视化图表...")
    plot_nesting(rl_env.history_plates, order_colors,
                 save_dir=str(run_dir), file_prefix="RL")
    plot_gantt(rl_env.scheduler_state_machine.log, rl_env.orders,
               order_colors, save_dir=str(run_dir), file_prefix="RL")
    plot_jit_analysis(rl_env.orders, rl_metrics,
                      save_dir=str(run_dir), file_prefix="RL")

    # ── 4. FFD+EDD 基线 ────────────────────────────────────────────────────────
    print("\n[Player 2] FFD+EDD Baseline ...")
    base_plates, base_logs, base_orders, base_metrics = run_ffd_edd_baseline(
        parts_pool_snap, orders_snap, TEST_W, TEST_H
    )
    print("  生成基线可视化图表...")
    plot_nesting(base_plates, order_colors,
                 save_dir=str(run_dir), file_prefix="Baseline")
    plot_gantt(base_logs, base_orders, order_colors,
               save_dir=str(run_dir), file_prefix="Baseline")
    plot_jit_analysis(base_orders, base_metrics,
                      save_dir=str(run_dir), file_prefix="Baseline")

    # ── 5. 对比图 ──────────────────────────────────────────────────────────────
    plot_comparison_bar(rl_metrics, base_metrics, save_dir=str(run_dir))
    baseline_formal = {
        "Utilization": base_metrics["utilization"],
        "Plate_Count": base_metrics["plate_count"],
        "Late_Count": base_metrics["late_count"],
        "Total_Delay_Time": base_metrics["total_delay"],
        "JIT_Cost": base_metrics["cost_jit"],
        "Material_Cost": base_metrics["cost_material"],
        "Total_Cost": base_metrics["cost_total"],
    }
    records = [
        case_record(run_id, case, "Dual-Agent RL", evaluation_mode="policy",
                    pair=pair, metrics=rl_formal),
        case_record(run_id, case, "FFD+EDD", evaluation_mode="edd",
                    pair=None, metrics=baseline_formal),
    ]
    manifest = build_manifest(
        run_id, "batch", "policy", SEED, 1, pair,
        {"num_parts": TEST_N, "plate_size": [TEST_W, TEST_H],
         "num_machines": case.instance.num_machines})
    write_evaluation_results(run_dir, manifest, records)

    # ── 6. 打印对比表 ──────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print(f"  A/B TEST RESULTS  (Seed={SEED}, {TEST_N} parts, {TEST_W}×{TEST_H})")
    print("═" * 68)
    print(f"  {'Metric':<22} {'Dual-Agent RL':>14} {'FFD+EDD':>14} {'Improve':>9}")
    print("  " + "─" * 64)

    def row(label, key, pct=False, lower_better=True):
        rv = rl_metrics.get(key, 0)
        bv = base_metrics.get(key, 0)
        if lower_better:
            imp = (bv - rv) / max(1e-9, abs(bv)) * 100
        else:
            imp = (rv - bv) / max(1e-9, abs(bv)) * 100
        sign = "↓" if lower_better else "↑"
        if pct:
            print(f"  {label:<22} {rv:>13.1%} {bv:>13.1%} {imp:>+8.1f}%")
        else:
            print(f"  {label:<22} {rv:>14.2f} {bv:>14.2f} {imp:>+8.1f}%")

    row("Total Cost",       "cost_total",    lower_better=True)
    row("Material Waste",   "cost_material", lower_better=True)
    row("JIT Penalty",      "cost_jit",      lower_better=True)
    row("Avg Utilization",  "utilization",   pct=True, lower_better=False)

    rl_pc   = int(rl_metrics.get('plate_count', 0))
    base_pc = int(base_metrics.get('plate_count', 0))
    print(f"  {'Plates Used':<22} {rl_pc:>14} {base_pc:>14}")
    print("═" * 68)
    print(f"  结果已保存至: {run_dir}\n")


if __name__ == "__main__":
    main()
