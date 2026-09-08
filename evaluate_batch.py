
import os, glob, random, math, copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.sched_policy_loader import load_scheduling_policy

from envs.packing_envs import NestingSchedulingEnv
from envs.scheduling_env import SchedulingEnv
from integration.scheduling_terminal_reward import SchedulingTerminalRewardWrapper
from models.pointer_extractor import NestingModel, load_nesting_state_dict_strict
from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from core.cost import GlobalCostFunction
from core.processing import plate_processing_time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ── 与 train_dual.py 保持一致的常量 ─────────────────────────────────────────
N_ACTIONS_PER  = 6                                      # 2旋转 × 3策略


# ═══════════════════════════════════════════════════════════════════════════════
# 一、模型查找与加载
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_models(exp_root: str = "./experiments"):
    """
    从 experiments/ 目录找最新实验的模型文件。
    返回 (nest_pt_path, sched_zip_prefix, exp_dir)
      nest_pt_path  : nesting_joint_cN_final.pt 的完整路径
      sched_zip_prefix : scheduling_joint_cN（不含 .zip 后缀，MaskablePPO.load 需要这样）
      exp_dir       : 实验根目录
    """
    if not os.path.exists(exp_root):
        print(f"[ERROR] experiments 目录不存在: {exp_root}")
        return None, None, None

    exp_dirs = sorted(
        glob.glob(os.path.join(exp_root, "exp_*")),
        key=os.path.getctime, reverse=True
    )
    if not exp_dirs:
        print("[ERROR] 未找到任何实验目录")
        return None, None, None

    for exp_dir in exp_dirs:
        model_dir = os.path.join(exp_dir, "models")
        if not os.path.exists(model_dir):
            continue

        files = os.listdir(model_dir)

        # 找最大的 joint cycle（nesting 用 _final.pt，scheduling 用 .zip）
        nest_cycles  = set()
        sched_cycles = set()
        for f in files:
            if f.startswith("nesting_joint_c") and f.endswith("_final.pt"):
                try:
                    c = int(f.replace("nesting_joint_c", "").replace("_final.pt", ""))
                    nest_cycles.add(c)
                except ValueError:
                    pass
            if f.startswith("scheduling_joint_c") and f.endswith(".zip"):
                try:
                    c = int(f.replace("scheduling_joint_c", "").replace(".zip", ""))
                    sched_cycles.add(c)
                except ValueError:
                    pass

        valid = nest_cycles & sched_cycles
        if valid:
            c = max(valid)
            nest_pt   = os.path.join(model_dir, f"nesting_joint_c{c}_final.pt")
            sched_zip = os.path.join(model_dir, f"scheduling_joint_c{c}")
            print(f"[INFO] 实验: {os.path.basename(exp_dir)} | joint_cycle={c}")
            return nest_pt, sched_zip, exp_dir

        # fallback：phase1 预热模型
        if "nesting_phase1_final.pt" in files:
            nest_pt = os.path.join(model_dir, "nesting_phase1_final.pt")
            sched_zip = (os.path.join(model_dir, "scheduling_phase2")
                         if "scheduling_phase2.zip" in files else None)
            print("[WARN] 使用 Phase1 预热模型（未找到联合微调模型）")
            return nest_pt, sched_zip, exp_dir

    print("[ERROR] 未找到可用的模型文件")
    return None, None, None


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

    # ── 1. 找模型 ──────────────────────────────────────────────────────────────
    nest_pt, sched_zip, exp_dir = find_latest_models()
    if not nest_pt:
        return

    rl_env = NestingSchedulingEnv()
    nest_model  = load_nesting_model(nest_pt, rl_env.layout, device)
    sched_model = None
    if sched_zip and os.path.exists(sched_zip + ".zip"):
        sched_model = load_scheduling_policy(sched_zip, device=device)
        print(f"[INFO] Scheduling 模型加载完成: {os.path.basename(sched_zip)}")
    else:
        print("[WARN] 未找到 Scheduling 模型，排样将不感知调度信息")

    # ── 2. 固定测试数据 ─────────────────────────────────────────────────────────
    TEST_N, TEST_W, TEST_H = 60, 200, 200
    SEED = 1000

    print(f"\n生成测试数据: {TEST_N} 件 | 板材 {TEST_W}×{TEST_H} | Seed={SEED}")

    # ── 3. RL 推理 ─────────────────────────────────────────────────────────────
    print("\n[Player 1] Dual-Agent RL ...")
    rl_evaluator = SchedulingTerminalRewardWrapper(
        rl_env, evaluation_mode="edd")
    run_rl_episode(
        rl_evaluator, nest_model, device=device,
        seed=SEED,
        options={"num_parts": TEST_N, "plate_size": (TEST_W, TEST_H)},
    )
    rl_metrics = rl_env.cost_metrics

    # 保存测试数据快照（确保基线和 RL 使用完全相同的订单）
    parts_pool_snap = copy.deepcopy(rl_env.parts_pool)
    orders_snap     = copy.deepcopy(rl_env.orders)
    for o in orders_snap.values():
        o['finished_time'] = 0.0

    order_colors = _rand_colors(len(orders_snap) + 5)

    print("  生成 RL 可视化图表...")
    plot_nesting(rl_env.history_plates, order_colors,
                 save_dir=exp_dir, file_prefix="RL")
    plot_gantt(rl_env.scheduler_state_machine.log, rl_env.orders,
               order_colors, save_dir=exp_dir, file_prefix="RL")
    plot_jit_analysis(rl_env.orders, rl_metrics,
                      save_dir=exp_dir, file_prefix="RL")

    # ── 4. FFD+EDD 基线 ────────────────────────────────────────────────────────
    print("\n[Player 2] FFD+EDD Baseline ...")
    base_plates, base_logs, base_orders, base_metrics = run_ffd_edd_baseline(
        parts_pool_snap, orders_snap, TEST_W, TEST_H
    )
    print("  生成基线可视化图表...")
    plot_nesting(base_plates, order_colors,
                 save_dir=exp_dir, file_prefix="Baseline")
    plot_gantt(base_logs, base_orders, order_colors,
               save_dir=exp_dir, file_prefix="Baseline")
    plot_jit_analysis(base_orders, base_metrics,
                      save_dir=exp_dir, file_prefix="Baseline")

    # ── 5. 对比图 ──────────────────────────────────────────────────────────────
    plot_comparison_bar(rl_metrics, base_metrics, save_dir=exp_dir)

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
    print(f"  图表已保存至: {exp_dir}\n")


if __name__ == "__main__":
    main()
