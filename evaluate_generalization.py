
import os, random, math, copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from models.sched_policy_loader import load_scheduling_policy

from envs.packing_envs import NestingSchedulingEnv
from integration.scheduling_terminal_reward import (
    SchedulingTerminalRewardWrapper,
    make_dual_agent_terminal_wrapper,
)
from models.pointer_extractor import NestingModel, load_nesting_state_dict_strict
from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import TEST_SCENARIOS
from core.cost import GlobalCostFunction
from core.processing import plate_processing_time
from integration.evaluation_checkpoint import resolve_latest_phase3_pair
from integration.evaluation_results import (
    build_manifest, case_record, create_run_directory, formal_metrics,
    make_evaluation_case, make_run_id, write_evaluation_results,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

N_ACTIONS_PER  = 6
NUM_EPISODES   = 20   # 每场景跑多少局


# ═══════════════════════════════════════════════════════════════════════════════
# 一、模型查找与加载（与 evaluate_batch.py 完全相同）
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_models(exp_root: str = "./experiments"):
    pair = resolve_latest_phase3_pair(exp_root)
    return (str(pair.nesting_checkpoint), str(pair.scheduling_checkpoint),
            str(pair.experiment_dir))


def load_nesting_model(pt_path: str, layout, device: str = "cpu") -> NestingModel:
    model = NestingModel(
        part_feat_dim=layout.part_dim,
        state_feat_dim=layout.state_dim,
        embed_dim=128, n_heads=4, n_enc_layers=2,
        n_actions_per_part=N_ACTIONS_PER,
        max_parts=layout.max_parts,
        layout=layout,
    ).to(device)
    load_nesting_state_dict_strict(
        model, torch.load(pt_path, map_location=device))
    model.eval()
    print(f"[INFO] Nesting 模型: {os.path.basename(pt_path)}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 二、RL 推理
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_rl_episode(
    model: NestingModel,
    sched_model,
    device: str,
    seed: int,
    num_parts: int,
    plate_size: tuple,
    evaluation_mode: str = "policy",
    instance=None,
) -> NestingSchedulingEnv:
    """每局创建新环境，避免状态污染。"""
    base_env = NestingSchedulingEnv(observation_layout=model.layout)
    if evaluation_mode == "policy":
        env = make_dual_agent_terminal_wrapper(base_env, sched_model)
    elif evaluation_mode == "edd":
        env = SchedulingTerminalRewardWrapper(base_env, evaluation_mode="edd")
    else:
        raise ValueError("evaluation_mode must be 'policy' or 'edd'")
    options = ({"instance": copy.deepcopy(instance)} if instance is not None
               else {"num_parts": num_parts, "plate_size": plate_size})
    obs, _ = env.reset(seed=seed, options=options)

    done = False
    while not done:
        pf = torch.as_tensor(
            env.get_part_feats(), dtype=torch.float32, device=device
        ).unsqueeze(0)
        sf   = torch.as_tensor(env.get_state_feat(),
                               dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.as_tensor(env._get_action_mask(),
                               dtype=torch.bool, device=device).unsqueeze(0)
        logits, _ = model.forward_decision(pf, sf, mask)
        action = int(logits.argmax(dim=-1).item())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return base_env


# ═══════════════════════════════════════════════════════════════════════════════
# 三、FFD+EDD 基线（与 evaluate_batch.py 相同逻辑）
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline_episode(
    parts_pool: list,
    orders_raw: dict,
    plate_w: int,
    plate_h: int,
) -> dict:
    """返回成本指标字典。"""
    parts_pool = copy.deepcopy(parts_pool)
    orders     = copy.deepcopy(orders_raw)
    for o in orders.values():
        o['finished_time'] = 0.0

    # FFD
    sorted_parts  = sorted(parts_pool, key=lambda x: x['area'], reverse=True)
    active_plates = [PlateLayoutManager(width=plate_w, height=plate_h)]
    for part in sorted_parts:
        placed = False
        for plate in active_plates:
            ok, *_ = plate.place_part(part['w'], part['h'], part['order_id'], 1)
            if not ok:
                ok, *_ = plate.place_part(part['w'], part['h'], part['order_id'], 2)
            if ok:
                placed = True; break
        if not placed:
            np_ = PlateLayoutManager(width=plate_w, height=plate_h)
            np_.place_part(part['w'], part['h'], part['order_id'], 1)
            active_plates.append(np_)
    final_plates = [p for p in active_plates if p.placed_parts]

    # EDD
    scheduler = SchedulerStateMachine(num_machines=3)
    tasks = []
    for idx, plate in enumerate(final_plates):
        cut  = plate_processing_time(plate.placed_parts)
        oids = list(set(int(p[4]) for p in plate.placed_parts))
        due  = min((orders[o]['due_date'] for o in oids if o in orders), default=999.0)
        tasks.append({'cut': cut, 'due': due, 'idx': idx, 'oids': oids})
    tasks.sort(key=lambda t: t['due'])
    for task in tasks:
        m   = int(np.argmin(scheduler.get_state()))
        end = scheduler.execute_assignment(m, task['cut'], task['idx'])
        for oid in task['oids']:
            if oid in orders:
                orders[oid]['finished_time'] = max(orders[oid]['finished_time'], end)

    # 成本
    total_area  = sum(p['area'] for p in parts_pool)
    consumed    = len(final_plates) * plate_w * plate_h
    util        = total_area / consumed if consumed > 0 else 0.001
    cost = GlobalCostFunction().compute(
        final_plates, orders, parts_pool, plate_w, plate_h,
        {oid: order['finished_time'] for oid, order in orders.items()},
    )

    return {
        'utilization':   util,
        'cost_total':    cost['cost_total'],
        'cost_material': cost['cost_material'],
        'cost_jit':      cost['cost_jit'],
        'plate_count':   len(final_plates),
        'late_rate':     cost['late_count'] / max(1, len(orders)),
        'late_count':    cost['late_count'],
        'total_delay':   cost['total_delay'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 四、可视化
# ═══════════════════════════════════════════════════════════════════════════════

def _rand_colors(n):
    return ["#" + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))
            for _ in range(n)]


def plot_nesting_sample(plates, order_colors, save_dir, prefix):
    if not plates:
        return
    utils = [p.utilization for p in plates]
    cols  = min(4, len(plates))
    rows  = math.ceil(len(plates) / cols)
    fig, axes = plt.subplots(max(1, rows), cols,
                             figsize=(cols * 4, rows * 4))
    fig.suptitle(
        f"{prefix}  {len(plates)} plates | Util {np.mean(utils):.1%}",
        fontsize=11, y=0.99,
    )
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(plates):
            p = plates[i]
            ax.add_patch(patches.Rectangle(
                (0, 0), p.width, p.height, fc='white', ec='black', lw=1.5))
            for part in p.placed_parts:
                x, y, w, h, oid = part[:5]
                c = order_colors[int(oid) % len(order_colors)]
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h, fc=c, alpha=0.8, ec='black', lw=0.8))
            ax.set_xlim(0, p.width); ax.set_ylim(0, p.height)
            ax.set_title(f"P{i} {p.utilization:.1%}", fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_nesting.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_scenario_summary(rl_vals, base_vals, metric_labels,
                          scene_name, save_dir):
    """单场景 RL vs 基线柱状对比图。"""
    x = np.arange(len(metric_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - w/2, rl_vals,   w, label='RL',     color='#4dabf7', ec='black')
    b2 = ax.bar(x + w/2, base_vals, w, label='FFD+EDD', color='#ff6b6b', ec='black')
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h*1.02,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title(f"{scene_name}  RL vs FFD+EDD", fontsize=11)
    ax.legend(); ax.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    safe = scene_name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
    path = os.path.join(save_dir, f"{safe}_compare.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_overall_summary(df_rl, df_base, save_dir):
    """跨场景汇总对比图（利用率 + 总成本 + 准时率）。"""
    scenes = df_rl['Scenario'].tolist()
    x = np.arange(len(scenes))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Generalization Test: RL vs FFD+EDD (All Scenarios)", fontsize=13)

    for ax, (col, label, lower_better) in zip(axes, [
        ('Avg Util',  'Avg Utilization',  False),
        ('Avg Cost',  'Avg Total Cost',    True),
        ('Late Rate', 'Late Order Rate',   True),
    ]):
        rv = df_rl[col].values
        bv = df_base[col].values
        ax.bar(x - w/2, rv, w, label='RL',     color='#4dabf7', ec='black')
        ax.bar(x + w/2, bv, w, label='FFD+EDD', color='#ff6b6b', ec='black')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.split('.')[0].strip()[:12] for s in scenes],
            rotation=20, ha='right', fontsize=8,
        )
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis='y', ls='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "overall_comparison.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  [图] {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 五、主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    pair = resolve_latest_phase3_pair()
    nest_pt = str(pair.nesting_checkpoint)
    sched_zip = str(pair.scheduling_checkpoint)

    runtime_env = NestingSchedulingEnv()
    nest_model  = load_nesting_model(nest_pt, runtime_env.layout, device)
    sched_model = load_scheduling_policy(sched_zip, device=device)
    print(f"[INFO] Scheduling 模型: {os.path.basename(sched_zip)}")

    base_seed = 2000
    run_id = make_run_id("generalization", pair, base_seed)
    report_dir = str(create_run_directory("./evaluation_results", run_id))
    print(f"报告目录: {report_dir}\n")

    rl_rows   = []
    base_rows = []
    case_records = []
    next_case_id = 0

    for scenario in TEST_SCENARIOS:
        name      = scenario["name"]
        num_parts = scenario["num_parts"]
        plate_size = scenario["plate_size"]
        plate_w, plate_h = plate_size
        safe_name = (name.replace(" ", "_")
                     .replace(".", "").replace("(", "").replace(")", ""))

        print(f"场景: {name}  ({num_parts} 件, {plate_size})")
        scene_dir = os.path.join(report_dir, safe_name)
        os.makedirs(scene_dir, exist_ok=True)

        # 收集指标
        rl_util   = []; rl_cost = []; rl_mat = []; rl_jit = []
        rl_late   = []; rl_pc   = []
        base_util = []; base_cost=[]; base_mat=[]; base_jit=[]
        base_late = []; base_pc  = []

        last_rl_env = None

        for ep in tqdm(range(NUM_EPISODES), desc=f"  {safe_name[:28]}"):
            case = make_evaluation_case(
                next_case_id, base_seed, num_parts=num_parts,
                plate_size=plate_size, scenario=name,
                config={"scenario_index": TEST_SCENARIOS.index(scenario)})
            next_case_id += 1
            seed = case.case_seed

            # ── RL ──
            rl_env = run_rl_episode(
                nest_model, sched_model, device,
                seed, num_parts, plate_size,
                evaluation_mode="policy", instance=case.independent_instance(),
            )
            rl_formal = formal_metrics(
                rl_env.history_plates, rl_env.orders, rl_env.parts_pool,
                rl_env.plate_w, rl_env.plate_h)
            rl_util.append(rl_formal['Utilization'])
            rl_cost.append(rl_formal['Total_Cost'])
            rl_mat.append(rl_formal['Material_Cost'])
            rl_jit.append(rl_formal['JIT_Cost'])
            rl_pc.append(rl_formal['Plate_Count'])
            rl_late.append(rl_formal['Late_Count'] / max(1, len(rl_env.orders)))

            # ── 基线（用相同的 parts_pool 和 orders）──
            base_m = run_baseline_episode(
                case.independent_instance().parts,
                case.independent_instance().orders,
                plate_w, plate_h,
            )
            base_util.append(base_m['utilization'])
            base_cost.append(base_m['cost_total'])
            base_mat.append(base_m['cost_material'])
            base_jit.append(base_m['cost_jit'])
            base_pc.append(base_m['plate_count'])
            base_late.append(base_m['late_rate'])
            base_formal = {
                "Utilization": base_m["utilization"],
                "Plate_Count": base_m["plate_count"],
                "Late_Count": base_m["late_count"],
                "Total_Delay_Time": base_m["total_delay"],
                "JIT_Cost": base_m["cost_jit"],
                "Material_Cost": base_m["cost_material"],
                "Total_Cost": base_m["cost_total"],
            }
            case_records.extend([
                case_record(run_id, case, "Dual-Agent RL",
                            evaluation_mode="policy", pair=pair,
                            metrics=rl_formal),
                case_record(run_id, case, "FFD+EDD", evaluation_mode="edd",
                            pair=None, metrics=base_formal),
            ])

            last_rl_env = rl_env

        # ── 汇总 ──
        def agg(lst): return float(np.mean(lst))

        rl_row = {
            'Scenario': name, 'Parts': num_parts, 'Plate Size': str(plate_size),
            'Avg Util': agg(rl_util), 'Avg Cost': agg(rl_cost),
            'Mat Cost': agg(rl_mat),  'JIT Cost': agg(rl_jit),
            'Late Rate': agg(rl_late), 'Avg Plates': agg(rl_pc),
            'Std Cost': float(np.std(rl_cost)),
        }
        base_row = {
            'Scenario': name, 'Parts': num_parts, 'Plate Size': str(plate_size),
            'Avg Util': agg(base_util), 'Avg Cost': agg(base_cost),
            'Mat Cost': agg(base_mat),  'JIT Cost': agg(base_jit),
            'Late Rate': agg(base_late), 'Avg Plates': agg(base_pc),
            'Std Cost': float(np.std(base_cost)),
        }
        rl_rows.append(rl_row)
        base_rows.append(base_row)

        # 改进幅度
        cost_imp = (base_row['Avg Cost'] - rl_row['Avg Cost']) / max(1e-9, base_row['Avg Cost']) * 100
        late_imp = (base_row['Late Rate'] - rl_row['Late Rate']) / max(1e-9, base_row['Late Rate']) * 100
        print(f"  RL  Util={rl_row['Avg Util']:.1%}  Cost={rl_row['Avg Cost']:.1f}"
              f"  Late={rl_row['Late Rate']:.1%}")
        print(f"  Base Util={base_row['Avg Util']:.1%}  Cost={base_row['Avg Cost']:.1f}"
              f"  Late={base_row['Late Rate']:.1%}")
        print(f"  Improve → Cost {cost_imp:+.1f}%  Late {late_imp:+.1f}%")

        # ── 场景可视化 ──
        if last_rl_env:
            colors = _rand_colors(len(last_rl_env.orders) + 5)
            plot_nesting_sample(last_rl_env.history_plates, colors,
                                scene_dir, "RL_sample")
        plot_scenario_summary(
            rl_vals=[rl_row['Avg Util'], rl_row['Avg Cost']/1000,
                     rl_row['Late Rate'], rl_row['Avg Plates']],
            base_vals=[base_row['Avg Util'], base_row['Avg Cost']/1000,
                       base_row['Late Rate'], base_row['Avg Plates']],
            metric_labels=['Util', 'Cost(k)', 'Late Rate', 'Plates'],
            scene_name=name,
            save_dir=scene_dir,
        )

    # ── 保存逐 case source-of-truth、manifest 和派生汇总 ──
    manifest = build_manifest(
        run_id, "generalization", "policy", base_seed,
        len(TEST_SCENARIOS) * NUM_EPISODES, pair,
        {"scenarios": TEST_SCENARIOS, "num_machines": 3})
    write_evaluation_results(report_dir, manifest, case_records)

    # 下列 DataFrame 仅用于现有可视化/控制台展示。
    df_rl   = pd.DataFrame(rl_rows)
    df_base = pd.DataFrame(base_rows)

    # ── 跨场景总览图 ──
    plot_overall_summary(df_rl, df_base, report_dir)

    # ── 打印汇总表 ──
    print("\n" + "═" * 100)
    print("  GENERALIZATION REPORT — Dual-Agent RL")
    print("═" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df_rl.to_string(index=False, float_format="{:.4f}".format))

    print("\n" + "═" * 100)
    print("  GENERALIZATION REPORT — FFD+EDD Baseline")
    print("═" * 100)
    print(df_base.to_string(index=False, float_format="{:.4f}".format))

    # ── 打印改进幅度汇总 ──
    print("\n" + "═" * 70)
    print("  IMPROVEMENT SUMMARY  (RL vs FFD+EDD)")
    print("═" * 70)
    print(f"  {'Scenario':<30} {'Cost Imp':>10} {'Late Imp':>10} {'Util Imp':>10}")
    print("  " + "─" * 64)
    for rl, base in zip(rl_rows, base_rows):
        ci = (base['Avg Cost']  - rl['Avg Cost'])  / max(1e-9, base['Avg Cost'])  * 100
        li = (base['Late Rate'] - rl['Late Rate']) / max(1e-9, base['Late Rate']) * 100
        ui = (rl['Avg Util']   - base['Avg Util']) / max(1e-9, base['Avg Util']) * 100
        print(f"  {rl['Scenario']:<30} {ci:>+9.1f}% {li:>+9.1f}% {ui:>+9.1f}%")
    print("═" * 70)
    print(f"\n  CSV & 图表保存至: {report_dir}")


if __name__ == "__main__":
    main()
