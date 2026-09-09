import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import TEST_SCENARIOS
from core.cost import GlobalCostFunction
from core.processing import plate_processing_time
from integration.evaluation_results import (
    build_manifest, case_record, create_run_directory, make_evaluation_case,
    make_run_id, write_evaluation_results,
)


def run_heuristic_baseline():
    print("\n" + "=" * 80)
    print("🚀 开始执行传统工业基线测试 (FFD + EDD)")
    print("=" * 80)

    base_seed = 2000
    run_id = make_run_id("baseline", None, base_seed)
    eval_report_dir = str(create_run_directory("./evaluation_results", run_id))
    summary_data = []
    case_records = []
    next_case_id = 0

    # 初始化环境仅用于生成相同的随机订单数据
    env = NestingSchedulingEnv()

    for scenario in TEST_SCENARIOS:
        scene_name = scenario["name"]
        num_parts = scenario["num_parts"]
        plate_size = scenario["plate_size"]
        plate_w, plate_h = plate_size

        print(f"\n🧪 Baseline Testing: {scene_name} | Parts: {num_parts} | Size: {plate_size}")

        metrics = {
            "utilization": [], "total_cost": [], "mat_cost": [], "jit_cost": [],
            "late_rate": [], "plate_count": []
        }

        NUM_EPISODES = 20

        for i in tqdm(range(NUM_EPISODES), desc=f"Scenario: {scene_name}"):

            # ==========================================================
            # 🔥 绝对控制变量：强行锁死全局 Numpy 和 Python 随机种子！
            # 确保生成的订单长、宽、交期与 RL 模型做的一模一样！
            # ==========================================================
            case = make_evaluation_case(
                next_case_id, base_seed, num_parts=num_parts,
                plate_size=plate_size, scenario=scene_name,
                config={"scenario_index": TEST_SCENARIOS.index(scenario)})
            next_case_id += 1
            seed_val = case.case_seed

            # 1. 重置环境获取初始数据
            env.reset(seed=seed_val, options={"instance": case.independent_instance()})
            parts_pool = copy.deepcopy(env.unwrapped.parts_pool)
            orders = copy.deepcopy(env.unwrapped.orders)

            # ==========================================
            # 阶段一：按照面积从大到小贪心排样 (FFD)
            # ==========================================
            sorted_parts = sorted(parts_pool, key=lambda x: x['area'], reverse=True)
            active_plates = [PlateLayoutManager(width=plate_w, height=plate_h)]

            for part in sorted_parts:
                placed = False
                for plate in active_plates:
                    ok, sx, sy, sw, sh, _ = plate.place_part(part['w'], part['h'], part['order_id'], 1)
                    if not ok:
                        ok, sx, sy, sw, sh, _ = plate.place_part(part['w'], part['h'], part['order_id'], 2)
                    if ok:
                        placed = True
                        break

                if not placed:
                    new_plate = PlateLayoutManager(width=plate_w, height=plate_h)
                    new_plate.place_part(part['w'], part['h'], part['order_id'], 1)
                    active_plates.append(new_plate)

            final_plates = [p for p in active_plates if len(p.placed_parts) > 0]

            # ==========================================
            # 阶段二：传统调度 (按照板材最小交期 EDD 派工)
            # ==========================================
            scheduler = SchedulerStateMachine(num_machines=3)
            tasks = []
            for idx, plate in enumerate(final_plates):
                cut_time = plate_processing_time(plate.placed_parts)
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

            # ==========================================
            # 阶段三：计算各项经济成本 (🔥 缩进已修复，必须在循环最外层！)
            # ==========================================
            total_part_area = sum([p['area'] for p in parts_pool])
            consumed_area = len(final_plates) * (plate_w * plate_h)
            utilization = total_part_area / consumed_area if consumed_area > 0 else 0.001

            cost = GlobalCostFunction().compute(
                final_plates, orders, parts_pool, plate_w, plate_h,
                {oid: order['finished_time'] for oid, order in orders.items()},
            )

            # 收集该局数据
            metrics["utilization"].append(utilization)
            metrics["total_cost"].append(cost['cost_total'])
            metrics["mat_cost"].append(cost['cost_material'])
            metrics["jit_cost"].append(cost['cost_jit'])
            metrics["late_rate"].append(cost['late_count'] / len(orders) if orders else 0)
            metrics["plate_count"].append(len(final_plates))
            formal = {
                "Utilization": cost["utilization"],
                "Plate_Count": cost["plate_count"],
                "Late_Count": cost["late_count"],
                "Total_Delay_Time": cost["total_delay"],
                "JIT_Cost": cost["cost_jit"],
                "Material_Cost": cost["cost_material"],
                "Total_Cost": cost["cost_total"],
            }
            case_records.append(case_record(
                run_id, case, "FFD+EDD", evaluation_mode="edd",
                pair=None, metrics=formal))

        # 汇总该场景数据
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

    df = pd.DataFrame(summary_data)
    manifest = build_manifest(
        run_id, "baseline", "edd", base_seed, len(case_records), None,
        {"scenarios": TEST_SCENARIOS, "num_machines": 3})
    write_evaluation_results(eval_report_dir, manifest, case_records)

    print("\n" + "=" * 100)
    print("📊 传统基线测试报告 (BASELINE EVALUATION REPORT)")
    print("=" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 100)
    print(f"✅ 基线数据已保存在: {eval_report_dir}")


if __name__ == "__main__":
    run_heuristic_baseline()
