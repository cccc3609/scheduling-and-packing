import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

# 引入项目模块
from envs.packing_envs import NestingSchedulingEnv
from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import TEST_SCENARIOS, COST_CONFIG


def run_heuristic_baseline():
    print("\n" + "=" * 80)
    print("FFD + EDD)")
    print("=" * 80)

    # 准备输出目录
    eval_report_dir = "./experiments/baseline_report"
    os.makedirs(eval_report_dir, exist_ok=True)
    summary_data = []

    # 初始化环境仅用于生成相同的随机订单数据
    env = NestingSchedulingEnv()

    for scenario in TEST_SCENARIOS:
        scene_name = scenario["name"]
        num_parts = scenario["num_parts"]
        plate_size = scenario["plate_size"]
        plate_w, plate_h = plate_size

        print(f"\ Baseline Testing: {scene_name} | Parts: {num_parts} | Size: {plate_size}")

        metrics = {
            "utilization": [], "total_cost": [], "mat_cost": [], "jit_cost": [],
            "late_rate": [], "plate_count": []
        }

        # 同样跑 20 局取平均
        NUM_EPISODES = 20

        for i in tqdm(range(NUM_EPISODES), desc=f"Scenario: {scene_name}"):
            # 1. 重置环境获取初始数据 (控制变量法)
            env.reset(seed=2000 + i, options={"num_parts": num_parts, "plate_size": plate_size})
            parts_pool = copy.deepcopy(env.unwrapped.parts_pool)
            orders = copy.deepcopy(env.unwrapped.orders)


            # 按照面积从大到小贪心排样

            sorted_parts = sorted(parts_pool, key=lambda x: x['area'], reverse=True)
            active_plates = [PlateLayoutManager(width=plate_w, height=plate_h)]

            for part in sorted_parts:
                placed = False
                for plate in active_plates:
                    # 尝试 Skyline 策略 (ID: 1)
                    ok, sx, sy, sw, sh, _ = plate.place_part(part['w'], part['h'], part['order_id'], 1)
                    if not ok:
                        # 尝试 MaxRects 策略 (ID: 2)
                        ok, sx, sy, sw, sh, _ = plate.place_part(part['w'], part['h'], part['order_id'], 2)
                    if ok:
                        placed = True
                        break

                # 如果所有当前板材都排不下，开一张新板
                if not placed:
                    new_plate = PlateLayoutManager(width=plate_w, height=plate_h)
                    new_plate.place_part(part['w'], part['h'], part['order_id'], 1)
                    active_plates.append(new_plate)

            final_plates = [p for p in active_plates if len(p.placed_parts) > 0]

            # 阶段二：传统调度 (按照板材最小交期 EDD 派工)

            scheduler = SchedulerStateMachine(num_machines=3)
            tasks = []
            speed = COST_CONFIG['cutting_speed']

            # 提取每张板材的特征用于调度
            for idx, plate in enumerate(final_plates):
                cut_time = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / speed
                oids = list(set([int(p[4]) for p in plate.placed_parts]))
                min_due = min([orders[o]['due_date'] for o in oids]) if oids else 999.0
                tasks.append({'cut': cut_time, 'due': min_due, 'idx': idx, 'oids': oids, 'done': False})

            # 按照交期最紧迫优先 (EDD)
            tasks.sort(key=lambda x: x['due'])

            # 贪心派工到最早空闲的机器
            for task in tasks:
                mach_times = scheduler.get_state()
                best_mach_idx = np.argmin(mach_times)  # 找最早有空的机器
                end_t = scheduler.execute_assignment(best_mach_idx, task['cut'], task['idx'])

                # 记录订单完工时间
                for oid in task['oids']:
                    orders[oid]['finished_time'] = max(orders[oid]['finished_time'], end_t)

                    # ==========================================
                    # 阶段三：计算各项经济成本 (逻辑与 RL 保持绝对对齐)
                    # ==========================================

                    # 1. 物理总面积与利用率统计
                    total_part_area = sum([p['area'] for p in parts_pool])
                    consumed_area = len(final_plates) * (plate_w * plate_h)
                    utilization = total_part_area / consumed_area if consumed_area > 0 else 0.001

                    # 🔥 修复点：材料成本 = 单价系数 * 浪费掉的面积 (对齐 RL 环境)
                    wasted_area = max(0.0, consumed_area - total_part_area)
                    cost_material = wasted_area * COST_CONFIG['cost_material']

                    cost_jit = 0.0
                    late_count = 0

                    # 同样获取 episode_time_scale 计算梯形窗口
                    total_perimeter = sum([2 * (p['w'] + p['h']) for p in parts_pool])
                    episode_time_scale = max(10.0, total_perimeter / speed)

                    for oid, order in orders.items():
                        due = order['due_date']
                        fin = order['finished_time']
                        order_parts = [p for p in parts_pool if p['order_id'] == oid]
                        order_value = sum([p['area'] for p in order_parts])
                        order_perim = sum([2 * (p['w'] + p['h']) for p in order_parts])
                        order_proc_time = max(1.0, order_perim / speed)

                        diff = fin - due
                        abs_diff = abs(diff)

                        if diff > 0: late_count += 1

                        ratio = abs_diff / order_proc_time
                        R_FREE, R_FULL = 0.025, 0.075
                        if ratio <= R_FREE:
                            coef = 0.0
                        elif ratio <= R_FULL:
                            coef = (ratio - R_FREE) / (R_FULL - R_FREE)
                        else:
                            coef = 1.0

                        if diff > 0:
                            cost_jit += order_value * (COST_CONFIG['cost_tardiness'] * coef) * abs_diff
                        else:
                            cost_jit += order_value * (COST_CONFIG['cost_earliness'] * coef) * abs_diff

                    # 这里的 total_cost 现在是纯纯的“罚款总和”（浪费的材料钱 + 迟到的违约金）
                    total_cost = cost_material + cost_jit
            total_part_area = sum([p['area'] for p in parts_pool])
            utilization = total_part_area / consumed_area if consumed_area else 0

            # 收集该局数据
            metrics["utilization"].append(utilization)
            metrics["total_cost"].append(total_cost)
            metrics["mat_cost"].append(cost_material)
            metrics["jit_cost"].append(cost_jit)
            metrics["late_rate"].append(late_count / len(orders) if orders else 0)
            metrics["plate_count"].append(len(final_plates))

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

    # 生成并保存汇总报表
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(eval_report_dir, "baseline_summary_report.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print("📊 传统基线测试报告 (BASELINE EVALUATION REPORT)")
    print("=" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 100)
    print(f"✅ 基线数据已保存在: {csv_path} (请将其与强化学习测试结果对比！)")


if __name__ == "__main__":
    run_heuristic_baseline()