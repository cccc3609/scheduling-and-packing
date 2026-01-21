import matplotlib.pyplot as plt
import random

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def get_colors(n):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]


def plot_gantt(scheduler_log, orders_info):
    """
    绘制生产调度甘特图，并标记订单交货期。

    参数:
    - scheduler_log: List[dict], 包含 {'machine_id', 'start', 'end', 'plate_idx'}
    - orders_info: dict, 包含 {order_id: {'due_date':..., 'finished_time':...}}
    """
    if not scheduler_log:
        print("⚠️ 调度日志为空，跳过甘特图绘制。")
        return None, None

    # 获取机器数量
    machine_ids = sorted(list(set(t['machine_id'] for t in scheduler_log)))
    num_machines = len(machine_ids)

    fig, ax = plt.subplots(figsize=(15, 6))

    # 计算最大时间跨度用于设置坐标轴
    max_time = max([t['end'] for t in scheduler_log]) if scheduler_log else 100

    # --- 1. 绘制机器加工任务 ---
    for task in scheduler_log:
        mid = task['machine_id']
        start = task['start']
        end = task['end']
        duration = end - start
        plate_idx = task['plate_idx']

        # 绘制条形
        ax.broken_barh([(start, duration)], (mid * 10, 8),
                       facecolors='#87CEEB', edgecolor='black', linewidth=1)

        # 标记板材ID
        ax.text(start + duration / 2, mid * 10 + 4, f"P{plate_idx}",
                ha='center', va='center', fontsize=8)

    # --- 2. 绘制订单交期 (垂直虚线) ---
    if orders_info:
        # 生成颜色池
        order_colors = get_colors(len(orders_info) + 5)
        sorted_orders = sorted(orders_info.items(), key=lambda x: x[0])

        for oid, info in sorted_orders:
            due = info['due_date']
            finish = info['finished_time']
            color = order_colors[oid % len(order_colors)]

            # 画交期线
            ax.axvline(x=due, color=color, linestyle='--', alpha=0.7, linewidth=1.5)

            # 标记订单号 (错开高度防止重叠)
            label_y = 32 + (oid % 3) * 3
            ax.text(due, label_y, f"Ord{oid}", color=color, fontsize=9, rotation=90)

            # 如果拖期，画一条红线连接 交期 和 完成时间
            if finish > due:
                # 在底部画红色实线表示拖期长度
                line_y = -2 - (oid % 5)
                ax.hlines(y=line_y, xmin=due, xmax=finish, colors='red', linewidth=2)
                ax.text(finish, line_y, "Late", color='red', fontsize=6, va='center')

    # --- 3. 图表格式设置 ---
    ax.set_yticks([i * 10 + 4 for i in machine_ids])
    ax.set_yticklabels([f'Machine {i}' for i in machine_ids])
    ax.set_xlabel('Time')
    ax.set_title('Production Schedule & Order Due Dates')
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)

    # 设置X轴范围，稍微留点余地
    ax.set_xlim(0, max(max_time * 1.1, 50))
    # 设置Y轴范围以容纳下方的拖期红线
    ax.set_ylim(-10, num_machines * 10 + 5)

    plt.tight_layout()
    return fig, ax