import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import numpy as np

# 配置字体以支持中文显示 (可选)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def get_colors(n):
    """生成 n 种高对比度的随机颜色"""
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]


def plot_nesting(plates, orders_info=None):
    """
    绘制所有板材的排样结果。

    参数:
    - plates: List[PlateLayoutManager], 包含 placed_parts 数据
    - orders_info: dict, 用于获取订单数量来生成颜色池 (可选)
    """
    total_plates = len(plates)
    if total_plates == 0:
        print("⚠️ 没有板材数据，跳过绘制排样图。")
        return None, None

    # 布局设置：每行画 4 张板
    cols = 4
    rows = math.ceil(total_plates / cols)

    # 动态调整画布高度
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    fig.suptitle(f'Nesting Result: Total {total_plates} Plates', fontsize=16, y=0.99)

    # 处理 axes 维度
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 生成颜色池 (基于订单ID区分颜色)
    num_orders = len(orders_info) if orders_info else 100
    order_colors = get_colors(num_orders + 5)

    # 计算平均利用率
    avg_util = np.mean([p.utilization for p in plates])
    print(f"📊 排样统计：共 {total_plates} 张板材，平均利用率: {avg_util:.2%}")

    for i in range(len(axes)):
        ax = axes[i]

        if i < total_plates:
            plate = plates[i]

            # 1. 绘制板材轮廓
            ax.add_patch(patches.Rectangle((0, 0), plate.width, plate.height,
                                           linewidth=2, edgecolor='black', facecolor='white'))

            # 2. 绘制零件
            for part_data in plate.placed_parts:
                # 兼容性解包：只取前6个数据 (x, y, w, h, oid, rotated)
                # 有些旧代码可能没有 rotated，这里做个防护
                if len(part_data) >= 6:
                    x, y, w, h, oid, is_rotated = part_data[:6]
                else:
                    x, y, w, h, oid = part_data[:5]
                    is_rotated = False

                # 获取颜色
                color = order_colors[int(oid) % len(order_colors)]

                # 绘制矩形
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.85
                )
                ax.add_patch(rect)

                # 标记文字 (太小的零件不标记)
                if w > 0.05 and h > 0.05:
                    label = f"{oid}"
                    if is_rotated: label += "R"
                    ax.text(x + w / 2, y + h / 2, label,
                            ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            # 设置坐标轴
            ax.set_xlim(0, plate.width)
            ax.set_ylim(0, plate.height)
            ax.set_title(f"Plate {i} (Util: {plate.utilization:.1%})", fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')  # 不显示坐标轴刻度，更美观

        else:
            # 隐藏多余的子图
            ax.axis('off')

    plt.tight_layout()
    return fig, axes