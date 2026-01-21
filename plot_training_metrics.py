import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# === 配置区域 ===
# 字体设置 (防止中文乱码)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 🔴 这里填你日志里的确切名字 (根据 custom_callbacks.py)
KEY_UTIL = 'Physical/Raw_Avg_Util'
KEY_JIT = 'Physical/Raw_Avg_JIT'


def read_tb(log_dir):
    # 1. 自动寻找最新的日志文件
    files = glob.glob(f"{log_dir}/**/*tfevents*", recursive=True)
    if not files:
        print(f"❌ 错误：在 {log_dir} 下没找到日志文件")
        return None

    latest_file = max(files, key=os.path.getctime)
    print(f"📂 读取日志: {latest_file}")

    # 2. 加载数据
    ea = EventAccumulator(latest_file, size_guidance={'scalars': 0})
    ea.Reload()

    # 3. 检查并提取
    tags = ea.Tags()['scalars']
    data = {}

    if KEY_UTIL in tags:
        ev = ea.Scalars(KEY_UTIL)
        data['util'] = pd.DataFrame([(e.step, e.value) for e in ev], columns=['step', 'value'])
        print(f"✅ 成功读取利用率数据 ({len(data['util'])} 条)")
    else:
        print(f"❌ 未找到指标: {KEY_UTIL}")
        print("   -> 现有指标: ", tags)

    if KEY_JIT in tags:
        ev = ea.Scalars(KEY_JIT)
        data['jit'] = pd.DataFrame([(e.step, e.value) for e in ev], columns=['step', 'value'])
        print(f"✅ 成功读取JIT成本数据 ({len(data['jit'])} 条)")
    else:
        print(f"❌ 未找到指标: {KEY_JIT}")

    return data


def plot(data):
    if not data or 'util' not in data or 'jit' not in data:
        print("❌ 数据不全，无法绘图")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.title("训练进化过程：利用率 vs JIT成本\n(Evolution: Utilization vs JIT Cost)", fontsize=16)

    # --- 左轴：利用率 (绿色) ---
    color_util = 'tab:green'
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Average Utilization (0~1)', color=color_util, fontsize=12)

    df_util = data['util']
    # 绘制平滑曲线 (Window=50)
    ax1.plot(df_util['step'], df_util['value'].rolling(50, min_periods=1).mean(),
             color=color_util, lw=2.5, label='Avg Utilization')
    ax1.tick_params(axis='y', labelcolor=color_util)
    ax1.set_ylim(0.4, 1.0)  # 设定利用率显示的合理范围
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右轴：JIT 成本 (红色) ---
    ax2 = ax1.twinx()
    color_jit = 'tab:red'
    ax2.set_ylabel('Avg JIT Cost (Time Units)', color=color_jit, fontsize=12)

    df_jit = data['jit']
    ax2.plot(df_jit['step'], df_jit['value'].rolling(50, min_periods=1).mean(),
             color=color_jit, lw=2.5, label='Avg JIT Cost')
    ax2.tick_params(axis='y', labelcolor=color_jit)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    save_name = "result_analysis_final.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ 图表已保存: {os.path.abspath(save_name)}")

    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    # 自动定位实验路径
    LOG_ROOT = "./logs_dual/nesting/"

    if not os.path.exists(LOG_ROOT):
        # 尝试去 experiments 找
        exp_dirs = glob.glob("./experiments/exp_*")
        if exp_dirs:
            latest_exp = max(exp_dirs, key=os.path.getctime)
            LOG_ROOT = os.path.join(latest_exp, "logs/nesting/")
            print(f"⚠️ 自动切换路径: {LOG_ROOT}")

    plot(read_tb(LOG_ROOT))