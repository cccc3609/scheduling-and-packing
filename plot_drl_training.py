import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 配置风格
sns.set_theme(style="darkgrid")
plt.rcParams['axes.unicode_minus'] = False


def read_tb_log(log_dir):
    # 递归查找最新的日志文件
    search_path = os.path.join(log_dir, "**", "*tfevents*")
    files = glob.glob(search_path, recursive=True)
    if not files:
        print("❌ 未找到日志文件，请检查路径！")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 读取日志: {latest_file}")

    ea = EventAccumulator(latest_file)
    ea.Reload()

    tags = ea.Tags()['scalars']
    data = {}

    # 关注的 DRL 核心指标
    keys = [
        'rollout/ep_rew_mean',  # 总分
        'train/value_loss',  # 价值误差
        'train/explained_variance',  # 价值解释度 (关键!)
        'train/entropy_loss',  # 探索熵
        'train/policy_gradient_loss',
        'train/approx_kl'  # 策略差异
    ]

    for k in keys:
        if k in tags:
            events = ea.Scalars(k)
            data[k] = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])

    return data


def plot_drl_metrics(data):
    if not data: return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Deep Reinforcement Learning (PPO) Training Diagnostics', fontsize=20, y=0.98)

    # 1. Reward (越高越好)
    if 'rollout/ep_rew_mean' in data:
        sns.lineplot(data=data['rollout/ep_rew_mean'], x='step', y='value', ax=axes[0, 0], color='tab:blue')
        axes[0, 0].set_title('Mean Episode Reward (Higher is Better)')

    # 2. Value Loss (越低越好，但需稳定)
    if 'train/value_loss' in data:
        sns.lineplot(data=data['train/value_loss'], x='step', y='value', ax=axes[0, 1], color='tab:orange')
        axes[0, 1].set_title('Critic Value Loss (Should Stabilize)')
        axes[0, 1].set_yscale('log')  # 使用对数坐标，因为初期可能很大

    # 3. Explained Variance (越接近 1 越好)
    if 'train/explained_variance' in data:
        sns.lineplot(data=data['train/explained_variance'], x='step', y='value', ax=axes[0, 2], color='tab:green')
        axes[0, 2].set_title('Explained Variance (Target: > 0.5)')
        axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.5)  # 0 是及格线
        axes[0, 2].set_ylim(-1, 1.1)

    # 4. Entropy (缓慢下降)
    if 'train/entropy_loss' in data:
        sns.lineplot(data=data['train/entropy_loss'], x='step', y='value', ax=axes[1, 0], color='tab:purple')
        axes[1, 0].set_title('Entropy Loss (Exploration vs Exploitation)')

    # 5. Policy Gradient Loss (震荡)
    if 'train/policy_gradient_loss' in data:
        sns.lineplot(data=data['train/policy_gradient_loss'], x='step', y='value', ax=axes[1, 1], color='tab:red')
        axes[1, 1].set_title('Actor Loss (Gradient Updates)')

    # 6. KL Divergence (越小越稳定)
    if 'train/approx_kl' in data:
        sns.lineplot(data=data['train/approx_kl'], x='step', y='value', ax=axes[1, 2], color='tab:brown')
        axes[1, 2].set_title('Approx KL Divergence (Stability)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("drl_training_diagnostics.png", dpi=300)
    print("✅ 诊断图表已保存至: drl_training_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    # 指定你要查看哪个智能体的日志
    # 例如查看排样智能体：
    LOG_DIR = "./logs_dual/nesting/"
    # 或者查看调度智能体：
    # LOG_DIR = "./logs_dual/scheduling/"

    data = read_tb_log(LOG_DIR)
    plot_drl_metrics(data)