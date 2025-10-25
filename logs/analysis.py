import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from datetime import datetime

# --- 1. 加载和处理数据 ---

def load_jsonl(file_path):
    """加载 JSONL 文件到 pandas DataFrame。"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"跳过一行错误数据: {e} in {file_path}")
    return pd.DataFrame(data)

# 加载数据
try:
    eval_df = load_jsonl('logs/stage2_1.0kg_eval.jsonl')
    metrics_df = load_jsonl('logs/stage2_1.0kg_metrics.jsonl')

    print(f"成功加载 eval.jsonl: {eval_df.shape[0]} 条记录")
    print(f"成功加载 metrics.jsonl: {metrics_df.shape[0]} 条记录")
    print("-" * 30)

    # --- 数据清理 (eval_df) ---
    eval_df['timestamp'] = pd.to_datetime(eval_df['timestamp'], unit='s')
    eval_df = eval_df.sort_values('timestamp')
    numeric_cols_eval = ['reward', 'loss', 'pos_rmse', 'att_rmse', 'final_error', 'force_mean']
    for col in numeric_cols_eval:
        eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce')
    eval_df.dropna(subset=numeric_cols_eval, inplace=True)
    if 'is_best' not in eval_df.columns:
        eval_df['is_best'] = False # 假设
    else:
        eval_df['is_best'] = eval_df['is_best'].astype(bool)

    # --- 数据清理 (metrics_df) ---
    metrics_df['step'] = pd.to_numeric(metrics_df['step'], errors='coerce')
    metrics_df['value'] = pd.to_numeric(metrics_df['value'], errors='coerce')
    metrics_df.dropna(subset=['step', 'value', 'tag'], inplace=True)
    metrics_df = metrics_df.sort_values('step')

    # 设置绘图风格
    sns.set_theme(style="whitegrid") # 移除 font="SimHei"
    # plt.rcParams['axes.unicode_minus'] = False # 这一行也可以删掉了

    # --- 2. 分析评估数据 (eval.jsonl) ---

    print("正在生成评估数据图表...")

    # 图 1: 评估奖励 (Reward) 趋势
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=eval_df, x='timestamp', y='reward', marker='o', label='评估奖励 (Reward)')
    # 标记最佳点
    best_runs_df = eval_df[eval_df['is_best'] == True]
    sns.scatterplot(data=best_runs_df, x='timestamp', y='reward', color='red', s=100, label='最佳表现 (is_best)', zorder=5)
    plt.title('评估奖励 (Reward) 随时间变化趋势')
    plt.xlabel('时间 (Timestamp)')
    plt.ylabel('奖励值')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eval_reward_trend.png')
    print("已生成: eval_reward_trend.png")

    # 图 2: 评估误差 (RMSE) 趋势
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=eval_df, x='timestamp', y='pos_rmse', marker='.', label='位置均方根误差 (pos_rmse)')
    sns.lineplot(data=eval_df, x='timestamp', y='att_rmse', marker='.', label='姿态均方根误差 (att_rmse)')
    plt.title('评估误差 (RMSE) 随时间变化趋势')
    plt.xlabel('时间 (Timestamp)')
    plt.ylabel('RMSE (越低越好)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eval_rmse_trend.png')
    print("已生成: eval_rmse_trend.png")

    # 总结最佳表现
    print("\n--- 最佳表现 (is_best=True) 总结 ---")
    if not best_runs_df.empty:
        best_summary = best_runs_df.sort_values('reward', ascending=False)
        print(best_summary[['timestamp', 'reward', 'pos_rmse', 'att_rmse', 'final_error', 'loss']])
    else:
        print("在 eval.jsonl 中未找到 'is_best'=True 的记录。")
    print("-" * 30)

    # --- 3. 分析训练指标 (metrics.jsonl) ---

    print("正在生成训练过程图表...")

    # 辅助绘图函数
    def plot_metric_by_tag(df, tag_name, title, filename, y_label='值'):
        metric_data = df[df['tag'] == tag_name].sort_values('step')
        if metric_data.empty:
            print(f"未找到指标: {tag_name}")
            return
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=metric_data, x='step', y='value')
        plt.title(title)
        plt.xlabel('训练步数 (Step)')
        plt.ylabel(y_label)
        # 使用科学计数法显示x轴
        plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(filename)
        print(f"已生成: {filename}")

    # 图 3: Actor 和 Critic 损失
    plt.figure(figsize=(14, 7))
    a_loss = metrics_df[metrics_df['tag'] == 'losses/a_loss']
    c_loss = metrics_df[metrics_df['tag'] == 'losses/c_loss']
    
    if not a_loss.empty:
        sns.lineplot(data=a_loss, x='step', y='value', label='Actor 损失 (a_loss)')
    if not c_loss.empty:
        sns.lineplot(data=c_loss, x='step', y='value', label='Critic 损失 (c_loss)')
        
    plt.title('训练损失 (Actor & Critic) 随步数变化')
    plt.xlabel('训练步数 (Step)')
    plt.ylabel('损失值')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('training_losses.png')
    print("已生成: training_losses.png")

    # 图 4: 熵 (Entropy) 趋势
    plot_metric_by_tag(metrics_df, 'losses/entropy', 
                       '训练熵 (Entropy) 随步数变化', 
                       'training_entropy.png', 
                       y_label='熵值')

    # 图 5: 训练奖励 (rewards/iter) 趋势
    plot_metric_by_tag(metrics_df, 'rewards/iter', 
                       '训练迭代奖励 (rewards/iter) 随步数变化', 
                       'training_reward.png', 
                       y_label='奖励值')

except FileNotFoundError as e:
    print(f"错误: 文件未找到 {e.filename}")
except Exception as e:
    print(f"分析过程中发生错误: {e}")