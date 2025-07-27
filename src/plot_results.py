import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

LOGS_DIR = "./logs"
PLOTS_DIR = "./plots"
ROLLING_WINDOW = 25

def plot_learning_curve(df, title, filename, alg = "all"):

    if alg == "all":
        plt.figure(figsize=(12, 7))
        df["smoothed_reward"] = df.groupby(['algorithm'])['reward'].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        sns.lineplot(data=df, x='episode', y='smoothed_reward', hue='algorithm', legend=True)
    else:
        plt.figure(figsize=(12, 7))
        df["smoothed_reward"] = df.groupby(["model"])['reward'].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        sns.lineplot(data=df, x='episode', y='smoothed_reward', hue='model', legend=True)

    
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel(f'Mean Episode Reward (Rolling Avg. of {ROLLING_WINDOW})', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm and Model')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()

def plot_metric(df, metric_col, title, filename, y_log_scale=False, alg="all"):

    plt.figure(figsize=(12, 7))
    metric_df = df.dropna(subset=[metric_col])
    if alg == "all":
        df[metric_col] = df.groupby(['algorithm'])[metric_col].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        sns.lineplot(data=metric_df, x='episode', y=metric_col, hue='algorithm', legend=True)
    else:
        df[metric_col] = df.groupby([ "model"])[metric_col].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

        sns.lineplot(data=metric_df, x='episode', y=metric_col, hue='model', legend=True)

    
    plt.title(title, fontsize=16)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    if y_log_scale:
        plt.yscale('log')
        plt.ylabel(f"{metric_col.replace('_', ' ').title()} (Log Scale)")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()

def plot_final_performance_bar_chart(df, title, filename):
    """Plots a bar chart of the final performance."""
    final_rewards = df.groupby(['algorithm'])['reward'].apply(
        lambda x: x.tail(int(len(x) * 0.1)).mean()
    )
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=final_rewards.index.get_level_values('algorithm'), y=final_rewards.values, capsize=0.1, errorbar='sd')
    
    plt.title(title, fontsize=16)
    plt.ylabel('Mean Final Episode Reward', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()


def main(alg="all"):
    """Main function to load all data from CSV files and generate all plots."""
    if not os.path.exists(LOGS_DIR):
        print(f"Error: Logs directory '{LOGS_DIR}' not found.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    all_data = []

    for filename in os.listdir(LOGS_DIR):
        if filename.endswith(".csv"):
            log_path = os.path.join(LOGS_DIR, filename)
            try:
                df = pd.read_csv(log_path)

                if 'mean_reward' in df.columns:
                    df = df.rename(columns={'mean_reward': 'reward'})
                
                parts = filename.replace('.csv', '').split('_')
                if parts[0] == alg or alg == "all":
                    df['algorithm'] = parts[0].upper()
                    if alg != "all":
                        df['model'] = parts[1]
                    elif parts[1] == "nn":
                        df['model'] = parts[1]
                    if len(parts) > 3:
                        df["degree"] = None
                        df['seed'] = int(parts[3])
                    else:
                        df['seed'] = int(parts[2])
                        df["degreee"] = None
                else:
                    continue
                    
                
                all_data.append(df)
            except Exception as e:
                print(f"Could not process {log_path}: {e}")

    if not all_data:
        print("No valid log data found to plot.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)

    plot_learning_curve(full_df, "Learning Curves", f"{alg}_reward_curve.png", alg=alg)
    plot_metric(full_df, 'value_loss', 'Value Loss over Time', f'{alg}_value_loss_curve.png', y_log_scale=True, alg=alg)
    
    ppo_df = full_df[full_df['algorithm'] == 'PPO']
    if not ppo_df.empty:
        plot_metric(ppo_df, 'entropy', 'PPO Policy Entropy', 'ppo_entropy_curve.png')

    if alg == "all":
        plot_final_performance_bar_chart(full_df, "Final Performance Comparison", "final_performance_bar_chart.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Visualize a Hill Climb Agent.")
    parser.add_argument('--alg', type=str, default='all', choices=['all', 'ppo', 'dqn', 'sarsa'], help="Algorithm to visualize (default: all)")
    args = parser.parse_args()

    main(alg = args.alg)
