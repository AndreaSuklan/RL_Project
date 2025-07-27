import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
LOGS_DIR = "./logs"
PLOTS_DIR = "./plots"
ROLLING_WINDOW = 25

def plot_learning_curve(df, title, filename):
    """Plots the smoothed reward curve for all algorithms against episodes."""
    plt.figure(figsize=(12, 7))
    df["smoothed_reward"] = df.groupby(['algorithm', 'model'])['reward'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    
    sns.lineplot(data=df, x='episode', y='smoothed_reward', hue='algorithm', style="model",  errorbar='sd', legend=True)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel(f'Mean Episode Reward (Rolling Avg. of {ROLLING_WINDOW})', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm & Model')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()

def plot_metric(df, metric_col, title, filename, y_log_scale=False):
    """Plots a generic metric curve (e.g., loss, entropy) against episodes."""
    plt.figure(figsize=(12, 7))
    metric_df = df.dropna(subset=[metric_col])
    df[metric_col] = df.groupby(['algorithm', 'model'])[metric_col].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )

    sns.lineplot(data=metric_df, x='episode', y=metric_col, hue='algorithm', style = "model", errorbar='sd', legend=True)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    if y_log_scale:
        plt.yscale('log')
        plt.ylabel(f"{metric_col.replace('_', ' ').title()} (Log Scale)")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm & Model')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()

def plot_ppo_buffer_comparison(df, title, filename):
    """Plots a comparison of PPO runs with different buffer sizes against episodes."""
    ppo_df = df[df['algorithm'] == 'PPO'].copy()
    if ppo_df.empty or 'buffer_size' not in ppo_df.columns or ppo_df['buffer_size'].isnull().all():
        print("No PPO data with buffer sizes found to plot.")
        return

    plt.figure(figsize=(12, 7))
    
    ppo_df['buffer_size'] = ppo_df['buffer_size'].astype('category')
    
    ppo_df['run_id'] = ppo_df['buffer_size'].astype(str) + '_' + ppo_df['seed'].astype(str)
    ppo_df['smoothed_reward'] = ppo_df.groupby('run_id')['reward'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    # Use 'episode' for the x-axis
    sns.lineplot(data=ppo_df, x='episode', y='smoothed_reward', hue='buffer_size', errorbar='sd', legend=True, palette='viridis')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel(f'Mean Episode Reward (Rolling Avg. of {ROLLING_WINDOW})', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Buffer Size')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()


def plot_final_performance_bar_chart(df, title, filename):
    """Plots a bar chart of the final performance."""
    # Create a unique identifier for each distinct run
    df['run_id'] = df['algorithm'] + '_' + df['model'] + '_' + df['seed'].astype(str)
    
    # Define a function to get the last 10% of data for each run
    def get_last_10_percent(x):
        # Ensure we take at least one row, even for very short runs
        n_rows = max(1, int(len(x) * 0.1))
        return x.tail(n_rows)

    # Apply the function to each run
    final_data = df.groupby('run_id').apply(get_last_10_percent).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    # Let seaborn perform the aggregation and calculate the standard deviation for error bars
    sns.barplot(data=final_data, x='algorithm', y='reward', hue='model', capsize=0.1, errorbar='sd')
    
    plt.title(title, fontsize=16)
    plt.ylabel('Mean Final Episode Reward', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")
    plt.close()


def main():
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

                # --- NEW: Handle different log formats (PPO vs others) ---
                if 'mean_reward' in df.columns:
                    # This is a PPO-style log. Rename reward column and filter.
                    df = df.rename(columns={'mean_reward': 'reward'})                
                # Parse algorithm and seed from filename (e.g., 'ppo_0.csv')
                parts = filename.replace('.csv', '').split('_')
                df['algorithm'] = parts[0].upper()
                df['model'] = parts[1]
                if len(parts) > 3:
                    df["degree"] = None
                    df['seed'] = int(parts[3])
                    # Keep only the rows where a reward was actually logged.
                    # These rows represent the end of a training cycle where episodes were completed.
                    df = df.dropna(subset=['reward']).copy()
                else:
                    # This is a DQN or SARSA-style log (assume one row per episode).
                    df['episode'] = df.index + 1

                # --- Filename parsing logic ---
                parts = filename.replace('.csv', '').split('_')
                algorithm = parts[0]
                model = parts[1]
                df['degree'] = None

                if algorithm == 'ppo' and model == 'nn' and len(parts) == 4:
                    df['buffer_size'] = int(parts[2])
                    df['seed'] = int(parts[3])
                elif model == 'poly' and len(parts) == 4:
                    df['degree'] = int(parts[2])
                    df['seed'] = int(parts[3])
                elif len(parts) == 3:
                    df['seed'] = int(parts[2])
                else:
                    print(f"Skipping file with unrecognized format: {filename}")
                    continue
                
                all_data.append(df)
            except Exception as e:
                print(f"Could not process {log_path}: {e}")

    if not all_data:
        print("No valid log data found to plot.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)

    # --- Generate All Plots (This section is unchanged) ---
    plot_learning_curve(full_df, "Learning Curves", "reward_curve.png")
    plot_metric(full_df, 'value_loss', 'Value Loss over Time', 'value_loss_curve.png', y_log_scale=True)
    
    ppo_df_for_metrics = full_df[full_df['algorithm'] == 'PPO']
    if not ppo_df_for_metrics.empty:
        plot_metric(ppo_df_for_metrics, 'entropy', 'PPO Policy Entropy', 'ppo_entropy_curve.png')

    plot_final_performance_bar_chart(full_df, "Final Performance Comparison", "final_performance_bar_chart.png")
    
    plot_ppo_buffer_comparison(full_df, "PPO Performance by Buffer Size", "ppo_buffer_comparison.png")


if __name__ == '__main__':
    main()
