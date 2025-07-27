import argparse
import os
import pandas as pd
from environment import HillClimbEnv
from ppo import PPO
from dqn import DQN
from sarsa import SARSA

MODELS_DIR = "./models"
LOGS_DIR = "./logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def train(algorithm, seed=None, model="nn", degree=3, verbose=0, total_episodes=100,  buffer_size=2048):
    """
    Trains a new agent by calling the appropriate creation function.
    """
    print(f"--- Starting training for {algorithm.upper()} with {model} model and seed {seed}")

    if model == "poly":
        run_name = f"{algorithm}_{model}_{degree}_{seed}"
    elif algorithm == "ppo":
        run_name = f"{algorithm}_{model}_{buffer_size}_{seed}"
    else:
        run_name = f"{algorithm}_{model}_{seed}"

    env = HillClimbEnv(enable_coins=False, seed=seed)

    if seed is not None:
        env.reset()

    if algorithm == 'ppo':
        agent = PPO(
            env,
            model=model,
            buffer_size=buffer_size,
            degree=degree,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3e-4,
            clip_epsilon=0.2,
            n_epochs=10,
            batch_size=64,
        )
        log_data = agent.learn(total_episodes=total_episodes, verbose=verbose)

    elif algorithm == 'dqn':
        agent = DQN(
            env,
            model=model,
            degree=degree,
            gamma=0.99,
            buffer_size=10000,
            lr=0.001,
            epsilon=0.1,
            batch_size=64
            ),
        log_data = agent.learn(total_episodes=total_episodes, verbose=verbose)

    elif algorithm == 'sarsa':
        agent = SARSA(
            env,
            model=model,
            gamma=0.99,
            lr=0.001,
            epsilon=0.1,
            degree=degree,
            seed=seed
        )
        log_data = agent.learn(total_episodes=total_episodes, verbose=verbose)

    else:
        print(f"Error: Unknown algorithm '{algorithm}'")
        env.close()
        return

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, f"{run_name}.zip")
    agent.save(model_path)

    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(os.path.join(LOGS_DIR, f"{run_name}.csv"), index=False)
        print(f"Logs saved to {LOGS_DIR}")

    env.close()

def visualize(model_path, seed=None):
    """
    Visualizes a pre-trained agent from a specific file path.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    # Parse algorithm and model info from the filename
    try:
        filename = os.path.basename(model_path)
        parts = filename.replace('.zip', '').split('_')
        algorithm = parts[0]
        model_type = parts[1]
        degree = 3  # Default degree
        if model_type == 'poly':
            degree = int(parts[2])
    except (IndexError, ValueError):
        print(f"Error: Could not parse details from filename '{filename}'.")
        print("Expected format like 'algorithm_model_... .zip' (e.g., 'dqn_nn_0.zip').")
        return

    print(f"--- Loading model from: {model_path}")
    print(f"--- Detected Algorithm: {algorithm.upper()}, Model: {model_type}")
    env = HillClimbEnv(enable_coins=False, render_mode="human", seed=seed)

    # Use parsed info to load the correct model
    if algorithm == 'ppo':
        m = PPO.create_model(env, model_type, degree=degree)
        model = PPO.load(model_path, env=env, model=m)
    elif algorithm == 'dqn':
        m = DQN.create_model(env, model_type, degree=degree)
        model = DQN.load(model_path, env=env, model=m)
    elif algorithm == 'sarsa':
        m = SARSA.create_model(env, model_type, degree=degree)
        model = SARSA.load(model_path, env=env, model=m)
    else:
        print(f"Error: Unknown algorithm '{algorithm}' parsed from filename.")
        env.close()
        return

    print("--- Starting Visualization")
    obs, info = env.reset()
    try:
        while True:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\n--- Visualization stopped by user")
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Visualize a Hill Climb Agent.")
    parser.add_argument("action", choices=["train", "visualize"], help="Action to perform.")

    # Arguments for training
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument("algorithm", nargs='?', default=None, choices=["ppo", "dqn", "sarsa"], help="Algorithm to use for training.")
    train_group.add_argument("--model", choices=["linear", "nn", "poly"], default="nn", help="Model type for the agent (default: nn).")
    train_group.add_argument("--degree", type=int, default=3, help="Degree for polynomial model (default: 3).")
    train_group.add_argument("--buffer_size", type=int, default=2048, help="Buffer size for the PPO algorithm (default: 2048).")
    train_group.add_argument("-e", "--episodes", type=int, default=100, help="Total episodes for training (default: 100).")
    
    # Argument for visualization
    vis_group = parser.add_argument_group('Visualization Arguments')
    vis_group.add_argument("--path", type=str, help="Path to the model file for visualization (e.g., models/ppo_nn_0.zip).")

    # General arguments
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for the run.")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level: 0=none, 1=summary, 2=debug")

    args = parser.parse_args()

    if args.action == "train":
        if not args.algorithm:
            parser.error("The 'algorithm' argument is required for the 'train' action.")
        train(args.algorithm, seed=args.seed, model=args.model, degree=args.degree, verbose=args.verbose, total_episodes=args.episodes, buffer_size=args.buffer_size)
    elif args.action == "visualize":
        if not args.path:
            parser.error("The --path argument is required for the 'visualize' action.")
        visualize(model_path=args.path, seed=args.seed)