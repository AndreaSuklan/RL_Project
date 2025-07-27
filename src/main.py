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

def train(algorithm, seed=None, model = "nn", degree=3, verbose=0):
    """
    Trains a new agent by calling the appropriate creation function.
    """
    print(f"--- Starting training for {algorithm.upper()} with {model} model and seed {seed} ---")

    
    if model == "poly":
        run_name = f"{algorithm}_{model}_d{degree}_{seed}"
    else:
        run_name = f"{algorithm}_{model}_{seed}"

    env = HillClimbEnv(enable_coins=False)

    if seed is not None:
        env.reset(seed=seed)

    if algorithm == 'ppo':
        if model != "nn":
            print(f"Warning: PPO does not support {model} model, using 'nn' instead.")
            model = "nn"
        agent = PPO(
            env,
            model=model,
            buffer_size=2048, 
            gamma=0.99, 
            gae_lambda=0.95, 
            lr=3e-4, 
            clip_epsilon=0.2, 
            n_epochs=10, 
            batch_size=64, 
        )

        log_data = agent.learn(total_episodes=20, verbose=verbose)
        
    elif algorithm == 'dqn':
        agent = DQN(
            env,
            model=model,
            gamma=0.99, 
            buffer_size=10000, 
            lr=0.001, 
            epsilon=0.1, 
            batch_size=64)
        log_data = agent.learn(total_episodes=20, verbose=1)

    elif algorithm == 'sarsa':
        agent = SARSA(
            env, 
            gamma=0.99, 
            lr=0.001, 
            epsilon=0.1
            )
        log_data = agent.learn(total_episodes=20, verbose=1)
        
    else:
        print(f"Error: Unknown algorithm '{algorithm}'")
        env.close()
        return
    
    # Save the trained model
    if model == "poly":
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_{degree}_hill_climb.zip")
    else:
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_hill_climb.zip")

    agent.save(model_path)

    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(os.path.join(LOGS_DIR, f"{run_name}.csv"), index=False)
        print(f"Logs saved to {LOGS_DIR}")
    
    env.close()


def visualize(algorithm, model = "nn", degree=3, verbose=0):
    """
    Visualizes a pre-trained agent.
    """
    
    model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_{degree}_hill_climb.zip")
        
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_3_hill_climb.zip")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_hill_climb.zip")

    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_hill_climb.zip")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return

    print(f"--- Loading model: {model_path} ---")
    env = HillClimbEnv(enable_coins=False, render_mode="human")

    if algorithm == 'ppo':
        m = PPO.create_model(env=env, model=model)
        model = PPO.load(model_path, env=env, model=m)
    elif algorithm == 'dqn':
        m = DQN.create_model(env, model, degree=degree)
        model = DQN.load(model_path, env=env, model=m)
    elif algorithm == 'sarsa':
        m = SARSA.create_model(env, model, degree=degree)
        model = SARSA.load(model_path, env=env, model=m)
    else:
        print(f"Error: Unknown algorithm '{algorithm}'")
        env.close()
        return

    print("--- Starting Visualization ---")
    obs, info = env.reset()
    try:
        while True:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\n--- Visualization stopped by user ---")
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Visualize a Hill Climb Agent.")
    parser.add_argument("action", choices=["train", "visualize"], help="Action to perform.")
    parser.add_argument("algorithm", choices=["ppo", "dqn", "sarsa"], help="Algorithm to use.")
    parser.add_argument("--model", choices=["linear", "nn", "poly"], default="nn", help="Model type for SARSA (default: nn).")
    parser.add_argument("--degree", type=int, default=3, help="Degree for polynomial model (default: 3).")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for the run.")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level: 0=none, 1=summary, 2=debug")

    args = parser.parse_args()

    if args.action == "train":
        train(args.algorithm, seed=args.seed, model=args.model, degree=args.degree, verbose=args.verbose)
    elif args.action == "visualize":
        visualize(args.algorithm, model=args.model, degree=args.degree, verbose=args.verbose)

