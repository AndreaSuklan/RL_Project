# main.py

import argparse
import os
from environment import HillClimbEnv
import ppo
import dqn


# --- Configuration ---
MODELS_DIR = "./models"
LOGS_DIR = "./logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def train(algorithm, timesteps):
    """
    Trains a new agent by calling the appropriate creation function.
    """
    print(f"--- Starting training for {algorithm.upper()} ---")
    
    env = HillClimbEnv()
    log_dir = os.path.join(LOGS_DIR, f"{algorithm}_logs")

    # --- Select and create the agent ---
    if algorithm == 'ppo':
        agent = ppo.create_agent(env, log_dir)
    elif algorithm == 'dqn':
        agent = dqn.create_agent(env, log_dir)
    else:
        print(f"Error: Unknown algorithm '{algorithm}'")
        env.close()
        return

    # Train the agent
    agent.learn()

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, f"{algorithm}_hill_climb.zip")
    agent.save(model_path)
    
    print("--- Training Finished ---")
    print(f"Model saved to: {model_path}")
    env.close()


def visualize(algorithm):
    """
    Visualizes a pre-trained agent.
    """
    model_path = os.path.join(MODELS_DIR, f"{algorithm}_hill_climb.zip")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return

    print(f"--- Loading model: {model_path} ---")

    env = HillClimbEnv(render_mode="human")
    
    # --- Select the correct model class to load ---
    if algorithm == 'ppo':
        model = ppo.PPO.load(model_path, env=env)
    elif algorithm == 'dqn':
        model = dqn.MyDQN.load(model_path, env=env)
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
    parser.add_argument("algorithm", choices=["ppo", "dqn"], help="Algorithm to use.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps for training.")
    
    args = parser.parse_args()

    if args.action == "train":
        train(args.algorithm, args.timesteps)
    elif args.action == "visualize":
        visualize(args.algorithm)
