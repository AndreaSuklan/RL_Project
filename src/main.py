import argparse
import os
from environment import HillClimbEnv
from ppo import PPO
from dqn import DQN
from sarsa import SARSA 

# --- Directory Setup ---
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train(algorithm, model = "nn", degree=3):
    """
    Trains a new agent by calling the appropriate creation function.
    """
    print(f"--- Starting training for {algorithm.upper()} with {model} model ---")
    
    env = HillClimbEnv()

    # --- Create and train the agent ---
    if algorithm == 'ppo':
        if model != "nn":
            print(f"Warning: PPO does not support {model} model, using 'nn' instead.")
            model = "nn"
        agent = PPO(
            env,
            buffer_size=2048, 
            gamma=0.99, 
            gae_lambda=0.95, 
            lr=3e-4, 
            clip_epsilon=0.2, 
            n_epochs=10, 
            minibatch_size=64, 
        )
        agent.learn(timesteps=200000, verbose=1)
        
    elif algorithm == 'dqn':
        agent = DQN(
            env, 
            gamma=0.99, 
            lr=0.001, 
            epsilon=0.1, 
            replay_capacity=10000, 
            batch_size=64)
        agent.learn(total_episodes=100, max_steps_per_episode=2000, verbose=1)

    elif algorithm == 'sarsa':
        agent = SARSA(
            env, 
            gamma=0.99, 
            lr=0.001, 
            epsilon=0.1
            )
        agent.learn(total_episodes=100, max_steps_per_episode=2000, verbose=1)
        
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
    
    print("--- Training Finished ---")
    print(f"Model saved to: {model_path}")
    env.close()


def visualize(algorithm, model = "nn", degree=3):
    """
    Visualizes a pre-trained agent.
    """
    
    model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_{degree}_hill_climb.zip")
        
    if not os.path.exists(model_path) and model == "poly":
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_{model}_hill_climb.zip")
        print(f"Poly with degree {degree} not found, trying visulizing degree = 3")

    if not os.path.exists(model_path):
        print(f"{model.upper()} model not found, visualizing {algorithm} with nn architecture")
        model_path = os.path.join(MODELS_DIR, f"{algorithm}_hill_climb.zip")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return

    print(f"--- Loading model: {model_path} ---")
    env = HillClimbEnv(render_mode="human")

    if algorithm == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algorithm == 'dqn':
        model = DQN.load(model_path, env=env)
    elif algorithm == 'sarsa':
        model = SARSA.load(model_path, env=env)
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

    args = parser.parse_args()

    if args.action == "train":
        train(args.algorithm, args.model, args.degree)
    elif args.action == "visualize":
        visualize(args.algorithm, args.model, args.degree)