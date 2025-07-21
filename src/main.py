import argparse
import os
from environment import HillClimbEnv
from ppo import PPO
from dqn import DQN
from networks import SimpleDQN, ActorCritic

# --- Configuration ---
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train(algorithm):
    """
    Trains a new agent by calling the appropriate creation function.
    """
    print(f"--- Starting training for {algorithm.upper()} ---")
    
    env = HillClimbEnv()

    if algorithm == 'ppo':
        model = ActorCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
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
        agent.learn(total_timesteps=200000, verbose=1)
        
    elif algorithm == 'dqn':
        model = SimpleDQN(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n
        )
        agent = DQN(
            env,
            model=model,
            gamma=0.99, 
            buffer_size=10000, 
            lr=0.001, 
            epsilon=0.1, 
            batch_size=64)
        agent.learn(total_episodes=100, max_steps_per_episode=2000, verbose=1)

    else:
        print(f"Error: Unknown algorithm '{algorithm}'")
        env.close()
        return
    
    # Save the trained model
    print("--- Training Finished ---")
    model_path = os.path.join(MODELS_DIR, f"{algorithm}_hill_climb.zip")
    agent.save(model_path)
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
    
    if algorithm == 'ppo':
        model = PPO.load(model_path, env=env, model_class=ActorCritic)
    elif algorithm == 'dqn':
        model = DQN.load(model_path, env=env, model_class=SimpleDQN)
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
    
    args = parser.parse_args()

    if args.action == "train":
        train(args.algorithm)
    elif args.action == "visualize":
        visualize(args.algorithm)