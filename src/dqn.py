# train_dqn.py

from stable_baselines3 import DQN
from environment import HillClimbEnv

# --- Configuration ---
MODEL_NAME = "dqn_hill_climb"
TOTAL_TIMESTEPS = 300_000  # DQN often requires more samples
LOG_DIR = "./logs/dqn_logs/"
MODEL_DIR = "./models/"

def train():
    """Trains a DQN agent."""
    print("--- Starting DQN Training ---")
    
    # Create the environment
    env = HillClimbEnv()
    
    # Instantiate the DQN model
    # DQN has different hyperparameters, e.g., buffer_size, learning_starts
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05
    )
    
    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save the model
    model_path = f"{MODEL_DIR}{MODEL_NAME}"
    model.save(model_path)
    
    print(f"--- Training Finished ---")
    print(f"Model saved to: {model_path}.zip")
    
    env.close()

if __name__ == '__main__':
    train()