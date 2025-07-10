import gymnasium as gym
from stable_baselines3 import PPO
from race import HillClimbEnv, MyContactListener

# --- Main execution block ---
if __name__ == '__main__':
    # Path to the saved model file
    MODEL_PATH = "ppo_hcr_model.zip"
    # Number of episodes to visualize
    NUM_EPISODES = 10

    # Create the Hill Climb environment with rendering enabled
    env = HillClimbEnv(render_mode="human")
    # Set the custom contact listener
    env.contactlistener = MyContactListener(env)

    try:
        # Load the pre-trained PPO model
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        # Handle the case where the model file is not found
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please make sure the trained model file exists in the same directory as this script.")
        exit()

    print("Starting Visualization")
    print(f"Model: {MODEL_PATH}")
    print(f"Running for {NUM_EPISODES} episodes.")

    # Loop through the specified number of episodes
    for episode in range(NUM_EPISODES):
        # Reset the environment for a new episode
        obs, info = env.reset()

        print(f"\nStarting Episode {episode + 1}/{NUM_EPISODES}")

        # Run the episode until it is terminated or truncated
        while True:
            # Get the action from the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take a step in the environment with the chosen action
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if the episode has ended
            if terminated or truncated:
                if terminated:
                    print("Episode finished: Terminated (e.g., car flipped).")
                elif truncated:
                    print("Episode finished: Truncated (e.g., time limit reached).")
                break

    # Close the environment and the rendering window
    env.close()
    print("\nVisualization finished")