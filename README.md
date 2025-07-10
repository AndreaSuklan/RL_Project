# Reinforcement Learning for Hill Climb Racing

## Project Overview

This project is a starting point for applying reinforcement learning (RL) algorithms to a 2D physics-based driving game similar to Hill Climb Racing. The goal is to train an agent that can control a vehicle to navigate a randomly generated, hilly terrain.

This implementation uses the following libraries:
- **Gymnasium**: A standard API for reinforcement learning environments.
- **Pygame**: For rendering the game environment.
- **Box2D**: A 2D physics engine to simulate the car and terrain interactions.
- **Stable-Baselines3**: A set of reliable implementations of RL algorithms.

## Current Status

The project currently consists of two main Python scripts: `race.py` and `visualize.py`.

### `race.py`

This script contains the core components of the project:

- **`HillClimbEnv`**: A custom Gymnasium environment that defines the game world, the car, and the rules of interaction.
  - **State/Observation**: The agent's observation of the environment includes the car's physical properties (angle, velocity) and LIDAR-like sensor readings that detect the distance to the terrain.
  - **Actions**: The agent can choose from three discrete actions: do nothing, accelerate left, or accelerate right.
  - **Reward**: The reward function is designed to encourage forward progress. The agent receives a positive reward for moving to the right and a large negative penalty for crashing (e.g., the driver hitting the ground) or moving backward.
- **Training Loop**: The `if __name__ == '__main__':` block demonstrates how to create the environment, instantiate a Proximal Policy Optimization (PPO) model from Stable-Baselines3, and train it for a set number of timesteps. After training, the model is saved to a file (`ppo_hcr_model.zip`).

### `visualize.py`

This script is used to load a pre-trained model and visualize its performance in the game environment. It loads the `ppo_hcr_model.zip` file, creates the `HillClimbEnv` with rendering enabled, and runs the agent for a specified number of episodes.

## How to Run

1. **Training the Agent**
To train the agent, run the race.py script:

```
python race.py
```

This will start the training process. You will see output from Stable-Baselines3 in your terminal, and a TensorBoard log directory (`./ppo_hcr_tensorboard/`) will be created for monitoring training progress. Once training is complete, a file named `ppo_hcr_model.zip` will be saved in the same directory. The script will then automatically load the saved model and run a short visualization.

2. **Visualizing a Trained Agent**
After a model has been trained and saved, you can visualize its performance by running the visualize.py script:

```
python visualize.py
```
This will open a Pygame window and show the agent driving the car through the environment for 10 episodes.