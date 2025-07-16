# Reinforcement Learning for Hill Climb Racing

## Project Overview

This project aims to develop and compare reinforcement learning agents capable of playing a 2D physics-based driving game similar to Hill Climb Racing. The environment is built using **Gymnasium** and **Box2D**, and the project is structured to allow for the modular implementation and training of different RL algorithms.

The current setup includes custom **PPO** (Proximal Policy Optimization) and **DQN** (Deep Q-Network) agents.

***

## File Structure

The project is organized into several key files within the `src` directory:

* **`main.py`**: The main entry point for the project. It handles command-line arguments to either train a new agent or visualize a pre-trained one.
* **`environment.py`**: Contains the `HillClimbEnv` class, which defines the game world, physics, rewards, and observation/action spaces.
* **`ppo.py`**: Defines the agent creation logic for the PPO algorithm.
* **`dqn.py`**: Defines the agent creation logic for the DQN algorithm.

***

## How to Run

### 1. Training an Agent

To train an agent, run the `main.py` script from the directory outside of `src` with the train action, specifying the algorithm.

- Train with PPO:
  ```bash
  python src/main.py train ppo
  ```

- Train with DQN:
  ```bash
  python src/main.py train dqn
  ```

The trained model will be saved as a `.zip` file inside the `models/` directory.


### 2. Visualizing an Agent
Once a model has been trained, you can watch it play the game using the `visualize` action.

- Visualize the PPO agent:
  ```bash
  python src/main.py visualize ppo
  ```

- Visualize the DQN agent:
  ```bash
  python src/main.py visualize dqn
  ```
