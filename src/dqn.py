import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from approximations import Linear, Polynomial
from base import RlAlgorithm
from buffers import ReplayBuffer
from tqdm import tqdm
from networks import MLP_Small
from approximations import Linear, Polynomial

class DQN(RlAlgorithm):
    """Deep Q-Network (DQN) agent.
    This agent uses a simple feedforward neural network to approximate Q-values.
    It implements experience replay and epsilon-greedy action selection.
    Arguments:
        env: The environment to interact with.
        gamma: Discount factor for future rewards.
        lr: Learning rate for the optimizer.
        epsilon: Initial exploration rate for epsilon-greedy action selection.
        buffer_size: Maximum size of the replay buffer.
        batch_size: Number of samples to draw from the replay buffer for training.
    """
    def __init__(self, env, gamma=0.99, lr=0.001, epsilon=0.1, buffer_size=10000, batch_size=64, model="nn", degree=3, verbose=0):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.model = model
        self.degree = degree

        if isinstance(model, str):
            self.model = self.__class__.create_model(env, model, degree)
        else:
            self.model = model

        super().__init__(env, model=self.model, buffer_size=buffer_size, gamma=gamma, lr=lr, batch_size=batch_size, verbose=0)


        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.performance_traj = []

    @staticmethod
    def create_model(env, model, degree):
        if model == "linear":
            model= Linear(env.observation_space.shape[0], env.action_space.n)
        elif model == "nn":
            model = MLP_Small(env.observation_space.shape[0], env.action_space.n)
        elif model == "poly":
            model = Polynomial(env.observation_space.shape[0], env.action_space.n, degree=degree)
        else:
            raise ValueError(f"Unrecognized model {model}")
        return model
        
    def predict(self, observation, deterministic=True):
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            if self.verbose >= 2:
                print(f"[PREDICT] Q-values: {q_values.numpy()}")
            return torch.argmax(q_values).item()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
            if self.verbose >= 2:
                print(f"[SELECT] Random action: {action} (epsilon={self.epsilon})")
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
                if self.verbose >= 2:
                    print(f"[SELECT] Greedy action: {action} from Q-values {q_values.numpy()}")
                return action

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            if self.verbose >= 2:
                print(f"[TRAIN] Not enough data in buffer ({len(self.buffer)}/{self.batch_size})")
            return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.verbose >= 2:
            print(f"[TRAIN] Loss: {loss.item():.4f}, Mean Target: {targets.mean().item():.4f}, Mean Q: {q_values.mean().item():.4f}")

        return loss.item(), None

    def learn(self, total_timesteps, verbose=0):
        self.verbose = verbose  # set the verbosity level
        log_data = []

        state, _ = self.env.reset()
        current_timesteps = 0
        episode_num = 0

        pbar = tqdm(total=total_timesteps, desc="Training DQN")

        while current_timesteps < total_timesteps:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, reward, next_state, done)
            state = next_state

            self.performance_traj.append(reward)
            current_timesteps += 1
            pbar.update(1)

            loss, _ = self.train_step()

            log_data.append({
                    "timestep": current_timesteps,
                    "reward": reward,
                    "value_loss": loss
                })

            if done:
                episode_num += 1

                if self.verbose >= 1 and episode_num % 10 == 0:
                    mean_reward = np.mean(self.performance_traj[-10:])
                    loss_display = f"{loss:.4f}" if loss is not None else "None"
                    print(f"\n[INFO] Episode {episode_num}, Timestep {current_timesteps}, Mean Reward (last 10): {mean_reward:.2f}, Last Loss: {loss_display}")

                state, _ = self.env.reset()

        pbar.close()
        return log_data

