# refactored_dqn.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from base import RlAlgorithm
from buffers import ReplayBuffer
from networks import SimpleDQN  # assumed external
from tqdm import tqdm


class DQN(RlAlgorithm):
    def __init__(self, env, model,  buffer_size=10000, gamma=0.99, lr=1e-3, epsilon=0.1, batch_size=64):
        self.epsilon = epsilon
        self.batch_size = batch_size
        super().__init__(env, model=model, buffer_size=buffer_size, gamma=gamma, lr=lr, batch_size=batch_size)

        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.performance_traj = []

    def predict(self, observation, deterministic=True):
        state_tensor = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            if deterministic:
                return torch.argmax(q_values).item()
            else:
                probs = F.softmax(q_values, dim=0).numpy()
                return np.random.choice(self.action_size, p=probs)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()


    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learn(self, total_episodes=1000, max_steps_per_episode=500, verbose=0):
        self.performance_traj = []

        for episode in tqdm(range(total_episodes), desc="Training Episodes"):
            state, _ = self.env.reset()
            total_reward = 0

            for _ in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                loss = self.train_step()
                if done:
                    break

            self.performance_traj.append(total_reward)

            if verbose and (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.performance_traj[-10:])
                if loss is not None:
                    print(f"\nEpisode {episode+1}, Mean Reward (last 10): {mean_reward:.2f}, Last Loss: {loss:.4f}")
                else:
                    print(f"\nEpisode {episode+1}, Mean Reward (last 10): {mean_reward:.2f}")

