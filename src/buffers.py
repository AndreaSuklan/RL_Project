import numpy as np
import torch
from collections import deque


class BaseBuffer:
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class ReplayBuffer(BaseBuffer):
    """Off-policy replay buffer for DQN and similar algorithms."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in samples])
        states = np.array(states)
        next_states = np.array(next_states)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class RolloutBuffer(BaseBuffer):
    """On-policy rollout buffer for PPO and similar algorithms."""
    def __init__(self, buffer_size, state_dim, gamma, gae_lambda):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0

    def add(self, state, action, reward, done, log_prob, value):
        if self.ptr >= self.buffer_size:
            raise BufferError("Rollout buffer is full.")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def sample(self):
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (
            torch.tensor(self.states, dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.int64),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.returns, dtype=torch.float32),
            torch.tensor(self.advantages, dtype=torch.float32),
        )

    def clear(self):
        self.ptr = 0

