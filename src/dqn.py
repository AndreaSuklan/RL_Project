import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import zipfile
import pickle
import os
from tqdm import tqdm


class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.from_numpy(np.array(states)).float(),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.array(next_states)).float(),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, env, gamma=0.99, lr=0.001, epsilon=0.1, replay_capacity=10000, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.batch_size = batch_size

        self.policy = SimpleDQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.performance_traj = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy(state_tensor)
            return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.policy(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learn(self, total_episodes=1000, max_steps_per_episode=500):
        self.performance_traj = []
        for episode in tqdm(range(total_episodes), desc="Training Episodes"):
            state, _ = self.env.reset()
            total_reward = 0

            for _ in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                loss = self.train_step()
                if done:
                    break

            self.performance_traj.append(total_reward)

            if verbose > 0 and (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.performance_traj[-10:])
                print(f"Episode {episode+1}, Mean Reward (last 10): {mean_reward:.2f}, Last Loss: {loss:.4f}" if loss is not None else f"Episode {episode+1}, Mean Reward (last 10): {mean_reward:.2f}")

    def predict(self, observation, deterministic=True):
        state_tensor = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy(state_tensor)
            if deterministic:
                return torch.argmax(q_values).item()
            else:
                probs = F.softmax(q_values, dim=0).numpy()
                return np.random.choice(self.action_size, p=probs)

    def save(self, path="dqn_model.zip"):
        tmp_dir = "tmp_dqn_save"
        os.makedirs(tmp_dir, exist_ok=True)

        torch.save(self.policy.state_dict(), os.path.join(tmp_dir, "policy.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pth"))

        metadata = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "action_size": self.action_size,
            "lr": self.lr,
            "performance_traj": self.performance_traj,
            "batch_size": self.batch_size
        }
        with open(os.path.join(tmp_dir, "meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        with zipfile.ZipFile(path, 'w') as zipf:
            for fname in ["policy.pth", "optimizer.pth", "meta.pkl"]:
                zipf.write(os.path.join(tmp_dir, fname), fname)

        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)

    @classmethod
    def load(cls, path, env):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(path, 'r') as zipf:
                zipf.extractall(tmp_dir)

            with open(os.path.join(tmp_dir, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)

            model = cls(env=env, gamma=meta["gamma"], lr=meta["lr"], epsilon=meta["epsilon"], batch_size=meta["batch_size"])
            model.performance_traj = meta["performance_traj"]

            model.policy.load_state_dict(torch.load(os.path.join(tmp_dir, "policy.pth")))
            model.optimizer.load_state_dict(torch.load(os.path.join(tmp_dir, "optimizer.pth")))

        return model
