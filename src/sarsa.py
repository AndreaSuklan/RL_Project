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

from approximations import Linear, Polynomial
from networks import MLP_Small, SimpleNN
from buffers import ReplayBuffer

class SARSA:
    """SARSA agent using a neural network for Q-value approximation (no replay buffer, online updates, decaying epsilon)."""
    def __init__(self, env, gamma=0.99, lr=0.001, epsilon=0.1, min_epsilon=0.01, epsilon_decay=0.995, model="nn", degree=3):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.model = model
        self.degree = degree

        if isinstance(model, str):
            self.policy = self.__class__.create_model(env, model, degree)
        else:
            self.policy = model
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
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
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy(state_tensor)
            return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        q_values = self.policy(state_tensor)
        q_value = q_values[action]

        # compute expectation for ExpectedSarsa
        with torch.no_grad():
            next_q = self.policy(next_state_tensor)             
            n = next_q.size(0)
            greedy = torch.argmax(next_q)
            probs = torch.full((n,), self.epsilon / n, device=next_q.device)
            probs[greedy] += 1 - self.epsilon
            expected_next_q = (probs * next_q).sum()            

        target = reward + self.gamma * expected_next_q * (1 - done)
        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learn(self, total_episodes=1000, max_steps_per_episode=500, verbose=0):
        """Train the SARSA agent.
        Args:
            total_episodes: Total number of episodes to train the agent.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Verbosity level (0 for no output, 1 for progress updates).
        """
        log_data = []
        current_timesteps = 0
        self.performance_traj = []

        for episode in tqdm(range(total_episodes), desc="Training Episodes"):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                loss = self.train_step(state, action, reward, next_state, done)
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                episode_reward += reward
                current_timesteps += 1

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.performance_traj.append(episode_reward)

            mean_loss = np.mean(episode_losses) if episode_losses else None
            log_data.append({
                "timestep": current_timesteps,
                "reward": episode_reward,
                "value_loss": mean_loss
            })

            if verbose > 0 and (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.performance_traj[-10:])
                print_loss = mean_loss if mean_loss is not None else float('nan')
                print(f"\nEpisode {episode+1}, Timestep {current_timesteps}, Mean Reward (last 10): {mean_reward:.2f}, Mean Loss: {print_loss:.4f}")

        return log_data

    def predict(self, observation, deterministic=True):
        """Predict the action based on the current observation.
        Args:
            observation: The current state of the environment.
            deterministic: If True, select the action with the highest Q-value.
        Returns:
            The predicted action.
        """
        state_tensor = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy(state_tensor)
            if deterministic:
                return torch.argmax(q_values).item()
            else:
                probs = F.softmax(q_values, dim=0).numpy()
                return np.random.choice(self.action_size, p=probs)

    def save(self, path="sarsa_model.zip"):
        """Save the SARSA model to a zip file."""
        tmp_dir = "tmp_sarsa_save"
        os.makedirs(tmp_dir, exist_ok=True)

        torch.save(self.policy.state_dict(), os.path.join(tmp_dir, "policy.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pth"))

        metadata = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "action_size": self.action_size,
            "lr": self.lr,
            "performance_traj": self.performance_traj,
            #"batch_size": self.batch_size
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
        """Load a SARSA model from a zip file."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(path, 'r') as zipf:
                zipf.extractall(tmp_dir)

            with open(os.path.join(tmp_dir, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)

            model = cls(env=env, gamma=meta["gamma"], lr=meta["lr"], epsilon=meta["epsilon"])
            model.performance_traj = meta["performance_traj"]

            model.policy.load_state_dict(torch.load(os.path.join(tmp_dir, "policy.pth")))
            model.optimizer.load_state_dict(torch.load(os.path.join(tmp_dir, "optimizer.pth")))

        return model