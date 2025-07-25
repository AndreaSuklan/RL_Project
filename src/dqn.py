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

def polynomial_features(x, degree=2):
    """
    Compute polynomial features up to the given degree.
    x: Tensor of shape [batch_size, input_dim]
    Returns: Tensor of shape [batch_size, num_poly_features]
    """
    # Start with degree 1 features
    features = [x]
    
    for d in range(2, degree + 1):
        features.append(x ** d)

class SimpleDQN(nn.Module):
    """A simple feedforward neural network for DQN policy.
    It takes the state as input and outputs Q-values for each action.
    """
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class LinearDQN(nn.Module):
    """Linear function approximator for DQN."""
    def __init__(self, input_dim, output_dim):
        super(LinearDQN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class PolynomialDQN(nn.Module):
    """Polynomial function approximator for DQN."""
    def __init__(self, input_dim, output_dim, degree=2):
        super(PolynomialDQN, self).__init__()
        self.degree = degree
        poly_dim = input_dim * degree  # e.g., x, x^2, ..., x^degree
        self.linear = nn.Linear(poly_dim, output_dim)

    def forward(self, x):
        poly_x = polynomial_features(x, self.degree)
        return self.linear(poly_x)

class ReplayBuffer:
    """A simple replay buffer to store experiences for DQN training."""
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
    """Deep Q-Network (DQN) agent.
    This agent uses a simple feedforward neural network to approximate Q-values.
    It implements experience replay and epsilon-greedy action selection.
    Arguments:
        env: The environment to interact with.
        gamma: Discount factor for future rewards.
        lr: Learning rate for the optimizer.
        epsilon: Initial exploration rate for epsilon-greedy action selection.
        replay_capacity: Maximum size of the replay buffer.
        batch_size: Number of samples to draw from the replay buffer for training.
    """
    def __init__(self, env, gamma=0.99, lr=0.001, epsilon=0.1, replay_capacity=10000, batch_size=64, model="nn", degree=3):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.batch_size = batch_size
        self.model = model
        self.degree = degree

        #Policy selection
        if model == "linear":
            self.policy = LinearDQN(self.state_size, self.action_size)
        elif model == "nn":
            self.policy = SimpleDQN(self.state_size, self.action_size)
        elif model == "poly":
            self.policy = PolynomialDQN(self.state_size, self.action_size, degree=self.degree)
        else:
            raise ValueError(f"Unrecognized model {self.model}")

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

    def learn(self, total_episodes=1000, max_steps_per_episode=500, verbose=0):
        """Train the DQN agent.
        Args:
            total_episodes: Total number of episodes to train the agent.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Verbosity level (0 for no output, 1 for progress updates).
        """ 
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
                print(f"\nEpisode {episode+1}, Mean Reward (last 10): {mean_reward:.2f}, Last Loss: {loss:.4f}" if loss is not None else f"Episode {episode+1}, Mean Reward (last 10): {mean_reward:.2f}")

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

    def save(self, path="dqn_model.zip"):
        """Save the DQN model to a zip file."""
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
        """Load a DQN model from a zip file."""
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
