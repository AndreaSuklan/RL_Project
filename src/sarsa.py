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
# def polynomial_features(x, degree=2):
#     """
#     Compute polynomial features up to the given degree.
#     x: Tensor of shape [batch_size, input_dim]
#     Returns: Tensor of shape [batch_size, num_poly_features]
#     """
#     # Start with degree 1 features
#     features = [x]
    
#     for d in range(2, degree + 1):
#         features.append(x ** d)
    
#     return torch.cat(features, dim=1)

# class SimpleSARSA(nn.Module):
#     """A simple feedforward neural network for SARSA policy.
#     It takes the state as input and outputs Q-values for each action.
#     """
#     def __init__(self, input_dim, output_dim):
#         super(SimpleSARSA, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    
# class LinearSARSA(nn.Module):
#     """Linear function approximator for SARSA."""
#     def __init__(self, input_dim, output_dim):
#         super(LinearSARSA, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)
    
# class PolynomialSARSA(nn.Module):
#     """Polynomial function approximator for SARSA."""
#     def __init__(self, input_dim, output_dim, degree=2):
#         super(PolynomialSARSA, self).__init__()
#         self.degree = degree
#         poly_dim = input_dim * degree  # e.g., x, x^2, ..., x^degree
#         self.linear = nn.Linear(poly_dim, output_dim)

#     def forward(self, x):
#         poly_x = polynomial_features(x, self.degree)
#         return self.linear(poly_x)

# class ReplayBuffer:
#     """A simple replay buffer to store experiences for training."""
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         samples = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*samples)

#         return (
#             torch.from_numpy(np.array(states)).float(),
#             torch.tensor(actions, dtype=torch.int64),
#             torch.tensor(rewards, dtype=torch.float32),
#             torch.from_numpy(np.array(next_states)).float(),
#             torch.tensor(dones, dtype=torch.float32)
#         )

#     def __len__(self):
#         return len(self.buffer)


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
            self.model = self.__class__.create_model(env, model, degree)
        else:
            self.model = model
        
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
            next_q = self.policy(next_state_tensor)              # tensor [n_actions]
            n = next_q.size(0)
            greedy = torch.argmax(next_q)
            probs = torch.full((n,), self.epsilon / n, device=next_q.device)
            probs[greedy] += 1 - self.epsilon
            expected_next_q = (probs * next_q).sum()             # still a tensor

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
        self.performance_traj = []
        for episode in tqdm(range(total_episodes), desc="Training Episodes"):
            state, _ = self.env.reset()
            total_reward = 0

            for _ in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                loss = self.train_step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
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
