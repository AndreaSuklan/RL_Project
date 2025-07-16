import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import zipfile
import io

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.body = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor head
        self.policy_head = nn.Linear(128, action_dim)
        
        # Critic head
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        shared_features = self.body(state)
        action_logits = self.policy_head(shared_features)
        state_value = self.value_head(shared_features)
        
        # Return a probability distribution for actions and the state value
        action_dist = Categorical(logits=action_logits)
        return action_dist, state_value
    

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, gamma, gae_lambda):
        # Store parameters
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate memory
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

        # These will be calculated after the rollout is complete
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Pointer to the current position in the buffer
        self.ptr = 0

    def add(self, state, action, reward, done, log_prob, value):
        """Adds one step of experience to the buffer."""
        if self.ptr >= self.buffer_size:
            # This should not happen if the buffer is cleared correctly
            raise BufferError("Rollout buffer is full.")
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        Computes the advantages and returns for the rollout using GAE.
        This should be called after the buffer is full.
        """
        # We use the value of the last state to bootstrap the calculation
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # Calculate the advantage
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Calculate the returns (value targets)
        self.returns = self.advantages + self.values

    def get(self):
        """Returns all buffer data as PyTorch tensors."""
        # Normalize advantages for more stable training
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Convert numpy arrays to torch tensors
        return (
            torch.tensor(self.states, dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.returns, dtype=torch.float32),
            torch.tensor(self.advantages, dtype=torch.float32),
        )

    def clear(self):
        """Clears the buffer by resetting the pointer."""
        self.ptr = 0


class PPO():
    """
    Proximal Policy Optimization (PPO) agent implementation.
    This class implements the PPO algorithm for training an agent in a Gymnasium environment.
    It uses an actor-critic architecture with a shared body for both the actor and critic networks.
    The agent collects experiences in a rollout buffer, computes advantages and returns using Generalized Advantage Estimation (GAE),
    and optimizes the policy using a clipped surrogate objective.   

    Args:
        env: The Gymnasium environment to train the agent in.
        buffer_size: The size of the rollout buffer.
        gamma: Discount factor for future rewards.
        gae_lambda: Lambda parameter for GAE.
        lr: Learning rate for the optimizer.
        clip_epsilon: Clipping parameter for the PPO objective.
        n_epochs: Number of epochs to train on each batch of data.
        minibatch_size: Size of the minibatches for optimization.
    """
    def __init__(self, env, buffer_size=2048, gamma=0.99, gae_lambda=0.95, lr=3e-4, clip_epsilon=0.2, n_epochs=10, minibatch_size=64, verbose=0):
        # Store parameters
        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.clip_epsilon = clip_epsilon
        self.verbose = verbose

        # Initialize the rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_dim=env.observation_space.shape[0],
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        # Initialize the actor-critic model
        self.actor_critic = ActorCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=lr
        )



    def learn(self, total_timesteps):
        state, _ = self.env.reset()
        current_timesteps = 0

        # Variables for logging statistics
        episode_rewards = []
        current_episode_reward = 0

        while current_timesteps < total_timesteps:

            # ROLLOUT PHASE
            for _ in range(self.buffer_size):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                # Get action and value from the actor-critic model
                with torch.no_grad():
                    action_dist, value = self.actor_critic(state_tensor)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)

                # Take action in the environment
                try:
                    next_state, reward, done, truncated, _ = self.env.step(action.item())
                except Exception as e:
                    print(f"Error during env.step(): {e}")
                    raise
                current_episode_reward += reward

                # Store the experience in the rollout buffer
                self.rollout_buffer.add(state, action.item(), reward, done, log_prob.item(), value.item())
                state = next_state
                current_timesteps += 1

                if done or truncated:
                    state, _ = self.env.reset()
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0


            # CALCULATION PHASE
            # Compute value for the last state
            with torch.no_grad():
                last_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                _, last_value = self.actor_critic(last_state_tensor)
            
            # Compute returns and advantages
            self.rollout_buffer.compute_returns_and_advantages(last_value.item(), done)
            

            # OPTIMIZATION PHASE
            states, actions, old_log_probs, returns, advantages = self.rollout_buffer.get()
            
            # Track losses for logging
            policy_losses, value_losses, entropy_losses = [], [], []

            for _ in range(self.n_epochs):
                # Shuffle and create minibatches
                indices = np.random.permutation(self.buffer_size)
                for start in range(0, self.buffer_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    batch_indices = indices[start:end]

                    # Get minibatch data
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]

                    # Evaluate current policy
                    action_dist, values = self.actor_critic(batch_states)
                    new_log_probs = action_dist.log_prob(batch_actions)
                    entropy = action_dist.entropy().mean()
                    
                    # Calculate PPO Loss
                    # Policy Loss
                    #       r = pi_new / pi_old
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    #       r * \hat{A}
                    surr1 = ratio * batch_advantages
                    #       clip(r, 1 - \epsilon, 1 + \epsilon) * \hat{A}
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    #       L^CLIP = min(r * \hat{A}, clip(r, 1 - \epsilon, 1 + \epsilon) * \hat{A})
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value Loss
                    #       L^VF = (V(s) - R)^2
                    value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)

                    # Total Loss
                    #       L = L^CLIP + 0.5 * L^VF - 0.01 * H
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Append losses for this minibatch
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy.item())

            # LOGGING
            if self.verbose > 0:
                # Calculate mean reward only if episodes have finished in this rollout
                mean_reward = np.mean(episode_rewards) if episode_rewards else None

                print("-" * 60)
                print(f"Timestep: {current_timesteps}/{total_timesteps}")
                if mean_reward is not None:
                    print(f"Mean Reward (last {len(episode_rewards)} episodes): {mean_reward:.2f}")
                print(f"Mean Policy Loss: {np.mean(policy_losses):.4f}")
                print(f"Mean Value Loss: {np.mean(value_losses):.4f}")
                print(f"Mean Entropy: {np.mean(entropy_losses):.4f}")
                print("-" * 60)

            # Reset for the next logging cycle
            episode_rewards = []

            # Clear Buffer
            self.rollout_buffer.clear()
    

    def save(self, path):
        """Saves the model's weights and hyperparameters to a .zip file."""
        # Ensure the path ends with .zip
        if not path.endswith(".zip"):
            path += ".zip"

        # Data to be saved
        data_to_save = {
            "model_state_dict": self.actor_critic.state_dict(),
            "hyperparameters": {
                "buffer_size": self.buffer_size,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "lr": self.lr,
                "clip_epsilon": self.clip_epsilon,
                "n_epochs": self.n_epochs,
                "minibatch_size": self.minibatch_size
            }
        }
        
        # Save to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(data_to_save, buffer)
        buffer.seek(0) # Rewind the buffer

        # Write the buffer to a zip file
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("agent_data.pth", buffer.read())
        
        print(f"Agent saved to {path}")


    @staticmethod
    def load(path, env):
        """Loads an agent from a .zip file."""
        # Ensure the path ends with .zip
        if not path.endswith(".zip"):
            path += ".zip"
            
        # Open the zip file and load the data
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open("agent_data.pth") as f:
                data = torch.load(f)

        # Create a new agent with the saved hyperparameters
        hyperparameters = data["hyperparameters"]
        agent = PPO(env, **hyperparameters)
        
        # Load the learned weights
        agent.actor_critic.load_state_dict(data["model_state_dict"])
        
        print(f"Agent loaded from {path}")
        return agent
    

    def predict(self, observation, deterministic=True):
        """
        Predicts the action to take for a given observation during inference.

        Args:
            observation: The input observation from the environment.
            deterministic (bool): If True, take the most likely action. If False, sample from the distribution.

        Returns:
            The selected action as an integer.
        """
        # Convert observation to a PyTorch tensor and add a batch dimension
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad(): # Disable gradient calculations
            # Get action distribution from the actor network
            action_dist, _ = self.actor_critic(state_tensor)

            if deterministic:
                # Select the action with the highest probability
                action = torch.argmax(action_dist.logits)
            else:
                # Sample a random action from the distribution
                action = action_dist.sample()
        
        # Return the action as a Python number
        return action.item()
        



def create_agent(env, log_dir=None):
    """
    Creates and returns a PPO agent.
    
    Args:
        env: The Gymnasium environment.
        log_dir: The directory to save logs (not used in this scratch implementation yet).
    
    Returns:
        An instance of PPO class.
    """
    print("Creating PPO agent...")
    agent = PPO(env, verbose=1)
    return agent

