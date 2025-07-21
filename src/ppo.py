import torch
import torch.nn.functional as F
import numpy as np
from buffers import RolloutBuffer
from base import RlAlgorithm
from tqdm import tqdm


class PPO(RlAlgorithm):
    def __init__(self, env, model, buffer_size=2048, gamma=0.99, gae_lambda=0.95,
                 lr=3e-4, clip_epsilon=0.2, n_epochs=10, batch_size=64):
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        super().__init__(env, model=model, buffer_size=buffer_size, gamma=gamma, lr=lr, batch_size=batch_size)

        self.buffer = RolloutBuffer(
            buffer_size = buffer_size,
            state_dim = self.state_size,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

    def predict(self, observation, deterministic=True):
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.model(state_tensor)
            if deterministic:
                return torch.argmax(dist.logits).item()
            return dist.sample().item()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.model(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def train_step(self, states, actions, old_log_probs, returns, advantages):
        policy_losses, value_losses, entropy_losses = [], [], []
        for _ in range(self.n_epochs):
            indices = np.random.permutation(self.buffer_size)
            for start in range(0, self.buffer_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                dist, values = self.model(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        return policy_losses, value_losses, entropy_losses

    def learn(self, total_timesteps, verbose=0):
        state, _ = self.env.reset()
        total_steps = 0
        episode_rewards = []
        current_reward = 0

        pbar = tqdm(total=total_timesteps, desc="Training PPO")

        while total_steps < total_timesteps:
            for _ in range(self.buffer_size):
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.buffer.add(state, action, reward, done, log_prob, value)
                state = next_state
                current_reward += reward
                total_steps += 1
                pbar.update(1)

                if done or truncated:
                    episode_rewards.append(current_reward)
                    current_reward = 0
                    state, _ = self.env.reset()

            with torch.no_grad():
                last_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                _, last_value = self.model(last_state_tensor)

            self.buffer.compute_returns_and_advantages(last_value.item(), done)
            states, actions, old_log_probs, returns, advantages = self.buffer.sample()
            policy_losses, value_losses, entropy_losses = self.train_step(
                states, actions, old_log_probs, returns, advantages
            )

            if verbose:
                print("-" * 60)
                print(f"Timestep: {total_steps}/{total_timesteps}")
                if episode_rewards:
                    print(f"Mean Reward (last {len(episode_rewards)} episodes): {np.mean(episode_rewards):.2f}")
                print(f"Mean Policy Loss: {np.mean(policy_losses):.4f}")
                print(f"Mean Value Loss: {np.mean(value_losses):.4f}")
                print(f"Mean Entropy: {np.mean(entropy_losses):.4f}")
                print("-" * 60)

            episode_rewards = []
            self.buffer.clear()

        pbar.close()

