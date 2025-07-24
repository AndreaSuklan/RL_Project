import numpy as np
import torch
import torch.nn.functional as F
from base import RlAlgorithm
from buffers import ReplayBuffer
from tqdm import tqdm

class DQN(RlAlgorithm):
    def __init__(self, env, model, buffer_size=10000, gamma=0.99, lr=1e-3, epsilon=0.1, batch_size=64, verbose=0):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.verbose = verbose
        super().__init__(env, model=model, buffer_size=buffer_size, gamma=gamma, lr=lr, batch_size=batch_size)

        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.performance_traj = []

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
        episode_reward = 0
        episode_num = 0

        episode_losses = []

        pbar = tqdm(total=total_timesteps, desc="Training DQN")

        while current_timesteps < total_timesteps:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            self.performance_traj.append(reward)
            current_timesteps += 1
            pbar.update(1)

            loss, _ = self.train_step()
            if loss is not None:
                episode_losses.append(loss)

            if done:
                mean_loss = np.mean(episode_losses) if episode_losses else None
                log_data.append({
                    "timestep": current_timesteps,
                    "reward": episode_reward,
                    "value_loss": mean_loss
                })
                episode_num += 1

                if self.verbose >= 1 and episode_num % 10 == 0:
                    mean_reward = np.mean(self.performance_traj[-10:])
                    print(f"\n[INFO] Episode {episode_num}, Timestep {current_timesteps}, Mean Reward (last 10): {mean_reward:.2f}, Last Loss: {loss:.4f if loss else None}")

                state, _ = self.env.reset()
                episode_reward = 0
                episode_losses = []

        pbar.close()
        return log_data



#class DQN(RlAlgorithm):
#    def __init__(self, env, model,  buffer_size=10000, gamma=0.99, lr=1e-3, epsilon=0.1, batch_size=64):
#        self.epsilon = epsilon
#        self.batch_size = batch_size
#        super().__init__(env, model=model, buffer_size=buffer_size, gamma=gamma, lr=lr, batch_size=batch_size)

#        self.buffer = ReplayBuffer(capacity=buffer_size)
#        self.performance_traj = []

#    def predict(self, observation, deterministic=True):
#        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
#        with torch.no_grad():
#            q_values = self.model(state_tensor)
#            return torch.argmax(q_values).item()

#    def select_action(self, state):
#        if np.random.rand() < self.epsilon:
#            return np.random.randint(self.action_size)
#        with torch.no_grad():
#            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#            q_values = self.model(state_tensor)
#            return torch.argmax(q_values).item()


#    def train_step(self):
#        if len(self.buffer) < self.batch_size:
#            return None, None

#        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

#        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
#        with torch.no_grad():
#            #mean_q_value = self.policy_network(states).mean().item()
#            next_q_values = self.model(next_states).max(1)[0]
#            targets = rewards + self.gamma * next_q_values * (1 - dones)

#        loss = F.mse_loss(q_values, targets)

#        self.optimizer.zero_grad()
#        loss.backward()
#        self.optimizer.step()

#        return loss.item(), None#, mean_q_value

#    def learn(self, total_timesteps, verbose=0):
#        log_data = [] 
        
#        state, _ = self.env.reset()
#        current_timesteps = 0
#        episode_reward = 0
#        episode_num = 0

#        episode_losses = []
#        episode_q_values = []

#        pbar = tqdm(total=total_timesteps, desc="Training DQN")

#        while current_timesteps < total_timesteps:
#            action = self.select_action(state)
#            next_state, reward, terminated, truncated, _ = self.env.step(action)
#            done = terminated or truncated

#            self.buffer.add(state, action, reward, next_state, done)
#            state = next_state
#            episode_reward += reward

#            self.performance_traj.append(reward)      

#            current_timesteps += 1
#            pbar.update(1)

#            loss, _ = self.train_step()
            
#            if loss is not None:
#                episode_losses.append(loss)
#                #episode_q_values.append(mean_q)
            
#            if done:
#                mean_loss = np.mean(episode_losses) if episode_losses else None
#                #mean_q = np.mean(episode_q_values) if episode_q_values else None

#                log_data.append({
#                    "timestep": current_timesteps,
#                    "reward": episode_reward,
#                    "value_loss": mean_loss,
#                    #"mean_q_value": mean_q
#                })
#                episode_num += 1
                
#                if verbose and episode_num % 10 == 0:
#                    mean_reward = np.mean(self.performance_traj[-10:])
#                    if loss is not None:
#                        print(f"\nEpisode {episode_num}, Timestep {current_timesteps}, Mean Reward (last 10): {mean_reward:.2f}, Last Loss: {loss:.4f}")
#                    else:
#                        print(f"\nEpisode {episode_num}, Timestep {current_timesteps}, Mean Reward (last 10): {mean_reward:.2f}")

#                state, _ = self.env.reset()
#                episode_reward = 0
#                episode_losses = []
#                episode_q_values = []

#        pbar.close()
#        return log_data
