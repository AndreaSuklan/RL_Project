import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

class ActorCritic(nn.Module):
    """    Actor-Critic model for PPO.
    This model has a shared body for both the actor and critic networks.
    The actor outputs a probability distribution over actions, while the critic outputs a state value.  
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
    """
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

