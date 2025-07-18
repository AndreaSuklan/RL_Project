# train_dqn.py
from stable_baselines3 import DQN
from environment import HillClimbEnv
from consts import ACTION_SIZE, INPUT_SIZE

# --- Configuration --- (should be useless here really)
# MODEL_NAME = "dqn_hill_climb"
# TOTAL_TIMESTEPS = 300_000  # DQN often requires more samples
LOG_DIR = "./logs/dqn_logs/"
# MODEL_DIR = "./models/"


# MY IMPLEMENTATION

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

import pickle
import zipfile
import os

def create_agent(env, log_dir=None):
    """
    Creates and returns a DQN agent.
    This function currently uses the Stable-Baselines3 library,
    but you can replace it with your own DQN implementation later.
    
    Args:
        env: The Gymnasium environment.
        log_dir: The directory to save TensorBoard logs.
        
    Returns:
        A DQN model instance.
    """
    print("Creating QDN agent...")
    # agent = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     buffer_size=50000,
    #     learning_starts=1000,
    #     batch_size=32,
    #     gamma=0.99,
    #     train_freq=4,
    #     gradient_steps=1,
    #     target_update_interval=1000,
    #     exploration_fraction=0.2,
    #     exploration_final_eps=0.05
    # )
    agent = MyDQN(env) 

    return agent

class SimpleNet(nn.Module):
    #TODO: replace this placeholder with an actually serious network (e.g the one described for ATARI or some other).
    def __init__(self):
        super(SimpleNet, self).__init__()
        # input has dimension 4
        self.fc1 = nn.Linear(INPUT_SIZE, 16) # input should be 7 i,e state vector returned by env.reset()  
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, ACTION_SIZE) # output should be one of 3 actions: # 0:No-op, 1:Accelerate Left, 2:Accelerate Right


    def forward(self, input):
        l1 = F.relu(self.fc1(input))
        l2 = F.relu(self.fc2(l1))
        l3 = F.relu(self.fc3(l2))
        output = self.fc4(l3)
        return output

class SimpleDQN(nn.Module):
    def __init__(self, input_dim=19, output_dim=3):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)      # Adapted from final hidden layer
        self.fc2 = nn.Linear(512, output_dim)     # Output Q-values for each action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # No activation, outputs raw Q-values


class MyDQN():
    def __init__(self, 
                 # space_size,
                 env,
                 policy,
                 action_size=3, 
                 gamma=1, 
                 lr_v=0.01, 
                 epsilon=0.05,
                 log_dir=LOG_DIR, 
                 debug):
        """
        Calculates optimal policy using off-policy Deep Q-Network control
        Evaluates Q-value for (S,A) pairs, using one-step updates.
        """        
        self.env = env
        # the discount factor
        self.gamma = gamma
        self.epsilon = epsilon
        # size of system
        # We don't truly need the space_size anymore, we work in feature space!
        # self.input_size = space_size
       
        # action size
        self.action_size = action_size
       
        # Model is as such:
        self.policy = policy # i,e the neural network taking in the observations and giving back the best policy for it.

        # the learning rate
        self.lr_v = lr_v
        self.our_SGD = optim.SGD(self.policy.parameters(), lr=lr_v)
        self.performance_traj_NeuralNetwork = [] 

               
    # -------------------   
    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the EXPERIENCED action in the new state  <- Q(S_new, A_new).
        """
       
        # Just need a different shape for the Neural Network
        x = torch.tensor(s)
        new_x = torch.tensor(new_s)
       
        # I apply the Neural Network to get the current estimation of the Q(s,a) and Q(new_s,new_a)
       
        self.our_SGD.zero_grad()
        Q_approx = self.policy(x)[a]
        # new_Q_approx = self.Q_NN_approx(new_x)[new_a].detach() OLD (SARSA from exercise)
        new_Q_approx = self.policy(new_x).max().detach() # NEW (Q-Learning: taking the max regaardless of the action returned by env.step())

       
        # Now Q_approx (and new_Q_approx) are value for the pair (s,a) (and (new_s, new_a))
       
        if done:
            # in SARSA it was
            # deltaQ = (r + 0 - self.Qvalues[ (*s, a) ])
            # deltaQ = (r + 0 - Q_approx)
            target = r + 0
        else:
            # in SARSA it was
            # deltaQ = (r + 
            #          self.gamma * self.Qvalues[ (*new_s, new_a) ] 
            #                     - self.Qvalues[ (*    s,     a) ])
            # deltaQ = (r + 
            #          self.gamma * new_Q_approx
            #                     -     Q_approx)
            target = r + self.gamma * new_Q_approx
           
        # in SARSA it was
        # self.Qvalues[ (*s, a) ] += self.lr_v * deltaQ
        # 
        # in Linear Approximation it was 
        # w <- deltaQ(s,a|w) nabla Q(s,a|w)
        loss = ((target - Q_approx)*(target-Q_approx)).sum()
       
        # Here the update with all the gradients is done via the torch library
        loss.backward()
        self.our_SGD.step() 
       
    # ---------------------
    def get_action_epsilon_greedy(self, s, eps):
        """
        Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        """
        ran = np.random.rand()
       
        s = torch.tensor(s)
        # Q value as defined by Neural Networks
        # It is a vector with dimension equal to the action space:
        Qvalue_approx = self.policy(s).detach().numpy()
       
       
        if (ran < eps):
            # probability is uniform for all actions!
            prob_actions = np.ones(self.action_size) / self.action_size 
       
        else:
            # I find the best Qvalue
            best_value = np.max( Qvalue_approx )
           
            # There could be actions with equal value! 
            best_actions = ( Qvalue_approx == best_value )


            # best_actions is 
            # *True* if the value is equal to the best (possibly ties)
            # *False* if the action is suboptimal
            prob_actions = best_actions / np.sum(best_actions)
       
        # take one action from the array of actions with the probabilities as defined above.
        assert self.action_size == len(prob_actions), f"Error: expected the action and probability vector to have the same size, but action_size={self.action_size}, len(p)={len(prob_actions)}"
        a = np.random.choice(self.action_size, p=prob_actions)
        return a 
       
    def greedy_policy(self, s):
        s = torch.tensor(s)
        Qvalue_approx = self.policy(s).detach().numpy()
        greedy_pol = np.argmax(Qvalue_approx)
        return greedy_pol

    def learn(self, total_timesteps=200_000, n_episodes=2000, gamma=1, lr_v=0.0025):

        self.performance_traj_NeuralNetwork = np.zeros(n_episodes)
        count = 0
        # for i in range(n_episodes) :
        for i in tqdm(range(n_episodes), desc="Training Episodes"):

           
            s, _ = self.env.reset()
            a = self.get_action_epsilon_greedy(s, self.epsilon)
            term = False
            trunc = False
           
            if (i%250==0):
                print(i, end='\t')
           
            step = 0
            if debug:
                print(f"episode {i}:state={s}, action={a}...") 
            while (not term) and (not trunc):


                # keeping track for convergence
                count += 1
               
                # Evolve one step
                new_s, r, term, trunc, info = self.env.step(a)
               
                done = (term or trunc)

                # Keeps track of performance for each episode
                self.performance_traj_NeuralNetwork[i] += r
                # Choose new action index
                new_a = self.get_action_epsilon_greedy(new_s, self.epsilon)
                #print(s,act,a, r,new_s,new_a, done, ' Qvalue ', SARSA.Qvalues[(*s,)])
                # Single update with (S, A, R', S', A')
               
                step += 1
                if (done and step<499):
                    r = -5
                else:
                    r = 0
               
                self.single_step_update(s, a, r, new_s, new_a, done)
               
                a = new_a
                s = new_s

    def predict(self, observation):
        state = None
        action = self.policy(observation) 
        return action, state

    def plot(self):
        if self.performance_traj_NeuralNetwork is not None:
            fig, ax = plt.subplots()  # Create a figure and an axes.
            ax.plot(np.arange(n_episodes), self.performance_traj_NeuralNetwork, label='MyDQN_Neural Network')
            ax.legend()
        else:
            raise ValueError(f"The object `self.performance_traj_NeuralNetwork` is None and cannot be plotted. Have you trained the model?")


    def save(self, path="my_dqn_model.zip"):
        # Create a temporary directory to store files
        tmp_dir = "tmp_dqn_save"
        os.makedirs(tmp_dir, exist_ok=True)

        # Save model weights
        torch.save(self.policy.state_dict(), os.path.join(tmp_dir, "policy.pth"))

        # Save optimizer
        torch.save(self.our_SGD.state_dict(), os.path.join(tmp_dir, "optimizer.pth"))

        # Save other relevant Python objects (like epsilon, gamma, etc.)
        metadata = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "action_size": self.action_size,
            "lr_v": self.lr_v,
            "performance_traj": self.performance_traj_NeuralNetwork,
        }
        with open(os.path.join(tmp_dir, "meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        # Zip all the files
        with zipfile.ZipFile(path, 'w') as zipf:
            for fname in ["policy.pth", "optimizer.pth", "meta.pkl"]:
                zipf.write(os.path.join(tmp_dir, fname), fname)

        # Clean up temp directory
        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)

    @classmethod
    def load(cls, path="my_dqn_model.zip", env=None, policy_class=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract the zip archive
            with zipfile.ZipFile(path, 'r') as zipf:
                zipf.extractall(tmp_dir)

            # Load metadata
            with open(os.path.join(tmp_dir, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)

            # Create the policy and DQN object
            policy = policy_class() if policy_class is not None else None
            model = cls(
                env=env,
                policy=policy,
                action_size=meta["action_size"],
                gamma=meta["gamma"],
                lr_v=meta["lr_v"],
                epsilon=meta["epsilon"],
            )
            model.performance_traj_NeuralNetwork = meta["performance_traj"]

            # Load weights
            model.policy.load_state_dict(torch.load(os.path.join(tmp_dir, "policy.pth")))
            model.our_SGD.load_state_dict(torch.load(os.path.join(tmp_dir, "optimizer.pth")))

        return model

