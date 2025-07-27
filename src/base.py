import os
import inspect
import zipfile
import pickle
import torch
import tempfile
from abc import ABC, abstractmethod
import numpy as np
import random

def set_seed(seed, env=None):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if env is not None:
        env.action_space.seed(seed)


class RlAlgorithm(ABC):
    """
    Abstract base class for reinforcement learning algorithms.
    Handles shared initialization, and provides a unified save/load interface using .zip files.
    """
    def __init__(self, env, model, buffer_size, gamma=0.99, lr=1e-3, batch_size=64, verbose=0):
        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.verbose=verbose
        self.model = model 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_model_name(self):
        return self.model.__class__.__name__

    def _get_model_state_dict(self):
        return {self._get_model_name(): self.model.state_dict()}

    def _set_model_state_dict(self, state_dicts):
        self.model.load_state_dict(state_dicts[self._get_model_name()])

    @classmethod
    def _build_from_hyperparameters(cls, env, hyperparams):
        """
        Builds an agent from hyperparameters, filtering out unexpected arguments.
        """
        constructor_args = inspect.signature(cls.__init__).parameters.keys()
        filtered_hyperparams = {
            key: value for key, value in hyperparams.items() if key in constructor_args
        }
        return cls(env=env, **filtered_hyperparams)

    def _get_optimizer_state_dict(self):
        return {"optimizer": self.optimizer.state_dict()}

    def _set_optimizer_state_dict(self, state_dicts):
        self.optimizer.load_state_dict(state_dicts["optimizer"])

    def get_hyperparameters(self):
        """Return a dictionary of hyperparameters for saving."""
        return {
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epochs": getattr(self, 'n_epochs', None),
            "gae_lambda": getattr(self, 'gae_lambda', None),
            "epsilon": getattr(self, 'epsilon', None),
            "clip_epsilon": getattr(self, 'clip_epsilon', None),
        }
  
    def save(self, path):
        if not path.endswith(".zip"):
            path += ".zip"

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(self.model.state_dict(), os.path.join(tmp_dir, "model.pth"))
            torch.save(self.optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pth"))
            hyperparameters = self.get_hyperparameters()
            with open(os.path.join(tmp_dir, "hyperparameters.pkl"), "wb") as f:
                pickle.dump(hyperparameters, f)
            
            with zipfile.ZipFile(path, 'w') as zipf:
                for fname in os.listdir(tmp_dir):
                    zipf.write(os.path.join(tmp_dir, fname), fname)
        
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path, env, model):
        if not path.endswith(".zip"):
            path += ".zip"

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(path, 'r') as zipf:
                zipf.extractall(tmp_dir)
            
            with open(os.path.join(tmp_dir, "hyperparameters.pkl"), "rb") as f:
                hyperparameters = pickle.load(f)

            if model is None:
                raise ValueError("A 'model_class' must be provided to load the agent.")
            
            constructor_args = inspect.signature(cls.__init__).parameters.keys()
            filtered_hyperparams = {
                key: value for key, value in hyperparameters.items() 
                if key in constructor_args
            }
            
            agent = cls(env=env, model=model, **filtered_hyperparams)
            agent.model.load_state_dict(torch.load(os.path.join(tmp_dir, "model.pth")))
            agent.optimizer.load_state_dict(torch.load(os.path.join(tmp_dir, "optimizer.pth")))

        print(f"Agent loaded from {path}")
        return agent

    @abstractmethod
    def predict(self, observation, deterministic=True):
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass
