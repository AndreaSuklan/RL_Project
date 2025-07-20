import os
import io
import zipfile
import pickle
import torch
from abc import ABC, abstractmethod


class RlAlgorithm(ABC):
    """
    Abstract base class for reinforcement learning algorithms.
    Handles shared initialization, and provides a unified save/load interface using .zip files.
    """
    def __init__(self, env, model, buffer_size, gamma=0.99, lr=1e-3, batch_size=64):
        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.state_size = env.observation_space.shape[0],
        self.action_size = env.action_space.n

        self.model = model 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_model_name(self):
        return self.model.__class__.__name__

    def _get_model_state_dict(self):
        return {f"{self._get_model_name()}": self.model.state_dict()}

    def _set_model_state_dict(self, state_dicts):
        self.model.load_state_dict(state_dicts[f"{self._get_model_name()}"])

    def _build_from_hyperparameters(cls, env, hyperparams):
        return cls(env=env, **hyperparams)

    def _get_optimizer_state_dict(self):
        """Return optimizer state dicts if needed (override in subclasses)."""
        return {"optimizer": self.optimizer.state_dict()}

    def _set_optimizer_state_dict(self, state_dicts):
        """Load optimizer state dicts (override in subclasses if needed)."""
        self.optimizer.load_state_dict(state_dicts["optimizer"])

    def get_hyperparameters(self):
        """Return a dictionary of hyperparameters for saving."""
        return {
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "n_epochs": self.n_epochs,
            "gae_lambda": self.gae_lambda if hasattr(self.model, 'gae_lambda') else None,
            "epsilon": self.epsilon if hasattr(self.model, 'epsilon') else None,
            "clip_epsilon": self.clip_epsilon if hasattr(self.model, 'clip_epsilon') else None,
        }

    def save(self, path):
        """Save the model and hyperparameters to a zip file."""
        if not path.endswith(".zip"):
            path += ".zip"

        tmp_dir = "_tmp_rl_save"
        os.makedirs(tmp_dir, exist_ok=True)

        # Save model(s)
        for name, state in self._get_model_state_dict().items():
            torch.save(state, os.path.join(tmp_dir, f"{name}.pth"))

        # Save optimizer(s)
        for name, state in self._get_optimizer_state_dict().items():
            torch.save(state, os.path.join(tmp_dir, f"{name}_opt.pth"))

        # Save metadata
        metadata = self.get_hyperparameters()
        with open(os.path.join(tmp_dir, "meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        # Write all to zip
        with zipfile.ZipFile(path, 'w') as zipf:
            for fname in os.listdir(tmp_dir):
                zipf.write(os.path.join(tmp_dir, fname), fname)

        # Clean up
        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)

        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path, env):
        """Load an agent from a zip file."""
        import tempfile
        if not path.endswith(".zip"):
            path += ".zip"

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(path, 'r') as zipf:
                zipf.extractall(tmp_dir)

            # Load metadata and rebuild agent
            with open(os.path.join(tmp_dir, "meta.pkl"), "rb") as f:
                hyperparameters = pickle.load(f)

            agent = cls._build_from_hyperparameters(env, hyperparameters)

            # Load models
            model_state_dict = {}
            for fname in os.listdir(tmp_dir):
                if fname.endswith(".pth") and not fname.endswith("_opt.pth"):
                    key = fname.replace(".pth", "")
                    model_state_dict[key] = torch.load(os.path.join(tmp_dir, fname))
            agent._set_model_state_dict(model_state_dict)

            # Load optimizers (if applicable)
            opt_state_dict = {}
            for fname in os.listdir(tmp_dir):
                if fname.endswith("_opt.pth"):
                    key = fname.replace("_opt.pth", "")
                    opt_state_dict[key] = torch.load(os.path.join(tmp_dir, fname))
            agent._set_optimizer_state_dict(opt_state_dict)

        print(f"Agent loaded from {path}")
        return agent

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def predict(self, observation, deterministic=True):
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass

