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
    def __init__(self, env, buffer, buffer_size, gamma=0.99, lr=1e-3, batch_size=64):
        self.env = env
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lr = lr

    @abstractmethod
    def _get_model_state_dict(self):
        """Return a dictionary of model state dicts to save."""
        pass

    @abstractmethod
    def _set_model_state_dict(self, state_dicts):
        """Load model state dicts from dictionary."""
        pass

    @abstractmethod
    def _build_from_hyperparameters(cls, env, hyperparameters):
        """Create a new instance from hyperparameters (used during load)."""
        pass

    def _get_optimizer_state_dict(self):
        """Return optimizer state dicts if needed (override in subclasses)."""
        return {}

    def _set_optimizer_state_dict(self, state_dicts):
        """Load optimizer state dicts (override in subclasses if needed)."""
        pass

    @abstractmethod
    def get_hyperparameters(self):
        """Return a dictionary of hyperparameters for saving."""
        pass

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
    def predict(self, observation, deterministic=True):
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass

