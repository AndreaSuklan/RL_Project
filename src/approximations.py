import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def polynomial_features(x, degree=3):
    """
    Compute polynomial features up to the given degree.
    x: Tensor of shape [batch_size, input_dim]
    Returns: Tensor of shape [batch_size, num_poly_features]
    """
    # Start with degree 1 features
    features = [x]
    
    for d in range(2, degree + 1):
        features.append(x ** d)
    
    return torch.cat(features, dim=0)  # Concatenate along feature dimension

class Linear(nn.Module):
    """Linear function approximator for DQN."""
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class Polynomial(nn.Module):
    """Polynomial function approximator for DQN."""
    def __init__(self, input_dim, output_dim, degree=3):
        super(Polynomial, self).__init__()
        self.degree = degree
        poly_dim = input_dim * degree  # e.g., x, x^2, ..., x^degree
        self.linear = nn.Linear(poly_dim, output_dim)

    def forward(self, x):
        poly_x = polynomial_features(x, self.degree)
        return self.linear(poly_x)

