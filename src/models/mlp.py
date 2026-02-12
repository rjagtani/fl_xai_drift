"""
Multi-Layer Perceptron model for classification.
"""

from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary/multi-class classification.
    
    Architecture: input -> [hidden layers with ReLU] -> output
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64],
        n_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.n_classes = n_classes
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append((f'fc{i+1}', nn.Linear(prev_size, hidden_size)))
            layers.append((f'relu{i+1}', nn.ReLU()))
            if dropout > 0:
                layers.append((f'dropout{i+1}', nn.Dropout(dropout)))
            prev_size = hidden_size
        
        layers.append(('output', nn.Linear(prev_size, n_classes)))
        
        self.network = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as a list of numpy arrays."""
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Set model weights from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def create_model(
    input_size: int,
    hidden_sizes: List[int] = [128, 64],
    n_classes: int = 2,
) -> MLP:
    """Factory function to create an MLP model."""
    return MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        n_classes=n_classes,
    )


class ModelWrapper:
    """
    Wrapper for using PyTorch model with numpy arrays.
    
    Useful for SAGE/SHAP which expect numpy-based function interfaces.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        Args:
            X: Input array of shape (n_samples, n_features)
        
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model.predict_proba(X_tensor)
            return probs.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        Args:
            X: Input array of shape (n_samples, n_features)
        
        Returns:
            Array of shape (n_samples,) with class labels
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model.predict(X_tensor)
            return preds.cpu().numpy()
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba."""
        return self.predict_proba(X)
