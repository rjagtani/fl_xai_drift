"""
Federated Learning Client implementation.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import MLP, get_weights, set_weights
from ..data.base import ClientDataset, TorchDataset


class FLClient:
    """
    Federated Learning client that performs local training and evaluation.
    """
    
    def __init__(
        self,
        client_id: int,
        model: MLP,
        learning_rate: float = 0.05,
        momentum: float = 0.5,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.device = device or torch.device('cpu')
        
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from server."""
        set_weights(self.model, parameters)
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get current model parameters."""
        return get_weights(self.model)
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Perform local training on client data.
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of local epochs
        
        Returns:
            Tuple of (updated parameters, number of samples, metrics dict)
        """
        initial_params = self.get_parameters()
        
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / max(n_batches, 1)
        n_samples = len(train_loader.dataset)
        
        final_params = self.get_parameters()
        weight_update = [fp - ip for fp, ip in zip(final_params, initial_params)]
        
        metrics = {
            'loss': avg_loss,
            'client_id': self.client_id,
            'n_samples': n_samples,
            'weight_update': weight_update,
        }
        
        return self.get_parameters(), n_samples, metrics
    
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (loss, number of samples, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * len(y_batch)
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'client_id': self.client_id,
            'n_samples': total,
        }
        
        return avg_loss, total, metrics
    
    def compute_confidence(
        self,
        model: MLP,
        val_loader: DataLoader,
    ) -> float:
        """
        Compute average model confidence on validation data.
        
        Confidence = average max softmax probability across all samples.
        This is the original signal used by CDA-FedAvg (Casado et al., 2021).
        
        Args:
            model: Model to evaluate
            val_loader: DataLoader for validation data
        
        Returns:
            Average confidence score (0-1)
        """
        model.eval()
        model.to(self.device)
        
        total_confidence = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                
                total_confidence += max_probs.sum().item()
                total_samples += len(y_batch)
        
        avg_confidence = total_confidence / max(total_samples, 1)
        return avg_confidence
    
    def evaluate_with_model(
        self,
        model: MLP,
        val_loader: DataLoader,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate a given model (e.g., global model) on client's validation data.
        
        Args:
            model: Model to evaluate
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (loss, number of samples, metrics dict)
        """
        model.eval()
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item() * len(y_batch)
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'client_id': self.client_id,
            'n_samples': total,
        }
        
        return avg_loss, total, metrics
