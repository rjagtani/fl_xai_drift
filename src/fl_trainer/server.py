"""
Federated Learning Server implementation with FedAvg aggregation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from pathlib import Path

from ..models import MLP, get_weights, set_weights


class FLServer:
    """
    Federated Learning server that coordinates training and aggregates updates.
    """
    
    def __init__(
        self,
        model: MLP,
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        # Training history
        self.round_losses: List[float] = []
        self.round_accuracies: List[float] = []
        self.client_losses: List[Dict[int, float]] = []
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get current global model parameters."""
        return get_weights(self.model)
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set global model parameters."""
        set_weights(self.model, parameters)
    
    def aggregate_fit(
        self,
        results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]],
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregate client model updates using FedAvg.
        
        Args:
            results: List of (parameters, n_samples, metrics) from clients
        
        Returns:
            Tuple of (aggregated parameters, aggregated metrics)
        """
        if not results:
            return self.get_parameters(), {}
        
        # Extract parameters and weights (number of samples)
        all_params = [params for params, _, _ in results]
        weights = np.array([n_samples for _, n_samples, _ in results], dtype=np.float32)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average of parameters
        aggregated_params = []
        for param_idx in range(len(all_params[0])):
            layer_params = np.array([params[param_idx] for params in all_params])
            weighted_avg = np.average(layer_params, axis=0, weights=weights)
            aggregated_params.append(weighted_avg)
        
        # Aggregate metrics
        total_samples = sum(n_samples for _, n_samples, _ in results)
        avg_loss = sum(metrics.get('loss', 0) * n_samples for _, n_samples, metrics in results) / total_samples
        
        aggregated_metrics = {
            'loss': avg_loss,
            'total_samples': total_samples,
            'n_clients': len(results),
        }
        
        # Update global model
        self.set_parameters(aggregated_params)
        
        return aggregated_params, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        results: List[Tuple[float, int, Dict[str, Any]]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate client evaluation results.
        
        Args:
            results: List of (loss, n_samples, metrics) from clients
        
        Returns:
            Tuple of (aggregated loss, aggregated metrics)
        """
        if not results:
            return float('inf'), {}
        
        total_samples = sum(n_samples for _, n_samples, _ in results)
        
        # Weighted average loss
        agg_loss = sum(loss * n_samples for loss, n_samples, _ in results) / total_samples
        
        # Weighted average accuracy
        agg_accuracy = sum(
            metrics.get('accuracy', 0) * n_samples 
            for _, n_samples, metrics in results
        ) / total_samples
        
        # Per-client losses
        client_losses = {
            metrics['client_id']: loss 
            for loss, _, metrics in results 
            if 'client_id' in metrics
        }
        
        aggregated_metrics = {
            'loss': agg_loss,
            'accuracy': agg_accuracy,
            'total_samples': total_samples,
            'n_clients': len(results),
            'client_losses': client_losses,
        }
        
        # Store history
        self.round_losses.append(agg_loss)
        self.round_accuracies.append(agg_accuracy)
        self.client_losses.append(client_losses)
        
        return agg_loss, aggregated_metrics
    
    def save_checkpoint(
        self,
        round_num: int,
        save_dir: Path,
        prefix: str = "global_model",
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            round_num: Current round number
            save_dir: Directory to save checkpoint
            prefix: Filename prefix
        
        Returns:
            Path to saved checkpoint
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_dir / f"{prefix}_round_{round_num}.pth"
        torch.save({
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'round_losses': self.round_losses,
            'round_accuracies': self.round_accuracies,
        }, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.round_losses = checkpoint.get('round_losses', [])
        self.round_accuracies = checkpoint.get('round_accuracies', [])
        
        return checkpoint
    
    def get_loss_history(self) -> List[float]:
        """Return history of aggregated losses."""
        return self.round_losses.copy()
    
    def get_client_loss_history(self) -> List[Dict[int, float]]:
        """Return history of per-client losses."""
        return self.client_losses.copy()
