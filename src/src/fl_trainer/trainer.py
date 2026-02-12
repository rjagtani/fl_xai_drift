"""
Main FL training orchestrator.
"""

from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import numpy as np
import torch
import json

from ..config import ExperimentConfig
from ..models import MLP, get_weights, set_weights
from ..data.base import BaseDataGenerator, ClientDataset
from ..data.hyperplane import HyperplaneDataGenerator
from ..data.agrawal import AgrawalDataGenerator
from ..data.wine_quality import WineQualityDataGenerator
from ..data.fed_heart import FedHeartDataGenerator
from ..data.elec2 import Elec2DataGenerator
from ..data.diabetes import DiabetesDataGenerator
from ..data.credit import CreditDataGenerator
from ..data.adult import AdultDataGenerator
from ..data.bank_marketing import BankMarketingDataGenerator
from .client import FLClient
from .server import FLServer


class FLTrainer:
    """
    Federated Learning trainer that orchestrates the full training process.
    
    Handles:
    - Data generation per round
    - Client training and evaluation
    - Server aggregation
    - Checkpoint saving
    - Loss logging for drift detection
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device('cpu')
        
        self.data_generator = self._create_data_generator()
        
        n_features = len(self.data_generator.get_feature_names())
        self.global_model = MLP(
            input_size=n_features,
            hidden_sizes=config.fl.hidden_sizes,
            n_classes=config.fl.n_classes,
        )
        
        self.server = FLServer(self.global_model, device=self.device)
        
        self.clients: Dict[int, FLClient] = {}
        for client_id in range(config.fl.n_clients):
            client_model = MLP(
                input_size=n_features,
                hidden_sizes=config.fl.hidden_sizes,
                n_classes=config.fl.n_classes,
            )
            self.clients[client_id] = FLClient(
                client_id=client_id,
                model=client_model,
                learning_rate=config.fl.learning_rate,
                momentum=config.fl.momentum,
                device=self.device,
            )
        
        self.drifted_clients = config.drift.get_drifted_clients(config.fl.n_clients)
        
        self.current_round = 0
        self.training_logs: List[Dict[str, Any]] = []
        self.global_loss_series: List[float] = []
        self.client_loss_matrix: np.ndarray = np.full(
            (config.fl.n_rounds, config.fl.n_clients), np.nan
        )
        
        self.client_confidence_matrix: np.ndarray = np.full(
            (config.fl.n_rounds, config.fl.n_clients), np.nan
        )
        self.client_weight_updates: List[Dict[int, List[np.ndarray]]] = []
        self.aggregated_weight_updates: List[List[np.ndarray]] = []
        
        self._pre_drift_data: Optional[Dict[int, ClientDataset]] = None
        self._post_drift_data: Optional[Dict[int, ClientDataset]] = None
    
    def _create_data_generator(self) -> BaseDataGenerator:
        """Create appropriate data generator based on config."""
        cfg = self.config
        
        if cfg.dataset.name == 'hyperplane':
            return HyperplaneDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                n_features=cfg.dataset.n_features,
                n_drift_features=cfg.dataset.n_drift_features,
                noise_percentage=cfg.dataset.noise,
                sigma=cfg.dataset.sigma,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
            )
        elif cfg.dataset.name == 'agrawal':
            return AgrawalDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                classification_function_pre=cfg.dataset.classification_function_pre,
                classification_function_post=cfg.dataset.classification_function_post,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
            )
        elif cfg.dataset.name == 'diabetes':
            return DiabetesDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                use_all_data_per_client=getattr(cfg.dataset, 'use_all_data_per_client', True),
            )
        elif cfg.dataset.name == 'credit':
            return CreditDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                use_all_data_per_client=getattr(cfg.dataset, 'use_all_data_per_client', True),
            )
        elif cfg.dataset.name == 'wine':
            return WineQualityDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                drift_condition_feature=cfg.dataset.drift_condition_feature,
                drift_condition_threshold=cfg.dataset.drift_condition_threshold,
                drift_flip_prob=cfg.dataset.drift_flip_prob,
                use_all_data_per_client=getattr(cfg.dataset, 'use_all_data_per_client', False),
            )
        elif cfg.dataset.name == 'fed_heart':
            return FedHeartDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                drift_condition_feature=getattr(cfg.dataset, 'drift_condition_feature', 'chol'),
                drift_condition_threshold=getattr(cfg.dataset, 'drift_condition_threshold', None),
                drift_flip_prob=getattr(cfg.dataset, 'drift_flip_prob', 0.3),
            )
        elif cfg.dataset.name == 'elec2':
            return Elec2DataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                drift_condition_feature=getattr(cfg.dataset, 'drift_condition_feature', 'nswprice'),
                drift_condition_threshold=getattr(cfg.dataset, 'drift_condition_threshold', None),
                drift_flip_prob=getattr(cfg.dataset, 'drift_flip_prob', 0.3),
            )
        elif cfg.dataset.name == 'adult':
            return AdultDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                use_all_data_per_client=getattr(cfg.dataset, 'use_all_data_per_client', True),
            )
        elif cfg.dataset.name == 'bank_marketing':
            return BankMarketingDataGenerator(
                n_clients=cfg.fl.n_clients,
                n_samples_per_client=cfg.dataset.n_samples_per_client,
                test_size=cfg.dataset.test_size,
                seed=cfg.seed,
                use_all_data_per_client=getattr(cfg.dataset, 'use_all_data_per_client', True),
            )
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    def _prepare_static_data(self):
        """Pre-generate static datasets for both phases."""
        cfg = self.config
        
        if cfg.dataset.name == 'hyperplane':
            self._pre_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                drift_magnitude=cfg.drift.drift_magnitude,
                generate_drifted=False,
            )
            self._post_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                drift_magnitude=cfg.drift.drift_magnitude,
                generate_drifted=True,
            )
        elif cfg.dataset.name in ('diabetes', 'credit'):
            self._pre_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                generate_drifted=False,
            )
            self._post_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                generate_drifted=True,
            )
        elif cfg.dataset.name == 'wine':
            self._pre_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                generate_drifted=False,
            )
            self._post_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                generate_drifted=True,
            )
        else:
            self._pre_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                generate_drifted=False,
            )
            self._post_drift_data = self.data_generator.generate_static_client_data(
                drifted_clients=self.drifted_clients,
                generate_drifted=True,
            )
    
    def get_round_data(self, round_num: int) -> Dict[int, ClientDataset]:
        """
        Get client data for a specific round.
        
        Data switches from pre-drift to post-drift after t0 for drifted clients.
        """
        if self._pre_drift_data is None:
            self._prepare_static_data()
        
        t0 = self.config.drift.t0
        
        round_data = {}
        for client_id in range(self.config.fl.n_clients):
            is_drifted = client_id in self.drifted_clients
            use_drifted_data = round_num >= t0 and is_drifted
            
            if use_drifted_data:
                round_data[client_id] = self._post_drift_data[client_id]
            else:
                round_data[client_id] = self._pre_drift_data[client_id]
        
        return round_data
    
    def train_round(self, round_num: int) -> Dict[str, Any]:
        """
        Execute one round of federated learning.
        
        Args:
            round_num: Current round number (1-indexed)
        
        Returns:
            Dictionary with round metrics
        """
        cfg = self.config
        self.current_round = round_num
        
        round_data = self.get_round_data(round_num)
        
        global_params = self.server.get_parameters()
        
        fit_results = []
        round_weight_updates = {}
        
        for client_id in range(cfg.fl.n_clients):
            client = self.clients[client_id]
            client_data = round_data[client_id]
            
            client.set_parameters(global_params)
            
            train_loader, _ = self.data_generator.create_dataloaders(
                client_data,
                batch_size=cfg.fl.batch_size,
            )
            
            params, n_samples, metrics = client.fit(
                train_loader,
                epochs=cfg.fl.local_epochs,
            )
            fit_results.append((params, n_samples, metrics))
            
            if 'weight_update' in metrics:
                round_weight_updates[client_id] = metrics['weight_update']
        
        self.client_weight_updates.append(round_weight_updates)
        
        _, fit_metrics = self.server.aggregate_fit(fit_results)
        
        if round_weight_updates:
            total_samples = sum(res[1] for res in fit_results)
            n_layers = len(next(iter(round_weight_updates.values())))
            aggregated_update = [np.zeros_like(round_weight_updates[0][i]) for i in range(n_layers)]
            for client_id, (params, n_samples, metrics) in enumerate(fit_results):
                if client_id in round_weight_updates:
                    weight = n_samples / total_samples
                    for i, delta in enumerate(round_weight_updates[client_id]):
                        aggregated_update[i] += weight * delta
            self.aggregated_weight_updates.append(aggregated_update)
        
        eval_results = []
        for client_id in range(cfg.fl.n_clients):
            client = self.clients[client_id]
            client_data = round_data[client_id]
            
            _, val_loader = self.data_generator.create_dataloaders(
                client_data,
                batch_size=cfg.fl.batch_size,
            )
            
            loss, n_samples, metrics = client.evaluate_with_model(
                self.server.model,
                val_loader,
            )
            eval_results.append((loss, n_samples, metrics))
            
            self.client_loss_matrix[round_num - 1, client_id] = loss
            
            confidence = client.compute_confidence(self.server.model, val_loader)
            self.client_confidence_matrix[round_num - 1, client_id] = confidence
        
        agg_loss, eval_metrics = self.server.aggregate_evaluate(eval_results)
        
        if np.isnan(agg_loss) or np.isinf(agg_loss):
            raise RuntimeError(
                f"NaN/Inf global loss at round {round_num}. "
                "Check data (e.g. Agrawal stream) and model (learning rate, scaling)."
            )
        
        self.global_loss_series.append(agg_loss)
        
        checkpoint_path = self.server.save_checkpoint(
            round_num,
            self.config.checkpoints_dir,
        )
        
        round_log = {
            'round': round_num,
            'global_loss': agg_loss,
            'global_accuracy': eval_metrics.get('accuracy', 0),
            'fit_loss': fit_metrics.get('loss', 0),
            'client_losses': eval_metrics.get('client_losses', {}),
            'checkpoint_path': str(checkpoint_path),
        }
        self.training_logs.append(round_log)
        
        return round_log
    
    def train(self) -> Dict[str, Any]:
        """
        Execute full federated learning training.
        
        Returns:
            Dictionary with training summary
        """
        cfg = self.config
        
        cfg.create_directories()
        
        cfg.save()
        
        print(f"Starting FL training: {cfg.experiment_name}")
        print(f"  Clients: {cfg.fl.n_clients}")
        print(f"  Rounds: {cfg.fl.n_rounds}")
        print(f"  Drift at round: {cfg.drift.t0}")
        print(f"  Drifted clients: {self.drifted_clients}")
        
        for round_num in range(1, cfg.fl.n_rounds + 1):
            round_log = self.train_round(round_num)
            
            print(
                f"Round {round_num}/{cfg.fl.n_rounds}: "
                f"Loss={round_log['global_loss']:.4f}, "
                f"Acc={round_log['global_accuracy']:.2%}"
            )
        
        logs_path = cfg.logs_dir / "training_logs.json"
        with open(logs_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        np.save(cfg.logs_dir / "global_loss_series.npy", np.array(self.global_loss_series))
        np.save(cfg.logs_dir / "client_loss_matrix.npy", self.client_loss_matrix)
        
        np.save(cfg.logs_dir / "client_confidence_matrix.npy", self.client_confidence_matrix)
        
        if self.aggregated_weight_updates:
            flattened_updates = []
            for update in self.aggregated_weight_updates:
                flat = np.concatenate([u.flatten() for u in update])
                flattened_updates.append(flat)
            np.save(cfg.logs_dir / "aggregated_weight_updates.npy", np.array(flattened_updates))
        
        if self.client_weight_updates:
            n_rounds = len(self.client_weight_updates)
            n_clients = cfg.fl.n_clients
            n_params = None
            for round_updates in self.client_weight_updates:
                if round_updates:
                    first_update = next(iter(round_updates.values()))
                    n_params = sum(u.size for u in first_update)
                    break
            
            if n_params:
                client_updates_matrix = np.full((n_rounds, n_clients, n_params), np.nan)
                for r, round_updates in enumerate(self.client_weight_updates):
                    for client_id, update in round_updates.items():
                        flat = np.concatenate([u.flatten() for u in update])
                        client_updates_matrix[r, client_id, :] = flat
                np.save(cfg.logs_dir / "client_weight_updates.npy", client_updates_matrix)
        
        summary = {
            'experiment_name': cfg.experiment_name,
            'final_loss': self.global_loss_series[-1],
            'final_accuracy': self.training_logs[-1]['global_accuracy'],
            'global_loss_series': self.global_loss_series,
            'drifted_clients': list(self.drifted_clients),
        }
        
        print(f"Training complete. Results saved to {cfg.output_dir}")
        
        return summary
    
    def load_checkpoint(self, round_num: int) -> MLP:
        """
        Load model checkpoint for a specific round.
        
        Args:
            round_num: Round number
        
        Returns:
            Loaded model
        """
        checkpoint_path = self.config.checkpoints_dir / f"global_model_round_{round_num}.pth"
        
        model = MLP(
            input_size=len(self.data_generator.get_feature_names()),
            hidden_sizes=self.config.fl.hidden_sizes,
            n_classes=self.config.fl.n_classes,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def get_client_data(self, client_id: int, round_num: int) -> ClientDataset:
        """Get data for a specific client at a specific round."""
        round_data = self.get_round_data(round_num)
        return round_data[client_id]
    
    def get_drifted_feature_indices(self) -> Set[int]:
        """Get ground truth drifted feature indices."""
        return self.data_generator.get_drifted_feature_indices()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.data_generator.get_feature_names()
