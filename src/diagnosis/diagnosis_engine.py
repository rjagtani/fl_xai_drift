"""
Diagnosis engine that orchestrates FI computation and diagnosis.
"""

from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import numpy as np
import json

from ..config import ExperimentConfig
from ..models.mlp import MLP, ModelWrapper
from ..fl_trainer.trainer import FLTrainer
from ..feature_importance import SAGEComputer, PFIComputer, SHAPComputer
from .dist_fi import DistFIDiagnosis, DistFIResult
from .delta_fi import DeltaFIDiagnosis, DeltaFIResult


class DiagnosisEngine:
    """
    Engine for computing feature importance and running diagnosis.
    
    Orchestrates:
    1. Loading model checkpoints for diagnosis window
    2. Computing FI values (SAGE, PFI, SHAP) per client per round
    3. Running Dist(FI) RDS ranking and Delta(FI) mean-change ranking
    4. Saving results
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        trainer: FLTrainer,
    ):
        self.config = config
        self.trainer = trainer
        
        # Initialize FI computers
        diag_cfg = config.diagnosis
        
        self.sage_computer = SAGEComputer(
            background_size=diag_cfg.background_size,
            use_kmeans=diag_cfg.use_kmeans_background,
            random_state=config.seed,
        ) if diag_cfg.compute_sage else None
        
        self.pfi_computer = PFIComputer(
            n_permutations=diag_cfg.n_perm,
            random_state=config.seed,
        ) if diag_cfg.compute_pfi else None
        
        self.shap_computer = SHAPComputer(
            background_size=diag_cfg.background_size,
            use_kmeans=diag_cfg.use_kmeans_background,
            random_state=config.seed,
        ) if diag_cfg.compute_shap else None
        
        # Initialize diagnosis methods
        self.dist_fi = DistFIDiagnosis()
        self.delta_fi = DeltaFIDiagnosis()
        
        # Storage for FI values
        self.fi_matrices: Dict[str, np.ndarray] = {}
    
    def determine_diagnosis_window(
        self,
        trigger_round: int,
    ) -> List[int]:
        """
        Determine rounds to include in diagnosis window.
        
        Args:
            trigger_round: Round when drift was triggered
        
        Returns:
            List of round numbers for diagnosis
        """
        window_size = self.config.diagnosis.window_size
        start_round = max(1, trigger_round - window_size)
        
        return list(range(start_round, trigger_round + 1))
    
    def compute_fi_for_round(
        self,
        round_num: int,
        track_sage_times: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute all FI values for a single round.
        
        Args:
            round_num: Round number
            track_sage_times: If True, record per-client SAGE wall-clock
                times and attach them under key ``'_sage_client_times'``
                in the returned dict (list of floats, one per client;
                NaN if computation failed).
        
        Returns:
            Dictionary mapping method -> (n_clients, n_features) array.
            When *track_sage_times* is True an extra key
            ``'_sage_client_times'`` is included.
        """
        import time as _time

        cfg = self.config
        n_clients = cfg.fl.n_clients
        n_features = len(self.trainer.get_feature_names())
        
        # Load model checkpoint
        model = self.trainer.load_checkpoint(round_num)
        model.eval()
        model_wrapper = ModelWrapper(model)
        
        # Initialize result arrays
        results: Dict[str, Any] = {}
        if self.sage_computer:
            results['sage'] = np.full((n_clients, n_features), np.nan)
        if self.pfi_computer:
            results['pfi'] = np.full((n_clients, n_features), np.nan)
        if self.shap_computer:
            results['shap'] = np.full((n_clients, n_features), np.nan)

        sage_client_times: List[float] = []
        
        # Compute FI for each client
        for client_id in range(n_clients):
            client_data = self.trainer.get_client_data(client_id, round_num)
            
            # Create background set
            X_background = self.sage_computer.create_background_set(
                client_data.X_train,
                max_size=cfg.diagnosis.background_size,
            ) if self.sage_computer else None
            
            # Create estimation set (full val or max_samples, whichever is lower)
            max_est = cfg.diagnosis.estimation_max_samples
            X_est, y_est = self.sage_computer.create_estimation_set(
                client_data.X_val,
                client_data.y_val,
                max_samples=max_est,
            ) if self.sage_computer else (None, None)
            
            if X_est is None and self.pfi_computer:
                X_est, y_est = self.pfi_computer.create_estimation_set(
                    client_data.X_val,
                    client_data.y_val,
                    max_samples=max_est,
                )
            
            # Compute SAGE
            if self.sage_computer:
                t0_sage = _time.perf_counter()
                try:
                    sage_values = self.sage_computer.compute(
                        model_wrapper.predict_proba,
                        X_background,
                        X_est,
                        y_est,
                    )
                    results['sage'][client_id] = sage_values
                    sage_client_times.append(_time.perf_counter() - t0_sage)
                except Exception as e:
                    sage_client_times.append(float('nan'))
                    print(f"SAGE computation failed for client {client_id}, round {round_num}: {e}")
            
            # Compute PFI
            if self.pfi_computer:
                try:
                    pfi_values = self.pfi_computer.compute(
                        model_wrapper.predict_proba,
                        None,
                        X_est,
                        y_est,
                    )
                    results['pfi'][client_id] = pfi_values
                except Exception as e:
                    print(f"PFI computation failed for client {client_id}, round {round_num}: {e}")
            
            # Compute SHAP
            if self.shap_computer:
                try:
                    shap_values = self.shap_computer.compute(
                        model_wrapper.predict_proba,
                        X_background,
                        X_est,
                        y_est,
                    )
                    results['shap'][client_id] = shap_values
                except Exception as e:
                    print(f"SHAP computation failed for client {client_id}, round {round_num}: {e}")

        if track_sage_times and sage_client_times:
            results['_sage_client_times'] = sage_client_times
        
        return results
    
    def compute_fi_for_window(
        self,
        diagnosis_rounds: List[int],
    ) -> Dict[str, Any]:
        """
        Compute FI values for all rounds in diagnosis window.
        
        Args:
            diagnosis_rounds: List of round numbers
        
        Returns:
            Dictionary mapping method -> (n_rounds, n_clients, n_features) array.
            Also includes ``'_sage_trigger_client_times'`` (list of floats)
            with per-client SAGE wall-clock seconds for the trigger round.
        """
        cfg = self.config
        n_clients = cfg.fl.n_clients
        n_features = len(self.trainer.get_feature_names())
        n_rounds = len(diagnosis_rounds)
        
        # Initialize matrices
        methods = []
        if self.sage_computer:
            methods.append('sage')
        if self.pfi_computer:
            methods.append('pfi')
        if self.shap_computer:
            methods.append('shap')
        
        fi_matrices: Dict[str, Any] = {
            method: np.full((n_rounds, n_clients, n_features), np.nan)
            for method in methods
        }
        
        sage_trigger_client_times: List[float] = []

        # Compute FI for each round
        for r_idx, round_num in enumerate(diagnosis_rounds):
            is_trigger = (r_idx == n_rounds - 1)
            print(f"Computing FI for round {round_num}/{diagnosis_rounds[-1]}...")
            round_results = self.compute_fi_for_round(
                round_num, track_sage_times=is_trigger)
            
            for method, values in round_results.items():
                if method.startswith('_'):
                    continue  # skip meta keys
                fi_matrices[method][r_idx] = values

            if is_trigger and '_sage_client_times' in round_results:
                sage_trigger_client_times = round_results['_sage_client_times']

        fi_matrices['_sage_trigger_client_times'] = sage_trigger_client_times
        self.fi_matrices = fi_matrices
        return fi_matrices
    
    def _compute_client_weights(self, round_num: int) -> np.ndarray:
        """
        Compute per-client probability weights from training data sizes.

        Returns:
            Array of shape (n_clients,) with w_i = n_i / N.
        """
        n_clients = self.config.fl.n_clients
        round_data = self.trainer.get_round_data(round_num)
        sizes = np.array([len(round_data[i].X_train) for i in range(n_clients)],
                         dtype=np.float64)
        return sizes / sizes.sum()

    def run_diagnosis(
        self,
        trigger_round: int,
        client_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run complete diagnosis pipeline.
        
        Args:
            trigger_round: Round when drift was triggered
            client_weights: Optional per-client probability weights (sum to 1).
                If None, weights are computed from client training data sizes.
        
        Returns:
            Dictionary with diagnosis results
        """
        # Determine diagnosis window
        diagnosis_rounds = self.determine_diagnosis_window(trigger_round)
        print(f"Diagnosis window: rounds {diagnosis_rounds[0]} to {diagnosis_rounds[-1]}")
        
        # Compute client weights if not provided
        if client_weights is None:
            client_weights = self._compute_client_weights(diagnosis_rounds[0])
        print(f"  Client weights: {client_weights}")
        
        # Compute FI values (timed)
        import time as _time
        _fi_start = _time.perf_counter()
        fi_matrices = self.compute_fi_for_window(diagnosis_rounds)
        fi_compute_time = _time.perf_counter() - _fi_start

        # Extract SAGE trigger-round per-client times (meta key)
        sage_trigger_client_times = fi_matrices.pop(
            '_sage_trigger_client_times', [])

        # Run Dist(FI) diagnosis (client-weighted RDS ranking at trigger round)
        dist_results = self.dist_fi.diagnose(fi_matrices,
                                             client_weights=client_weights)
        
        # Run Delta(FI) diagnosis (client-weighted |mean_FI(trigger) - mean_FI(prev)|)
        delta_results = self.delta_fi.diagnose(fi_matrices,
                                               client_weights=client_weights)
        
        # Combine results
        all_results = {
            'trigger_round': trigger_round,
            'diagnosis_rounds': diagnosis_rounds,
            'dist_fi': dist_results,
            'delta_fi': delta_results,
            'fi_matrices': fi_matrices,
            'sage_trigger_client_times': sage_trigger_client_times,
            'fi_compute_time_s': fi_compute_time,
        }
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save diagnosis results to a specific directory (public API)."""
        self._save_results(results, output_dir=output_dir)

    def _save_results(self, results: Dict[str, Any], output_dir: Path = None):
        """Save diagnosis results to files."""
        if output_dir is None:
            output_dir = self.config.diagnosis_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FI matrices
        for method, matrix in results['fi_matrices'].items():
            np.save(output_dir / f"fi_matrix_{method}.npy", matrix)
        
        # Save rankings
        rankings = {}
        
        for name, result in results['dist_fi'].items():
            rankings[name] = {
                'ranking': result.feature_ranking.tolist(),
                'scores': result.rds_scores.tolist(),
            }
        
        for name, result in results.get('delta_fi', {}).items():
            rankings[name] = {
                'ranking': result.feature_ranking.tolist(),
                'scores': result.delta_scores.tolist(),
            }
        
        with open(output_dir / "rankings.json", 'w') as f:
            json.dump(rankings, f, indent=2)
        
        # Save metadata
        metadata = {
            'trigger_round': results['trigger_round'],
            'diagnosis_rounds': results['diagnosis_rounds'],
            'methods': list(results['fi_matrices'].keys()),
        }
        
        with open(output_dir / "diagnosis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_feature_rankings(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Extract all feature rankings from diagnosis results.
        
        Returns:
            Dictionary mapping diagnosis variant name to ranking array
        """
        rankings = {}
        
        for name, result in results['dist_fi'].items():
            rankings[name] = result.feature_ranking
        
        for name, result in results.get('delta_fi', {}).items():
            rankings[name] = result.feature_ranking
        
        return rankings
