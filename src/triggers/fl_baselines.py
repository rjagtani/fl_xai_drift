"""
Federated Learning drift detection baselines.

Implements three methods from the literature using ORIGINAL signals:
1. FLASH (Panchal et al., ICML 2023) - Gradient disparity on aggregated weight updates
2. CDA-FedAvg (Casado et al., MTA 2021) - CUSUM on model confidence scores
3. Manias et al. (GLOBECOM 2021) - PCA + KMeans on model weight updates

System-level evaluation:
- False Positive: ANY client triggers before t0
- Success: ALL ground-truth drifted clients trigger at or after t0

Citations:
- FLASH: Panchal et al., "Flash: Concept Drift Adaptation in FL", ICML 2023
- CDA-FedAvg: Casado et al., "Concept-drift-aware federated averaging", MTA 2021
- Manias: Manias et al., "Concept Drift Detection in Federated Networked Systems", GLOBECOM 2021
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


@dataclass
class ClientDriftResult:
    """Result from per-client drift detection."""
    client_id: int
    triggered: bool
    trigger_round: Optional[int]
    score: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class SystemDriftResult:
    """Result from system-level drift detection."""
    triggered: bool  # True if system successfully detected drift
    trigger_round: Optional[int]  # Round when system triggered (last drifted client)
    detection_delay: Optional[int]  # trigger_round - t0
    method: str
    # Per-client results (for client-level methods)
    client_results: Dict[int, ClientDriftResult] = field(default_factory=dict)
    # System-level evaluation
    has_false_positive: bool = False  # Any client triggered before t0
    all_drifted_detected: bool = False  # All drifted clients triggered at or after t0
    detected_drifted_clients: Set[int] = field(default_factory=set)
    missed_drifted_clients: Set[int] = field(default_factory=set)
    false_positive_clients: Set[int] = field(default_factory=set)  # Triggered before t0


class FLASHDetector:
    """
    FLASH drift detector using gradient disparity on aggregated weight updates.
    
    Original method (Panchal et al., ICML 2023):
    - Track gradient disparity: ||(Δ^(r))² - v^(r)|| where Δ is the aggregated
      update and v is the rolling second moment (exponential moving average)
    - Large disparity indicates concept drift
    
    This is a SYSTEM-LEVEL detector.
    """
    
    def __init__(
        self,
        calibration_start: int = 21,
        calibration_end: int = 40,
        alpha: float = 3.0,
        beta2: float = 0.9,  # EMA decay for second moment
        confirm_consecutive: int = 3,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.alpha = alpha
        self.beta2 = beta2
        self.confirm_consecutive = confirm_consecutive
    
    def detect(
        self,
        aggregated_updates: np.ndarray,
        t0: int,
    ) -> SystemDriftResult:
        """
        Run FLASH detection on aggregated weight updates.
        
        Args:
            aggregated_updates: (n_rounds, n_params) flattened aggregated weight updates
            t0: Ground truth drift onset round
        
        Returns:
            SystemDriftResult
        """
        n_rounds = len(aggregated_updates)
        
        # Compute L2 norm of each update (proxy for gradient magnitude)
        update_norms = np.linalg.norm(aggregated_updates, axis=1)
        
        # Track second moment (rolling EMA) and disparity
        v = np.zeros(n_rounds)
        disparity = np.zeros(n_rounds)
        
        calibration_disparities = []
        consecutive_count = 0
        streak_start_round = None
        triggered = False
        trigger_round = None
        threshold = None
        
        for r in range(1, n_rounds):
            # Update second moment: v^(r) = β₂·v^(r-1) + (1-β₂)·||Δ^(r)||²
            v[r] = self.beta2 * v[r-1] + (1 - self.beta2) * (update_norms[r] ** 2)
            
            # Gradient disparity: ||(Δ^(r))² - v^(r)||
            disparity[r] = abs(update_norms[r] ** 2 - v[r])
            
            # Collect calibration statistics
            if self.calibration_start <= r <= self.calibration_end:
                calibration_disparities.append(disparity[r])
                continue
            
            # Set threshold from calibration
            if threshold is None and r > self.calibration_end:
                if len(calibration_disparities) >= 5:
                    cal_mu = np.mean(calibration_disparities)
                    cal_sigma = np.std(calibration_disparities, ddof=1)
                    if cal_sigma < 1e-10:
                        cal_sigma = 0.001
                    threshold = cal_mu + self.alpha * cal_sigma
                else:
                    threshold = 0.01  # Fallback
            
            # Detection with k-consecutive
            if threshold is not None and r > self.calibration_end:
                exceeded = disparity[r] > threshold
                if exceeded:
                    if consecutive_count == 0:
                        streak_start_round = r
                        consecutive_count = 1
                    else:
                        consecutive_count += 1
                    if consecutive_count >= self.confirm_consecutive:
                        triggered = True
                        trigger_round = streak_start_round
                        break
                else:
                    consecutive_count = 0
                    streak_start_round = None
        
        # Evaluate: FP if triggered before t0
        has_fp = triggered and trigger_round is not None and trigger_round < t0
        success = triggered and trigger_round is not None and trigger_round >= t0
        
        delay = None
        if trigger_round is not None and trigger_round >= t0:
            delay = trigger_round - t0
        
        return SystemDriftResult(
            triggered=success,
            trigger_round=trigger_round if success else None,
            detection_delay=delay,
            method='flash',
            has_false_positive=has_fp,
            all_drifted_detected=success,
        )


class CDAFedAvgDetector:
    """
    CDA-FedAvg drift detector using model confidence scores.
    
    Original method (Casado et al., MTA 2021):
    - Each client monitors model confidence (max posterior probability)
    - Confidence drop indicates drift (CUSUM-style test on confidence)
    - Client triggers when confidence drops significantly below calibration baseline
    
    System-level evaluation:
    - FP if ANY client triggers before t0
    - Success if ALL drifted clients trigger at or after t0
    """
    
    def __init__(
        self,
        calibration_start: int = 21,
        calibration_end: int = 40,
        alpha: float = 3.0,  # threshold = μ - α·σ (confidence DROPS indicate drift)
        confirm_consecutive: int = 3,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.alpha = alpha
        self.confirm_consecutive = confirm_consecutive
    
    def detect(
        self,
        confidence_matrix: np.ndarray,
        drifted_clients: Set[int],
        t0: int,
    ) -> SystemDriftResult:
        """
        Run CDA-FedAvg detection on per-client confidence scores.
        
        Args:
            confidence_matrix: (n_rounds, n_clients) confidence scores
            drifted_clients: Ground truth set of drifted client IDs
            t0: Ground truth drift onset round
        
        Returns:
            SystemDriftResult with per-client results and evaluation
        """
        n_rounds, n_clients = confidence_matrix.shape
        client_results = {}
        
        # Per-client detection
        for client_id in range(n_clients):
            client_confidence = confidence_matrix[:, client_id]
            result = self._detect_client(client_id, client_confidence)
            client_results[client_id] = result
        
        # System-level evaluation
        return self._evaluate_system(client_results, drifted_clients, t0)
    
    def _detect_client(self, client_id: int, confidence_series: np.ndarray) -> ClientDriftResult:
        """Run detection for a single client using confidence scores."""
        n_rounds = len(confidence_series)
        
        # Calibration: compute baseline confidence
        cal_confidences = confidence_series[self.calibration_start:self.calibration_end+1]
        cal_confidences = cal_confidences[~np.isnan(cal_confidences)]
        
        if len(cal_confidences) < 5:
            return ClientDriftResult(
                client_id=client_id,
                triggered=False,
                trigger_round=None,
            )
        
        cal_mu = np.mean(cal_confidences)
        cal_sigma = np.std(cal_confidences, ddof=1)
        if cal_sigma < 1e-10:
            cal_sigma = 0.001
        
        # Threshold: confidence drops BELOW this indicates drift
        threshold = cal_mu - self.alpha * cal_sigma
        
        # Detection with k-consecutive (look for confidence BELOW threshold)
        consecutive_count = 0
        streak_start_round = None
        triggered = False
        trigger_round = None
        
        for r in range(self.calibration_end + 1, n_rounds):
            if np.isnan(confidence_series[r]):
                continue
            
            # Confidence DROP indicates drift
            below_threshold = confidence_series[r] < threshold
            if below_threshold:
                if consecutive_count == 0:
                    streak_start_round = r
                    consecutive_count = 1
                else:
                    consecutive_count += 1
                if consecutive_count >= self.confirm_consecutive:
                    triggered = True
                    trigger_round = streak_start_round
                    break
            else:
                consecutive_count = 0
                streak_start_round = None
        
        return ClientDriftResult(
            client_id=client_id,
            triggered=triggered,
            trigger_round=trigger_round,
            threshold=threshold,
        )
    
    def _evaluate_system(
        self,
        client_results: Dict[int, ClientDriftResult],
        drifted_clients: Set[int],
        t0: int,
    ) -> SystemDriftResult:
        """
        Evaluate system-level detection.
        
        - FP if ANY client triggers before t0
        - Success if ALL drifted clients trigger at or after t0
        """
        # Check for false positives (any client triggering before t0)
        false_positive_clients = set()
        for client_id, result in client_results.items():
            if result.triggered and result.trigger_round is not None:
                if result.trigger_round < t0:
                    false_positive_clients.add(client_id)
        
        has_fp = len(false_positive_clients) > 0
        
        # Check drifted clients: did they all trigger at or after t0?
        detected_drifted = set()
        missed_drifted = set()
        
        for client_id in drifted_clients:
            if client_id in client_results:
                result = client_results[client_id]
                if result.triggered and result.trigger_round is not None and result.trigger_round >= t0:
                    detected_drifted.add(client_id)
                else:
                    missed_drifted.add(client_id)
            else:
                missed_drifted.add(client_id)
        
        all_drifted_detected = (detected_drifted == drifted_clients) and len(drifted_clients) > 0
        
        # System trigger round: max trigger round of drifted clients (if all detected)
        system_trigger_round = None
        if all_drifted_detected:
            trigger_rounds = [
                client_results[cid].trigger_round
                for cid in drifted_clients
                if client_results[cid].trigger_round is not None
            ]
            if trigger_rounds:
                system_trigger_round = max(trigger_rounds)
        
        # System success: all drifted detected AND no false positives
        system_success = all_drifted_detected and not has_fp
        
        delay = None
        if system_trigger_round is not None:
            delay = system_trigger_round - t0
        
        return SystemDriftResult(
            triggered=system_success,
            trigger_round=system_trigger_round if system_success else None,
            detection_delay=delay,
            method='cda_fedavg',
            client_results=client_results,
            has_false_positive=has_fp,
            all_drifted_detected=all_drifted_detected,
            detected_drifted_clients=detected_drifted,
            missed_drifted_clients=missed_drifted,
            false_positive_clients=false_positive_clients,
        )


class ManiasPCAKMeansDetector:
    """
    Manias et al. (GLOBECOM 2021) detector using PCA + KMeans on weight updates.
    
    Original method:
    - Each client collects model weight updates under "normal" conditions
    - Apply PCA to reduce dimensionality of weight updates
    - Apply KMeans (k=2) clustering
    - Compute Euclidean distance between cluster centers
    - Drift if distance > baseline μ + 3σ (or < μ - 3σ)
    
    System-level evaluation:
    - FP if ANY client triggers before t0
    - Success if ALL drifted clients trigger at or after t0
    """
    
    def __init__(
        self,
        calibration_start: int = 21,
        calibration_end: int = 40,
        n_components: int = 10,  # PCA components
        alpha: float = 3.0,  # threshold = μ ± α·σ
        detection_window: int = 5,  # Rounds of updates to cluster
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.n_components = n_components
        self.alpha = alpha
        self.detection_window = detection_window
    
    def detect(
        self,
        client_weight_updates: np.ndarray,
        drifted_clients: Set[int],
        t0: int,
    ) -> SystemDriftResult:
        """
        Run Manias-style detection on per-client weight updates.
        
        Args:
            client_weight_updates: (n_rounds, n_clients, n_params) weight updates
            drifted_clients: Ground truth set of drifted client IDs
            t0: Ground truth drift onset round
        
        Returns:
            SystemDriftResult with per-client results and evaluation
        """
        n_rounds, n_clients, n_params = client_weight_updates.shape
        
        # Phase 1: Calibration - compute baseline cluster distances
        calibration_distances = {}
        for client_id in range(n_clients):
            cal_updates = client_weight_updates[self.calibration_start:self.calibration_end+1, client_id, :]
            # Remove NaN rows
            valid_mask = ~np.isnan(cal_updates).any(axis=1)
            cal_updates = cal_updates[valid_mask]
            
            if len(cal_updates) >= 6:  # Need enough points for PCA + KMeans
                dist = self._compute_cluster_distance(cal_updates)
                calibration_distances[client_id] = dist
        
        if not calibration_distances:
            return SystemDriftResult(
                triggered=False,
                trigger_round=None,
                detection_delay=None,
                method='manias_pca_kmeans',
                has_false_positive=False,
                all_drifted_detected=False,
            )
        
        # Compute system-wide baseline threshold
        all_distances = list(calibration_distances.values())
        baseline_mu = np.mean(all_distances)
        baseline_sigma = np.std(all_distances, ddof=1)
        if baseline_sigma < 1e-10:
            baseline_sigma = 0.001
        
        lower_bound = baseline_mu - self.alpha * baseline_sigma
        upper_bound = baseline_mu + self.alpha * baseline_sigma
        
        # Phase 2: Detection
        client_results = {}
        for client_id in range(n_clients):
            result = self._detect_client(
                client_id,
                client_weight_updates[:, client_id, :],
                lower_bound,
                upper_bound,
            )
            client_results[client_id] = result
        
        # System-level evaluation
        return self._evaluate_system(client_results, drifted_clients, t0)
    
    def _compute_cluster_distance(self, updates: np.ndarray) -> float:
        """
        Apply PCA + KMeans and compute cluster center distance.
        
        Args:
            updates: (n_samples, n_params) weight updates
        
        Returns:
            Euclidean distance between two cluster centers
        """
        if len(updates) < 4:
            return 0.0
        
        # PCA
        n_comp = min(self.n_components, updates.shape[1], updates.shape[0] - 1)
        if n_comp < 2:
            return 0.0
        
        pca = PCA(n_components=n_comp)
        try:
            X_reduced = pca.fit_transform(updates)
        except Exception:
            return 0.0
        
        # KMeans clustering
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(X_reduced)
            
            # Distance between cluster centers
            centers = kmeans.cluster_centers_
            distance = np.linalg.norm(centers[0] - centers[1])
            return distance
        except Exception:
            return 0.0
    
    def _detect_client(
        self,
        client_id: int,
        weight_updates: np.ndarray,
        lower_bound: float,
        upper_bound: float,
    ) -> ClientDriftResult:
        """Run detection for a single client."""
        n_rounds = len(weight_updates)
        
        triggered = False
        trigger_round = None
        last_distance = None
        
        # Check at intervals after calibration
        for r in range(self.calibration_end + self.detection_window, n_rounds, self.detection_window):
            # Get recent updates
            start_idx = max(self.calibration_end + 1, r - self.detection_window)
            recent_updates = weight_updates[start_idx:r+1]
            
            # Remove NaN rows
            valid_mask = ~np.isnan(recent_updates).any(axis=1)
            recent_updates = recent_updates[valid_mask]
            
            if len(recent_updates) < 4:
                continue
            
            # Compute cluster distance
            distance = self._compute_cluster_distance(recent_updates)
            last_distance = distance
            
            # Check if outside baseline bounds
            if distance < lower_bound or distance > upper_bound:
                triggered = True
                trigger_round = r
                break
        
        return ClientDriftResult(
            client_id=client_id,
            triggered=triggered,
            trigger_round=trigger_round,
            score=last_distance,
            threshold=upper_bound,
        )
    
    def _evaluate_system(
        self,
        client_results: Dict[int, ClientDriftResult],
        drifted_clients: Set[int],
        t0: int,
    ) -> SystemDriftResult:
        """Evaluate system-level detection."""
        # Check for false positives
        false_positive_clients = set()
        for client_id, result in client_results.items():
            if result.triggered and result.trigger_round is not None:
                if result.trigger_round < t0:
                    false_positive_clients.add(client_id)
        
        has_fp = len(false_positive_clients) > 0
        
        # Check drifted clients
        detected_drifted = set()
        missed_drifted = set()
        
        for client_id in drifted_clients:
            if client_id in client_results:
                result = client_results[client_id]
                if result.triggered and result.trigger_round is not None and result.trigger_round >= t0:
                    detected_drifted.add(client_id)
                else:
                    missed_drifted.add(client_id)
            else:
                missed_drifted.add(client_id)
        
        all_drifted_detected = (detected_drifted == drifted_clients) and len(drifted_clients) > 0
        
        # System trigger round
        system_trigger_round = None
        if all_drifted_detected:
            trigger_rounds = [
                client_results[cid].trigger_round
                for cid in drifted_clients
                if client_results[cid].trigger_round is not None
            ]
            if trigger_rounds:
                system_trigger_round = max(trigger_rounds)
        
        # System success
        system_success = all_drifted_detected and not has_fp
        
        delay = None
        if system_trigger_round is not None:
            delay = system_trigger_round - t0
        
        return SystemDriftResult(
            triggered=system_success,
            trigger_round=system_trigger_round if system_success else None,
            detection_delay=delay,
            method='manias_pca_kmeans',
            client_results=client_results,
            has_false_positive=has_fp,
            all_drifted_detected=all_drifted_detected,
            detected_drifted_clients=detected_drifted,
            missed_drifted_clients=missed_drifted,
            false_positive_clients=false_positive_clients,
        )


class FLBaselineMultiDetector:
    """
    Combined detector running all FL baseline methods.
    
    Runs FLASH, CDA-FedAvg, and Manias detectors using original signals.
    """
    
    def __init__(
        self,
        calibration_start: int = 21,
        calibration_end: int = 40,
        alpha: float = 3.0,
        confirm_consecutive: int = 3,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        
        self.flash = FLASHDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            alpha=alpha,
            confirm_consecutive=confirm_consecutive,
        )
        
        self.cda_fedavg = CDAFedAvgDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            alpha=alpha,
            confirm_consecutive=confirm_consecutive,
        )
        
        self.manias = ManiasPCAKMeansDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            alpha=alpha,
        )
    
    def detect(
        self,
        aggregated_updates: np.ndarray,
        confidence_matrix: np.ndarray,
        client_weight_updates: np.ndarray,
        drifted_clients: Set[int],
        t0: int,
    ) -> Dict[str, SystemDriftResult]:
        """
        Run all FL baseline detectors using original signals.
        
        Args:
            aggregated_updates: (n_rounds, n_params) for FLASH
            confidence_matrix: (n_rounds, n_clients) for CDA-FedAvg
            client_weight_updates: (n_rounds, n_clients, n_params) for Manias
            drifted_clients: Ground truth set of drifted client IDs
            t0: Ground truth drift onset round
        
        Returns:
            Dict mapping method name to SystemDriftResult
        """
        results = {}
        
        # FLASH (system-level, uses aggregated weight updates)
        if aggregated_updates is not None and len(aggregated_updates) > 0:
            results['flash'] = self.flash.detect(aggregated_updates, t0)
        else:
            results['flash'] = SystemDriftResult(
                triggered=False, trigger_round=None, detection_delay=None,
                method='flash', has_false_positive=False, all_drifted_detected=False
            )
        
        # CDA-FedAvg (client-level, uses confidence scores)
        if confidence_matrix is not None and len(confidence_matrix) > 0:
            results['cda_fedavg'] = self.cda_fedavg.detect(confidence_matrix, drifted_clients, t0)
        else:
            results['cda_fedavg'] = SystemDriftResult(
                triggered=False, trigger_round=None, detection_delay=None,
                method='cda_fedavg', has_false_positive=False, all_drifted_detected=False
            )
        
        # Manias (client-level, uses weight updates)
        if client_weight_updates is not None and len(client_weight_updates) > 0:
            results['manias'] = self.manias.detect(client_weight_updates, drifted_clients, t0)
        else:
            results['manias'] = SystemDriftResult(
                triggered=False, trigger_round=None, detection_delay=None,
                method='manias_pca_kmeans', has_false_positive=False, all_drifted_detected=False
            )
        
        return results
