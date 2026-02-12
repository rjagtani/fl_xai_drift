"""
Drift detection methods: RDS, CUSUM, Page-Hinkley, ADWIN, KSWIN.

All detectors share a common Phase-I / Phase-II framework (standard SPC):
  Phase I  (calibration): rounds [cal_start, cal_end] — estimate null parameters.
  Phase II (monitoring):  rounds > cal_end — detect drift with fixed thresholds.

Threshold strategy (unified across methods):
  - RDS / ADWIN / KSWIN: θ = μ̂ + k·σ̂ where k is derived from Bonferroni
    correction: α = p / M, k = Φ⁻¹(1 − α), with p = 0.05 (FWER target)
    and M = N − cal_end (number of detection rounds).
  - CUSUM / Page-Hinkley: ARL-based design.  Reference value k_ref = 0.5 σ₀
    (optimised for 1σ shift), decision interval h ≈ 7 σ₀ (ARL₀ ≈ 3 000).

Uses River library implementations for CUSUM and Page-Hinkley where available,
wrapped with calibration and k-consecutive confirmation logic.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
from scipy.stats import wasserstein_distance, norm
from river.drift import PageHinkley as RiverPageHinkley


def bonferroni_alpha(n_rounds: int, calibration_end: int, p: float = 0.05) -> float:
    """Compute per-round FPR via Bonferroni: α = p / M where M = n_rounds - cal_end."""
    M = max(1, n_rounds - calibration_end)
    return p / M


def bonferroni_k(n_rounds: int, calibration_end: int, p: float = 0.05) -> float:
    """Compute threshold multiplier k = Φ⁻¹(1 − α) via Bonferroni correction."""
    alpha = bonferroni_alpha(n_rounds, calibration_end, p)
    return float(norm.ppf(1.0 - alpha))


@dataclass
class DriftDetectionResult:
    """Result from drift detection."""
    triggered: bool
    trigger_round: Optional[int]
    detection_delay: Optional[int]
    method: str
    score: Optional[float] = None
    all_scores: Optional[List[float]] = None
    threshold: Optional[float] = None
    threshold_series: Optional[List[float]] = None


def compute_rds(pre_values: np.ndarray, post_values: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute Relative Distribution Shift using Wasserstein distance.
    
    RDS = W(pre, post) / (std(pre) + epsilon)
    """
    if len(pre_values) < 2 or len(post_values) < 2:
        return 0.0
    wd_score = wasserstein_distance(pre_values.flatten(), post_values.flatten())
    std_pre = np.std(pre_values, ddof=1) + epsilon
    return wd_score / std_pre


class RDSDetector:
    """
    RDS (Relative Distribution Shift) detector with Phase-I calibration threshold.
    
    Threshold = μ̂ + k·σ̂ from calibration, with k derived via Bonferroni
    correction (FWER ≤ p over M detection rounds).  Fixed after calibration.
    
    Requires `confirm_consecutive` consecutive exceedances before triggering.
    """
    
    def __init__(
        self,
        warmup_rounds: int = 40,
        calibration_start: int = 41,
        calibration_end: int = 100,
        window_size: int = 5,
        alpha: float = 3.4,
        min_instances: int = 5,
        confirm_consecutive: int = 3,
        require_loss_increase: bool = True,
        use_fixed_threshold: bool = True,
    ):
        self.warmup_rounds = warmup_rounds
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.window_size = window_size
        self.alpha = alpha
        self.min_instances = min_instances
        self.confirm_consecutive = confirm_consecutive
        self.require_loss_increase = require_loss_increase
        self.use_fixed_threshold = use_fixed_threshold
        
        self.rds_scores: List[float] = []
        self.calibration_scores: List[float] = []
        self.threshold: Optional[float] = None
        self.threshold_series: List[float] = []
        self.triggered = False
        self.trigger_round: Optional[int] = None
        self.trigger_score: Optional[float] = None
    
    def reset(self):
        """Reset detector state."""
        self.rds_scores = []
        self.calibration_scores = []
        self.threshold = None
        self.threshold_series = []
        self.triggered = False
        self.trigger_round = None
        self.trigger_score = None
    
    def detect(
        self,
        loss_matrix: np.ndarray,
        aggregated_loss: Optional[np.ndarray] = None,
        t0: Optional[int] = None,
    ) -> DriftDetectionResult:
        """
        Run RDS detection on loss matrix.
        
        Args:
            loss_matrix: Per-client losses, shape (n_rounds, n_clients)
            aggregated_loss: Optional aggregated loss per round for trend check
            t0: Ground truth drift onset (for delay calculation)
        
        Returns:
            DriftDetectionResult
        """
        self.reset()
        n_rounds = loss_matrix.shape[0]
        
        if aggregated_loss is None:
            aggregated_loss = np.mean(loss_matrix, axis=1)
        
        max_rds = 0.0
        max_rds_round = None
        
        consecutive_count = 0
        streak_start_round = None
        frozen_reference_window = None
        
        for round_idx in range(self.warmup_rounds, n_rounds):
            current_loss = loss_matrix[round_idx, :]
            
            if frozen_reference_window is not None:
                pre_window = frozen_reference_window
            else:
                start_idx = max(0, round_idx - self.window_size)
                pre_window = loss_matrix[start_idx:round_idx, :].flatten()
            
            rds = compute_rds(pre_window, current_loss)
            self.rds_scores.append(rds)
            
            if rds > max_rds:
                max_rds = rds
                max_rds_round = round_idx
            
            if self.calibration_start <= round_idx <= self.calibration_end:
                self.calibration_scores.append(rds)
                self.threshold_series.append(None)
                continue
            
            if self.threshold is None and round_idx > self.calibration_end:
                if len(self.calibration_scores) >= self.min_instances:
                    cal_mu = np.mean(self.calibration_scores)
                    cal_sigma = np.sqrt(max(np.var(self.calibration_scores, ddof=1), 1e-10))
                    self.threshold = cal_mu + self.alpha * cal_sigma
                else:
                    cal_mu = np.mean(self.rds_scores)
                    cal_sigma = np.sqrt(max(np.var(self.rds_scores, ddof=1), 1e-10)) if len(self.rds_scores) > 1 else 0.001
                    self.threshold = cal_mu + self.alpha * cal_sigma
            
            if self.threshold is not None and round_idx > self.calibration_end:
                self.threshold_series.append(self.threshold)
            
            if self.threshold is not None and round_idx > self.calibration_end:
                if rds > self.threshold:
                    if consecutive_count == 0:
                        loss_increasing = True
                        if self.require_loss_increase and round_idx > 0:
                            loss_increasing = aggregated_loss[round_idx] > aggregated_loss[round_idx - 1]
                        
                        if loss_increasing:
                            streak_start_round = round_idx
                            consecutive_count = 1
                            start_idx = max(0, round_idx - self.window_size)
                            frozen_reference_window = loss_matrix[start_idx:round_idx, :].flatten().copy()
                    else:
                        consecutive_count += 1
                    
                    if consecutive_count >= self.confirm_consecutive:
                        self.triggered = True
                        self.trigger_round = streak_start_round
                        self.trigger_score = rds
                        break
                else:
                    consecutive_count = 0
                    streak_start_round = None
                    frozen_reference_window = None
        
        if not self.triggered and max_rds_round is not None:
            self.trigger_round = max_rds_round
            self.trigger_score = max_rds
        
        delay = None
        if self.trigger_round is not None and t0 is not None:
            delay = self.trigger_round - t0
        
        return DriftDetectionResult(
            triggered=self.triggered,
            trigger_round=self.trigger_round,
            detection_delay=delay,
            method='rds',
            score=self.trigger_score,
            all_scores=self.rds_scores.copy(),
            threshold=self.threshold,
            threshold_series=self.threshold_series.copy(),
        )


class CUSUMDetector:
    """
    CUSUM detector using River's PageHinkley (which implements CUSUM).
    
    ARL-based design (Phase I / Phase II, standard SPC):
    - Phase I: estimate σ₀ from calibration window.
    - Phase II: River PH with reference value k_ref·σ₀ and decision
      interval h·σ₀, where k_ref = 0.5 (optimised for 1σ shift) and
      h ≈ 7 (giving ARL₀ ≈ 3 000).
    """
    
    def __init__(
        self,
        calibration_start: int = 41,
        calibration_end: int = 100,
        confirm_consecutive: int = 3,
        min_calibration_samples: int = 10,
        mode: str = "up",
        cusum_k_ref: float = 0.5,
        cusum_h: float = 7.0,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.confirm_consecutive = confirm_consecutive
        self.min_calibration_samples = min_calibration_samples
        self.mode = mode
        self.cusum_k_ref = cusum_k_ref
        self.cusum_h = cusum_h
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.detector = None
        self.triggered = False
        self.trigger_round = None
        self.calibration_losses = []
        self.tuned_delta = None
        self.tuned_threshold = None
    
    def detect(
        self,
        loss_series: np.ndarray,
        warmup_rounds: int = 20,
        t0: Optional[int] = None,
    ) -> DriftDetectionResult:
        """
        Run CUSUM detection using River's PageHinkley.
        
        Args:
            loss_series: Global loss per round
            warmup_rounds: Rounds to skip before starting
            t0: Ground truth drift onset
        
        Returns:
            DriftDetectionResult
        """
        self.reset()
        n_rounds = len(loss_series)
        
        for round_idx in range(warmup_rounds, n_rounds):
            x = loss_series[round_idx]
            
            if round_idx <= self.calibration_end:
                if round_idx >= self.calibration_start:
                    self.calibration_losses.append(x)
                continue
            
            if self.detector is None and round_idx > self.calibration_end:
                if len(self.calibration_losses) >= self.min_calibration_samples:
                    cal_std = np.std(self.calibration_losses, ddof=1)
                    if cal_std < 1e-10:
                        cal_std = 0.001
                    self.tuned_delta = self.cusum_k_ref * cal_std
                    self.tuned_threshold = self.cusum_h * cal_std
                else:
                    self.tuned_delta = self.cusum_k_ref * 0.01
                    self.tuned_threshold = self.cusum_h * 0.01
                
                self.detector = RiverPageHinkley(
                    min_instances=1,
                    delta=self.tuned_delta,
                    threshold=self.tuned_threshold,
                    alpha=0.9999,
                    mode=self.mode,
                )
            
            if self.detector is not None:
                self.detector.update(x)
                
                if self.detector.drift_detected:
                    self.triggered = True
                    self.trigger_round = round_idx
                    break
        
        delay = None
        if self.trigger_round is not None and t0 is not None:
            delay = self.trigger_round - t0
        
        return DriftDetectionResult(
            triggered=self.triggered,
            trigger_round=self.trigger_round,
            detection_delay=delay,
            method='cusum',
            threshold=self.tuned_threshold,
        )


class PageHinkleyDetector:
    """
    Page-Hinkley detector using River's implementation.
    
    ARL-based design (same as CUSUM), two-sided detection:
    - Phase I: estimate σ₀ from calibration window.
    - Phase II: River PH with delta = k_ref·σ₀, threshold = h·σ₀.
    """
    
    def __init__(
        self,
        calibration_start: int = 41,
        calibration_end: int = 100,
        confirm_consecutive: int = 3,
        min_calibration_samples: int = 10,
        mode: str = "both",
        cusum_k_ref: float = 0.5,
        cusum_h: float = 7.0,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.confirm_consecutive = confirm_consecutive
        self.min_calibration_samples = min_calibration_samples
        self.mode = mode
        self.cusum_k_ref = cusum_k_ref
        self.cusum_h = cusum_h
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.detector = None
        self.triggered = False
        self.trigger_round = None
        self.calibration_losses = []
        self.tuned_delta = None
        self.tuned_threshold = None
    
    def detect(
        self,
        loss_series: np.ndarray,
        warmup_rounds: int = 20,
        t0: Optional[int] = None,
    ) -> DriftDetectionResult:
        """
        Run Page-Hinkley detection using River's implementation.
        
        Args:
            loss_series: Global loss per round
            warmup_rounds: Rounds to skip before starting
            t0: Ground truth drift onset
        
        Returns:
            DriftDetectionResult
        """
        self.reset()
        n_rounds = len(loss_series)
        
        for round_idx in range(warmup_rounds, n_rounds):
            x = loss_series[round_idx]
            
            if round_idx <= self.calibration_end:
                if round_idx >= self.calibration_start:
                    self.calibration_losses.append(x)
                continue
            
            if self.detector is None and round_idx > self.calibration_end:
                if len(self.calibration_losses) >= self.min_calibration_samples:
                    cal_std = np.std(self.calibration_losses, ddof=1)
                    if cal_std < 1e-10:
                        cal_std = 0.001
                    self.tuned_delta = self.cusum_k_ref * cal_std
                    self.tuned_threshold = self.cusum_h * cal_std
                else:
                    self.tuned_delta = self.cusum_k_ref * 0.01
                    self.tuned_threshold = self.cusum_h * 0.01
                
                self.detector = RiverPageHinkley(
                    min_instances=1,
                    delta=self.tuned_delta,
                    threshold=self.tuned_threshold,
                    alpha=0.9999,
                    mode=self.mode,
                )
            
            if self.detector is not None:
                self.detector.update(x)
                
                if self.detector.drift_detected:
                    self.triggered = True
                    self.trigger_round = round_idx
                    break
        
        delay = None
        if self.trigger_round is not None and t0 is not None:
            delay = self.trigger_round - t0
        
        return DriftDetectionResult(
            triggered=self.triggered,
            trigger_round=self.trigger_round,
            detection_delay=delay,
            method='page_hinkley',
            threshold=self.tuned_threshold,
        )


class ADWINDetector:
    """
    ADWIN-style detector adapted for loss series with calibration threshold.
    
    Adaptation: compute ADWIN-style statistic during the full calibration
    period to estimate null (μ̂, σ̂), then set θ = μ̂ + k·σ̂ (Bonferroni k).
    Trigger when statistic exceeds threshold for k-consecutive rounds.
    """
    
    def __init__(
        self,
        calibration_start: int = 41,
        calibration_end: int = 100,
        alpha: float = 3.4,
        delta: float = 0.01,
        window_size: int = 20,
        confirm_consecutive: int = 3,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.alpha = alpha
        self.delta = delta
        self.window_size = window_size
        self.confirm_consecutive = confirm_consecutive
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.triggered = False
        self.trigger_round = None
        self.threshold = None
        self.statistic_series = []
    
    def _compute_adwin_statistic(self, window: np.ndarray) -> float:
        """
        Compute ADWIN-style statistic: max difference between sub-window means.
        
        Finds the cut point that maximizes |mean(W0) - mean(W1)|.
        """
        n = len(window)
        if n < 4:
            return 0.0
        
        max_diff = 0.0
        for cut in range(2, n - 2):
            w0 = window[:cut]
            w1 = window[cut:]
            diff = abs(np.mean(w0) - np.mean(w1))
            pooled_std = np.sqrt((np.var(w0) * len(w0) + np.var(w1) * len(w1)) / n)
            if pooled_std > 1e-10:
                diff = diff / pooled_std
            if diff > max_diff:
                max_diff = diff
        
        return max_diff
    
    def detect(
        self,
        aggregated_loss: np.ndarray,
        warmup_rounds: int,
        t0: Optional[int] = None,
    ) -> DriftDetectionResult:
        """
        Run ADWIN-style detection on aggregated loss.
        
        Args:
            aggregated_loss: Global loss per round
            warmup_rounds: Rounds before detection starts
            t0: Ground truth drift onset
        
        Returns:
            DriftDetectionResult
        """
        self.reset()
        n_rounds = len(aggregated_loss)
        
        calibration_stats = []
        calibration_rounds = []
        consecutive_count = 0
        streak_start_round = None
        frozen_base_window = None
        
        for r in range(warmup_rounds + self.window_size, n_rounds):
            if frozen_base_window is not None:
                window = np.concatenate([frozen_base_window, aggregated_loss[streak_start_round:r + 1]])
            else:
                window = aggregated_loss[r - self.window_size:r + 1]
            
            stat = self._compute_adwin_statistic(window)
            self.statistic_series.append(stat)
            
            if self.calibration_start <= r <= self.calibration_end:
                calibration_stats.append(stat)
                continue
            
            if self.threshold is None and r > self.calibration_end:
                if len(calibration_stats) >= 5:
                    cal_mu = np.mean(calibration_stats)
                    cal_sigma = np.std(calibration_stats, ddof=1)
                    if cal_sigma < 1e-10:
                        cal_sigma = 0.001
                    self.threshold = cal_mu + self.alpha * cal_sigma
                else:
                    self.threshold = 2.0
            
            if self.threshold is not None and r > self.calibration_end:
                exceeded = stat > self.threshold
                if exceeded:
                    if consecutive_count == 0:
                        streak_start_round = r
                        consecutive_count = 1
                        frozen_base_window = aggregated_loss[r - self.window_size:r].copy()
                    else:
                        consecutive_count += 1
                    if consecutive_count >= self.confirm_consecutive:
                        self.triggered = True
                        self.trigger_round = streak_start_round
                        break
                else:
                    consecutive_count = 0
                    streak_start_round = None
                    frozen_base_window = None
        
        delay = None
        if self.trigger_round is not None and t0 is not None:
            delay = self.trigger_round - t0
        
        return DriftDetectionResult(
            triggered=self.triggered,
            trigger_round=self.trigger_round,
            detection_delay=delay,
            method='adwin',
            threshold=self.threshold,
        )


class KSWINDetector:
    """
    KSWIN-style detector using Kolmogorov-Smirnov test with calibration threshold.
    
    Adaptation: compute KS statistic during the full calibration period to
    estimate null (μ̂, σ̂), then set θ = μ̂ + k·σ̂ (Bonferroni k).
    Trigger when KS statistic exceeds threshold for k-consecutive rounds.
    """
    
    def __init__(
        self,
        calibration_start: int = 41,
        calibration_end: int = 100,
        alpha: float = 3.4,
        window_size: int = 20,
        confirm_consecutive: int = 3,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.alpha = alpha
        self.window_size = window_size
        self.confirm_consecutive = confirm_consecutive
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.triggered = False
        self.trigger_round = None
        self.threshold = None
        self.ks_series = []
    
    def _compute_ks_statistic(self, ref_window: np.ndarray, test_window: np.ndarray) -> float:
        """
        Compute Kolmogorov-Smirnov statistic between two windows.
        
        Returns the maximum absolute difference between CDFs.
        """
        from scipy import stats
        if len(ref_window) < 3 or len(test_window) < 3:
            return 0.0
        
        ks_stat, _ = stats.ks_2samp(ref_window, test_window)
        return ks_stat
    
    def detect(
        self,
        aggregated_loss: np.ndarray,
        warmup_rounds: int,
        t0: Optional[int] = None,
    ) -> DriftDetectionResult:
        """
        Run KSWIN-style detection on aggregated loss.
        
        Args:
            aggregated_loss: Global loss per round
            warmup_rounds: Rounds before detection starts
            t0: Ground truth drift onset
        
        Returns:
            DriftDetectionResult
        """
        self.reset()
        n_rounds = len(aggregated_loss)
        
        calibration_stats = []
        consecutive_count = 0
        streak_start_round = None
        
        start_round = warmup_rounds + 2 * self.window_size
        
        frozen_ref_window = None
        
        for r in range(start_round, n_rounds):
            if frozen_ref_window is not None:
                ref_window = frozen_ref_window
            else:
                ref_window = aggregated_loss[r - 2 * self.window_size:r - self.window_size]
            
            test_window = aggregated_loss[r - self.window_size + 1:r + 1]
            
            ks_stat = self._compute_ks_statistic(ref_window, test_window)
            self.ks_series.append(ks_stat)
            
            if self.calibration_start <= r <= self.calibration_end:
                calibration_stats.append(ks_stat)
                continue
            
            if self.threshold is None and r > self.calibration_end:
                if len(calibration_stats) >= 5:
                    cal_mu = np.mean(calibration_stats)
                    cal_sigma = np.std(calibration_stats, ddof=1)
                    if cal_sigma < 1e-10:
                        cal_sigma = 0.05
                    self.threshold = cal_mu + self.alpha * cal_sigma
                else:
                    self.threshold = 0.5
            
            if self.threshold is not None and r > self.calibration_end:
                exceeded = ks_stat > self.threshold
                if exceeded:
                    if consecutive_count == 0:
                        streak_start_round = r
                        consecutive_count = 1
                        frozen_ref_window = aggregated_loss[r - 2 * self.window_size:r - self.window_size].copy()
                    else:
                        consecutive_count += 1
                    if consecutive_count >= self.confirm_consecutive:
                        self.triggered = True
                        self.trigger_round = streak_start_round
                        break
                else:
                    consecutive_count = 0
                    streak_start_round = None
                    frozen_ref_window = None
        
        delay = None
        if self.trigger_round is not None and t0 is not None:
            delay = self.trigger_round - t0
        
        return DriftDetectionResult(
            triggered=self.triggered,
            trigger_round=self.trigger_round,
            detection_delay=delay,
            method='kswin',
            threshold=self.threshold,
        )


class MultiMethodDetector:
    """
    Combined drift detector using RDS, CUSUM, Page-Hinkley, ADWIN, and KSWIN.
    
    Unified Phase-I / Phase-II framework:
    - RDS, ADWIN, KSWIN: Bonferroni-corrected threshold θ = μ̂ + k·σ̂
      with k = z_{1 − p/M}, p = 0.05, M = N − cal_end.
    - CUSUM, Page-Hinkley: ARL-based design with k_ref = 0.5, h = 7
      (ARL₀ ≈ 3000 for M ≈ 100 detection rounds).
    """
    
    def __init__(
        self,
        warmup_rounds: int = 40,
        calibration_start: int = 41,
        calibration_end: int = 100,
        n_rounds: int = 200,
        rds_window: int = 5,
        rds_alpha: float = None,
        confirm_consecutive: int = 3,
        min_instances: int = 5,
        cusum_k_ref: float = 0.5,
        cusum_h: float = 7.0,
        fwer_p: float = 0.05,
    ):
        self.warmup_rounds = warmup_rounds
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        
        if rds_alpha is None:
            rds_alpha = bonferroni_k(n_rounds, calibration_end, fwer_p)
        
        self.rds = RDSDetector(
            warmup_rounds=warmup_rounds,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            window_size=rds_window,
            alpha=rds_alpha,
            min_instances=min_instances,
            confirm_consecutive=confirm_consecutive,
            use_fixed_threshold=True,
        )
        
        self.cusum = CUSUMDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            confirm_consecutive=confirm_consecutive,
            mode="up",
            cusum_k_ref=cusum_k_ref,
            cusum_h=cusum_h,
        )
        
        self.ph = PageHinkleyDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            confirm_consecutive=confirm_consecutive,
            mode="both",
            cusum_k_ref=cusum_k_ref,
            cusum_h=cusum_h,
        )
        
        self.adwin = ADWINDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            alpha=rds_alpha,
            window_size=rds_window * 2,
            confirm_consecutive=confirm_consecutive,
        )
        
        self.kswin = KSWINDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            alpha=rds_alpha,
            window_size=rds_window * 2,
            confirm_consecutive=confirm_consecutive,
        )
    
    def detect(
        self,
        loss_matrix: np.ndarray,
        aggregated_loss: Optional[np.ndarray] = None,
        t0: Optional[int] = None,
    ) -> Dict[str, DriftDetectionResult]:
        """
        Run all detection methods.
        
        Args:
            loss_matrix: Per-client losses, shape (n_rounds, n_clients)
            aggregated_loss: Global loss per round
            t0: Ground truth drift onset
        
        Returns:
            Dict mapping method name to DriftDetectionResult
        """
        if aggregated_loss is None:
            aggregated_loss = np.mean(loss_matrix, axis=1)
        
        results = {}
        
        results['rds'] = self.rds.detect(loss_matrix, aggregated_loss, t0)
        
        results['cusum'] = self.cusum.detect(aggregated_loss, self.warmup_rounds, t0)
        
        results['page_hinkley'] = self.ph.detect(aggregated_loss, self.warmup_rounds, t0)
        
        results['adwin'] = self.adwin.detect(aggregated_loss, self.warmup_rounds, t0)
        
        results['kswin'] = self.kswin.detect(aggregated_loss, self.warmup_rounds, t0)
        
        return results
    
    def get_first_trigger(
        self,
        results: Dict[str, DriftDetectionResult],
    ) -> Optional[DriftDetectionResult]:
        """
        Get the first triggered result (earliest trigger round).
        
        Returns None if no method triggered.
        """
        triggered = [r for r in results.values() if r.triggered]
        if not triggered:
            return None
        return min(triggered, key=lambda r: r.trigger_round or float('inf'))
