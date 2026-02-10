"""
Page-Hinkley drift detection for time series.

The Page-Hinkley test is a sequential analysis technique for detecting 
changes in the mean of a sequence of observations.
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PHDetectionResult:
    """Result of Page-Hinkley detection."""
    triggered: bool
    trigger_round: Optional[int]
    ph_statistic: float
    cumulative_sum: float
    running_mean: float
    detection_delay: Optional[int] = None  # If t0 is known


class PageHinkleyDetector:
    """
    Page-Hinkley Test for detecting drift in a time series.
    
    The test monitors the cumulative sum of differences between observed
    values and a running mean, triggering when the cumulative sum exceeds
    a threshold (lambda).
    
    Parameters:
    - delta: Minimum magnitude of change to detect (threshold for update)
    - lambda_: Detection threshold (sensitivity parameter)
    - warmup: Minimum number of observations before detection can occur
    - direction: 'increase', 'decrease', or 'both' for type of change to detect
    """
    
    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        warmup: int = 5,
        direction: str = 'increase',  # 'increase', 'decrease', or 'both'
    ):
        self.delta = delta
        self.lambda_ = lambda_
        self.warmup = warmup
        self.direction = direction
        
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.n_samples = 0
        self.running_mean = 0.0
        self.sum_increase = 0.0  # PH+ for detecting increase
        self.sum_decrease = 0.0  # PH- for detecting decrease
        self.min_ph_plus = float('inf')
        self.max_ph_minus = float('-inf')
        self.triggered = False
        self.trigger_round = None
        self.observations: List[float] = []
    
    def update(self, value: float) -> bool:
        """
        Update detector with new observation.
        
        Args:
            value: New observation value
        
        Returns:
            True if drift is detected, False otherwise
        """
        self.observations.append(value)
        self.n_samples += 1
        
        # Update running mean
        self.running_mean += (value - self.running_mean) / self.n_samples
        
        # Update cumulative sums
        # For detecting increase in mean
        self.sum_increase += value - self.running_mean - self.delta
        self.min_ph_plus = min(self.min_ph_plus, self.sum_increase)
        ph_plus = self.sum_increase - self.min_ph_plus
        
        # For detecting decrease in mean
        self.sum_decrease += self.running_mean - value - self.delta
        self.max_ph_minus = max(self.max_ph_minus, self.sum_decrease)
        ph_minus = self.max_ph_minus - self.sum_decrease
        
        # Check for drift after warmup period
        if self.n_samples >= self.warmup and not self.triggered:
            if self.direction == 'increase' and ph_plus > self.lambda_:
                self.triggered = True
                self.trigger_round = self.n_samples
            elif self.direction == 'decrease' and ph_minus > self.lambda_:
                self.triggered = True
                self.trigger_round = self.n_samples
            elif self.direction == 'both' and (ph_plus > self.lambda_ or ph_minus > self.lambda_):
                self.triggered = True
                self.trigger_round = self.n_samples
        
        return self.triggered
    
    def get_statistic(self) -> float:
        """Get current PH statistic based on detection direction."""
        ph_plus = self.sum_increase - self.min_ph_plus
        ph_minus = self.max_ph_minus - self.sum_decrease
        
        if self.direction == 'increase':
            return ph_plus
        elif self.direction == 'decrease':
            return ph_minus
        else:
            return max(ph_plus, ph_minus)
    
    def detect(
        self,
        series: List[float],
        t0: Optional[int] = None,
    ) -> PHDetectionResult:
        """
        Run detection on a complete time series.
        
        Args:
            series: List of observations
            t0: Known drift onset round (for computing detection delay)
        
        Returns:
            PHDetectionResult with detection information
        """
        self.reset()
        
        for value in series:
            self.update(value)
            if self.triggered:
                break
        
        delay = None
        if self.triggered and t0 is not None:
            delay = self.trigger_round - t0
        
        return PHDetectionResult(
            triggered=self.triggered,
            trigger_round=self.trigger_round,
            ph_statistic=self.get_statistic(),
            cumulative_sum=self.sum_increase if self.direction != 'decrease' else self.sum_decrease,
            running_mean=self.running_mean,
            detection_delay=delay,
        )


class FeaturePageHinkley:
    """
    Page-Hinkley detector for per-feature time series.
    
    Used for Mean(FI)+PH diagnosis method to detect changes in individual
    feature importance over time.
    """
    
    def __init__(
        self,
        n_features: int,
        delta: float = 0.005,
        lambda_: float = 50.0,
        warmup: int = 3,
    ):
        self.n_features = n_features
        self.detectors = [
            PageHinkleyDetector(delta=delta, lambda_=lambda_, warmup=warmup, direction='both')
            for _ in range(n_features)
        ]
    
    def update(self, feature_values: np.ndarray) -> np.ndarray:
        """
        Update detectors with new feature values.
        
        Args:
            feature_values: Array of shape (n_features,) with FI values
        
        Returns:
            Boolean array indicating which features triggered
        """
        triggered = np.zeros(self.n_features, dtype=bool)
        for i, (detector, value) in enumerate(zip(self.detectors, feature_values)):
            triggered[i] = detector.update(value)
        return triggered
    
    def get_statistics(self) -> np.ndarray:
        """Get PH statistics for all features."""
        return np.array([d.get_statistic() for d in self.detectors])
    
    def detect(
        self,
        series: np.ndarray,  # Shape: (n_rounds, n_features)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run detection on complete feature series.
        
        Args:
            series: Array of shape (n_rounds, n_features)
        
        Returns:
            Tuple of (triggered array, final statistics array)
        """
        for t in range(series.shape[0]):
            self.update(series[t])
        
        triggered = np.array([d.triggered for d in self.detectors])
        statistics = self.get_statistics()
        
        return triggered, statistics
    
    def get_feature_scores(self) -> np.ndarray:
        """
        Get PH-based scores for feature ranking.
        
        Higher scores indicate more change detected.
        """
        return self.get_statistics()
