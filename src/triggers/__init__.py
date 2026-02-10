"""
Drift trigger detection modules.

Includes:
- RDS, CUSUM, Page-Hinkley, ADWIN, KSWIN (system-level loss-based detectors)
- FL Baselines: FLASH, CDA-FedAvg, Manias (client-level → system-level)
"""

from .page_hinkley import PageHinkleyDetector as FeaturePHDetector, FeaturePageHinkley
from .drift_detectors import (
    RDSDetector,
    CUSUMDetector,
    PageHinkleyDetector,
    ADWINDetector,
    KSWINDetector,
    MultiMethodDetector,
    DriftDetectionResult,
    compute_rds,
)
from .fl_baselines import (
    FLASHDetector,
    CDAFedAvgDetector,
    ManiasPCAKMeansDetector,
    FLBaselineMultiDetector,
    ClientDriftResult,
    SystemDriftResult,
)

__all__ = [
    # Loss-based detectors (σ-normalized)
    'RDSDetector',
    'CUSUMDetector',
    'PageHinkleyDetector',
    'ADWINDetector',
    'KSWINDetector',
    'MultiMethodDetector',
    'DriftDetectionResult',
    'compute_rds',
    # Feature-level PH (for diagnosis)
    'FeaturePHDetector',
    'FeaturePageHinkley',
    # FL Baselines
    'FLASHDetector',
    'CDAFedAvgDetector',
    'ManiasPCAKMeansDetector',
    'FLBaselineMultiDetector',
    'ClientDriftResult',
    'SystemDriftResult',
]
