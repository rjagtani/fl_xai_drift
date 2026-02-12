"""
XAI2026: Synthetic Drift Diagnostics for Federated Learning
============================================================

This package implements drift detection and diagnosis using explainable AI methods
in federated learning settings with synthetic datasets (Hyperplane and Agrawal).

Main components:
- config: Experiment configuration dataclasses
- data: Dataset generation (Hyperplane, Agrawal)
- models: Neural network model definitions
- fl_trainer: Federated learning training loop
- triggers: Page-Hinkley drift trigger detection
- feature_importance: SAGE, PFI, and SHAP computation
- diagnosis: Dist(FI) and Mean(FI)+PH diagnosis methods
- metrics: Hits@K, Precision@K, Recall@K, MRR
- runner: Main experiment orchestration
"""

__version__ = "1.0.0"

from .config import (
    ExperimentConfig,
    DatasetConfig,
    FLConfig,
    DriftConfig,
    TriggerConfig,
    DiagnosisConfig,
    MetricsConfig,
    create_experiment_configs,
)

from .data import (
    HyperplaneDataGenerator,
    AgrawalDataGenerator,
    BaseDataGenerator,
    ClientDataset,
)

from .models import MLP, get_weights, set_weights

from .fl_trainer import FLTrainer, FLClient, FLServer

from .triggers import PageHinkleyDetector

from .feature_importance import (
    SAGEComputer,
    PFIComputer,
    SHAPComputer,
    FeatureImportanceResult,
)

from .diagnosis import (
    DistFIDiagnosis,
    MeanFIPHDiagnosis,
    DiagnosisEngine,
)

from .metrics import (
    DiagnosticMetrics,
    compute_metrics,
    aggregate_results,
)

from .runner import ExperimentRunner

__all__ = [
    'ExperimentConfig',
    'DatasetConfig',
    'FLConfig',
    'DriftConfig',
    'TriggerConfig',
    'DiagnosisConfig',
    'MetricsConfig',
    'create_experiment_configs',
    'HyperplaneDataGenerator',
    'AgrawalDataGenerator',
    'BaseDataGenerator',
    'ClientDataset',
    'MLP',
    'get_weights',
    'set_weights',
    'FLTrainer',
    'FLClient',
    'FLServer',
    'PageHinkleyDetector',
    'SAGEComputer',
    'PFIComputer',
    'SHAPComputer',
    'FeatureImportanceResult',
    'DistFIDiagnosis',
    'MeanFIPHDiagnosis',
    'DiagnosisEngine',
    'DiagnosticMetrics',
    'compute_metrics',
    'aggregate_results',
    'ExperimentRunner',
]
