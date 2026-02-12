"""
Configuration module for drift detection experiments.

Contains dataclasses for all experiment parameters including dataset config,
FL training config, drift config, and diagnosis config.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from pathlib import Path
import json


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    name: str = "hyperplane"
    n_features: int = 5
    n_samples_per_client: int = 500
    test_size: float = 0.2
    use_all_data_per_client: bool = False
    noise: float = 0.05
    
    n_drift_features: int = 1
    mag_change: float = 0.5
    sigma: float = 0.1
    
    classification_function_pre: int = 0
    classification_function_post: int = 1
    
    drift_condition_feature: str = "age"
    drift_condition_threshold: float = 50.0
    drift_flip_prob: float = 0.3
    
    @property
    def feature_names(self) -> List[str]:
        """Return feature names based on dataset type."""
        if self.name == 'hyperplane':
            return [f'x{i}' for i in range(self.n_features)]
        elif self.name == 'agrawal':
            return ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']
        elif self.name == 'adult':
            return ['age', 'workclass', 'fnlwgt', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        elif self.name == 'wine':
            return ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar',
                    'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        elif self.name == 'fed_heart':
            return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                    'oldpeak', 'slope', 'ca', 'thal']
        elif self.name == 'elec2':
            return ['nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer', 'period_sin', 'period_cos']
        elif self.name == 'diabetes':
            return ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        elif self.name == 'credit':
            return ['checking_status', 'duration', 'credit_history', 'purpose',
                    'credit_amount', 'savings_status', 'employment',
                    'installment_commitment', 'personal_status', 'other_parties',
                    'residence_since', 'property_magnitude', 'age',
                    'other_payment_plans', 'housing', 'existing_credits', 'job',
                    'num_dependents', 'own_telephone', 'foreign_worker']
        elif self.name == 'bank_marketing':
            return ['age', 'job', 'marital', 'education', 'default', 'balance',
                    'housing', 'loan', 'contact', 'day', 'month',
                    'campaign', 'pdays', 'previous', 'poutcome']
        else:
            raise ValueError(f"Unknown dataset: {self.name}")


@dataclass
class FLConfig:
    """Configuration for federated learning training."""
    
    n_clients: int = 10
    n_rounds: int = 40
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.05
    momentum: float = 0.5
    participation_fraction: float = 1.0
    
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    n_classes: int = 2


@dataclass
class DriftConfig:
    """Configuration for drift injection."""
    
    t0: int = 20
    drifted_client_proportion: float = 0.4
    drift_magnitude: float = 0.5
    drifted_features: Set[int] = field(default_factory=set)
    
    def get_drifted_clients(self, n_clients: int) -> Set[int]:
        """Return set of drifted client indices."""
        n_drifted = int(n_clients * self.drifted_client_proportion)
        return set(range(n_clients - n_drifted, n_clients))


@dataclass
class TriggerConfig:
    """
    Configuration for drift trigger detection (RDS + CUSUM + PH + ADWIN + KSWIN).
    
    Unified Phase-I / Phase-II design:
    - RDS / ADWIN / KSWIN: Bonferroni threshold θ = μ̂ + k·σ̂,
      k = z_{1 − p/M}, p = fwer_p, M = n_rounds − calibration_end.
    - CUSUM / PH: ARL-based, k_ref·σ₀ reference, h·σ₀ decision interval.
    """
    
    warmup_rounds: int = 40
    calibration_start_round: int = 41
    calibration_end_round: int = 80
    
    rds_window: int = 5
    rds_alpha: float = None
    confirm_consecutive: int = 3
    fwer_p: float = 0.05
    
    cusum_k_ref: float = 0.5
    cusum_h: float = 7.0
    
    min_instances: int = 5
    
    ph_delta: float = 0.005
    
    skip_diagnosis_if_no_trigger: bool = True


@dataclass
class DiagnosisConfig:
    """
    Configuration for diagnosis computation.
    
    Diagnosis window: window_size rounds before trigger + trigger round.
    With window_size=5: FI computed for 6 rounds (e.g., rounds 245-250
    if trigger at 250).  The last round is the trigger; the preceding 5
    rounds form the "past" distribution for Dist(FI) RDS ranking.
    
    Background set: 32 IID samples from each client's training data.
    Estimation set: full client validation set or 1000 IID samples,
    whichever is smaller.
    """
    
    window_size: int = 5
    
    background_size: int = 32
    estimation_max_samples: int = 1000
    n_perm: int = 5
    
    sage_n_samples_max: int = 2048
    sage_thresh: float = 0.025
    
    use_kmeans_background: bool = False
    
    dist_fi_rds_window: int = 5
    dist_fi_n_calibration: int = 3
    dist_fi_alpha: float = 2.33
    
    compute_sage: bool = True
    compute_pfi: bool = True
    compute_shap: bool = True


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    
    k: int = 3
    use_ground_truth_k: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    seed: int = 42
    experiment_name: str = ""
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    base_output_dir: Path = field(default_factory=lambda: Path("results"))
    
    def __post_init__(self):
        """Generate experiment name if not provided."""
        if not self.experiment_name:
            self.experiment_name = (
                f"{self.dataset.name}_seed{self.seed}_"
                f"drift{self.drift.drifted_client_proportion}_"
                f"mag{self.drift.drift_magnitude}"
            )
    
    @property
    def output_dir(self) -> Path:
        """Return the output directory for this experiment."""
        return self.base_output_dir / self.experiment_name
    
    @property
    def models_dir(self) -> Path:
        """Return directory for saving model checkpoints."""
        return self.output_dir / "models"
    
    @property
    def checkpoints_dir(self) -> Path:
        """Return directory for saving training checkpoints."""
        return self.output_dir / "checkpoints"
    
    @property
    def fi_scores_dir(self) -> Path:
        """Return directory for feature importance scores."""
        return self.output_dir / "fi_scores"
    
    @property
    def diagnosis_dir(self) -> Path:
        """Return directory for diagnosis results."""
        return self.output_dir / "diagnosis"
    
    @property
    def metrics_dir(self) -> Path:
        """Return directory for evaluation metrics."""
        return self.output_dir / "metrics"
    
    @property
    def logs_dir(self) -> Path:
        """Return directory for training logs."""
        return self.output_dir / "logs"
    
    @property
    def plots_dir(self) -> Path:
        """Return directory for plots."""
        return self.output_dir / "plots"
    
    def create_directories(self):
        """Create all output directories."""
        for dir_path in [
            self.output_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.fi_scores_dir,
            self.diagnosis_dir,
            self.metrics_dir,
            self.logs_dir,
            self.plots_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = self.output_dir / "config.json"
        
        config_dict = {
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'dataset': {
                'name': self.dataset.name,
                'n_features': self.dataset.n_features,
                'n_samples_per_client': self.dataset.n_samples_per_client,
                'test_size': self.dataset.test_size,
                'use_all_data_per_client': self.dataset.use_all_data_per_client,
                'noise': self.dataset.noise,
                'n_drift_features': self.dataset.n_drift_features,
                'mag_change': self.dataset.mag_change,
                'sigma': self.dataset.sigma,
                'classification_function_pre': self.dataset.classification_function_pre,
                'classification_function_post': self.dataset.classification_function_post,
            },
            'fl': {
                'n_clients': self.fl.n_clients,
                'n_rounds': self.fl.n_rounds,
                'local_epochs': self.fl.local_epochs,
                'batch_size': self.fl.batch_size,
                'learning_rate': self.fl.learning_rate,
                'momentum': self.fl.momentum,
                'participation_fraction': self.fl.participation_fraction,
                'hidden_sizes': self.fl.hidden_sizes,
                'n_classes': self.fl.n_classes,
            },
            'drift': {
                't0': self.drift.t0,
                'drifted_client_proportion': self.drift.drifted_client_proportion,
                'drift_magnitude': self.drift.drift_magnitude,
                'drifted_features': list(self.drift.drifted_features),
            },
            'trigger': {
                'warmup_rounds': self.trigger.warmup_rounds,
                'calibration_start_round': self.trigger.calibration_start_round,
                'calibration_end_round': self.trigger.calibration_end_round,
                'rds_window': self.trigger.rds_window,
                'rds_alpha': self.trigger.rds_alpha,
                'fwer_p': self.trigger.fwer_p,
                'confirm_consecutive': self.trigger.confirm_consecutive,
                'min_instances': self.trigger.min_instances,
                'cusum_k_ref': self.trigger.cusum_k_ref,
                'cusum_h': self.trigger.cusum_h,
                'skip_diagnosis_if_no_trigger': self.trigger.skip_diagnosis_if_no_trigger,
            },
            'diagnosis': {
                'window_size': self.diagnosis.window_size,
                'background_size': self.diagnosis.background_size,
                'estimation_max_samples': self.diagnosis.estimation_max_samples,
                'n_perm': self.diagnosis.n_perm,
                'sage_n_samples_max': self.diagnosis.sage_n_samples_max,
                'sage_thresh': self.diagnosis.sage_thresh,
                'use_kmeans_background': self.diagnosis.use_kmeans_background,
                'dist_fi_rds_window': self.diagnosis.dist_fi_rds_window,
                'dist_fi_n_calibration': self.diagnosis.dist_fi_n_calibration,
                'dist_fi_alpha': self.diagnosis.dist_fi_alpha,
                'compute_sage': self.diagnosis.compute_sage,
                'compute_pfi': self.diagnosis.compute_pfi,
                'compute_shap': self.diagnosis.compute_shap,
            },
            'metrics': {
                'k': self.metrics.k,
                'use_ground_truth_k': self.metrics.use_ground_truth_k,
            },
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            seed=config_dict['seed'],
            experiment_name=config_dict['experiment_name'],
            dataset=DatasetConfig(**config_dict['dataset']),
            fl=FLConfig(**config_dict['fl']),
            drift=DriftConfig(
                t0=config_dict['drift']['t0'],
                drifted_client_proportion=config_dict['drift']['drifted_client_proportion'],
                drift_magnitude=config_dict['drift']['drift_magnitude'],
                drifted_features=set(config_dict['drift']['drifted_features']),
            ),
            trigger=TriggerConfig(**config_dict['trigger']),
            diagnosis=DiagnosisConfig(**config_dict['diagnosis']),
            metrics=MetricsConfig(**config_dict['metrics']),
        )


def create_experiment_configs(
    dataset_name: str,
    seeds: List[int],
    drift_proportions: List[float],
    drift_magnitudes: List[float],
    drifted_features: Set[int],
    base_output_dir: Path = Path("results"),
) -> List[ExperimentConfig]:
    """
    Create a list of experiment configurations for grid search.
    
    Args:
        dataset_name: 'hyperplane' or 'agrawal'
        seeds: List of random seeds
        drift_proportions: List of drifted client proportions
        drift_magnitudes: List of drift magnitudes
        drifted_features: Ground truth drifted feature indices
        base_output_dir: Base output directory
    
    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    
    for seed in seeds:
        for prop in drift_proportions:
            for mag in drift_magnitudes:
                dataset_config = DatasetConfig(
                    name=dataset_name,
                    mag_change=mag if dataset_name == 'hyperplane' else 0.0,
                )
                
                drift_config = DriftConfig(
                    drifted_client_proportion=prop,
                    drift_magnitude=mag,
                    drifted_features=drifted_features,
                )
                
                config = ExperimentConfig(
                    seed=seed,
                    dataset=dataset_config,
                    drift=drift_config,
                    base_output_dir=base_output_dir,
                )
                configs.append(config)
    
    return configs
