"""
Fed-Heart Disease (UCI) dataset for federated learning.
Natural cross-silo partitioning: 4 hospitals (Cleveland, Hungary, Switzerland, VA Long Beach).
Binary target: heart disease present (1) vs absent (0).
Drift: label flip with prob when condition (e.g. cholesterol > median) holds.
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import BaseDataGenerator, ClientDataset

# UCI Heart Disease: 13 clinical features (after preprocessing)
HEART_FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal',
]

# Hospital names for reference
HOSPITAL_NAMES = ['cleveland', 'hungarian', 'switzerland', 'va_long_beach']

# URLs for the 4 hospital files from UCI
UCI_URLS = {
    'cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
    'hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
    'switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
    'va_long_beach': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data',
}

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target',
]


class FedHeartDataGenerator(BaseDataGenerator):
    """
    Heart Disease dataset with natural cross-silo FL partitioning by hospital.

    4 clients (hospitals):
      0: Cleveland  (~303 patients)
      1: Hungarian   (~294 patients)
      2: Switzerland  (~123 patients)
      3: VA Long Beach (~200 patients)

    Drift: conditional label flip on cholesterol (chol > median).
    """

    def __init__(
        self,
        n_clients: int = 4,
        n_samples_per_client: int = 500,  # Not used; natural sizes kept
        test_size: float = 0.2,
        seed: int = 42,
        drift_condition_feature: str = "chol",
        drift_condition_threshold: Optional[float] = None,  # Auto: median
        drift_flip_prob: float = 0.3,
    ):
        # Fixed at 4 clients (hospitals)
        super().__init__(
            n_clients=4,
            n_samples_per_client=n_samples_per_client,
            n_features=len(HEART_FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        self.drift_condition_feature = drift_condition_feature
        self.drift_condition_threshold = drift_condition_threshold
        self.drift_flip_prob = drift_flip_prob
        self._X_per_hospital: Optional[Dict[int, np.ndarray]] = None
        self._y_per_hospital: Optional[Dict[int, np.ndarray]] = None
        self._raw_condition_per_hospital: Optional[Dict[int, np.ndarray]] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = HEART_FEATURE_NAMES.copy()
        self._condition_col: int = HEART_FEATURE_NAMES.index(drift_condition_feature)

    def _load_single_hospital(self, url: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a single hospital's data from UCI."""
        df = pd.read_csv(url, header=None, names=COLUMN_NAMES, na_values='?')
        # Binary target: 0 = no disease, 1-4 = disease present → binary
        y = (df['target'].values > 0).astype(np.int64)
        # Impute missing numeric values with column median (Switzerland has many NaNs)
        for col in HEART_FEATURE_NAMES:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
        X = df[HEART_FEATURE_NAMES].values.astype(np.float64)
        return X, y

    def _ensure_data(self):
        if self._X_per_hospital is not None:
            return

        # Load all 4 hospitals
        all_X = []
        all_y = []
        hospital_ids = []
        for h_idx, (name, url) in enumerate(UCI_URLS.items()):
            X_h, y_h = self._load_single_hospital(url)
            all_X.append(X_h)
            all_y.append(y_h)
            hospital_ids.extend([h_idx] * len(y_h))

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        hospital_ids = np.array(hospital_ids)

        # Fit scaler on all data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_all)

        # Store raw condition column (before scaling) for drift
        raw_condition = X_all[:, self._condition_col].copy()

        # Auto-set threshold as median if not provided
        if self.drift_condition_threshold is None:
            self.drift_condition_threshold = float(np.median(raw_condition))

        # Split back into per-hospital
        self._X_per_hospital = {}
        self._y_per_hospital = {}
        self._raw_condition_per_hospital = {}

        for h_idx in range(4):
            mask = hospital_ids == h_idx
            self._X_per_hospital[h_idx] = X_scaled[mask].astype(np.float32)
            self._y_per_hospital[h_idx] = y_all[mask]
            self._raw_condition_per_hospital[h_idx] = raw_condition[mask]

        # Update feature count
        self.n_features = X_all.shape[1]
        self._feature_names = HEART_FEATURE_NAMES[:self.n_features]

    def _apply_drift(self, raw_cond: np.ndarray, y: np.ndarray, client_id: int,
                     flip_prob_override: float = None) -> np.ndarray:
        """Apply label flip for samples where condition feature > threshold."""
        y_out = y.copy()
        flip_prob = flip_prob_override if flip_prob_override is not None else self.drift_flip_prob
        cond = raw_cond >= self.drift_condition_threshold
        rng = np.random.default_rng(self.seed + 10000 * client_id)
        n_cond = cond.sum()
        if n_cond == 0:
            return y_out
        flip = rng.random(size=n_cond) < flip_prob
        idx = np.where(cond)[0]
        for i in idx[flip]:
            y_out[i] = 1 - y_out[i]
        return y_out

    def get_feature_names(self) -> List[str]:
        self._ensure_data()
        return self._feature_names.copy()

    def get_drifted_feature_indices(self) -> Set[int]:
        return {self._condition_col}

    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
        drift_magnitude: float = 0.0,
        flip_prob_override: float = None,
    ) -> Dict[int, ClientDataset]:
        """flip_prob_override allows gradual drift (variable flip prob)."""
        self._ensure_data()
        client_datasets = {}
        for client_id in range(self.n_clients):
            X = self._X_per_hospital[client_id]
            y = self._y_per_hospital[client_id]
            raw_cond = self._raw_condition_per_hospital[client_id]
            if generate_drifted and client_id in drifted_clients:
                y = self._apply_drift(raw_cond, y, client_id,
                                      flip_prob_override=flip_prob_override)
            # Split
            if len(X) < 5:
                # Too few samples, skip split
                X_train, X_val = X, X
                y_train, y_val = y, y
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.seed + client_id,
                    stratify=y if len(np.unique(y)) > 1 else None,
                )
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train.astype(np.float32),
                y_train=y_train,
                X_val=X_val.astype(np.float32),
                y_val=y_val,
                is_drifted=generate_drifted and client_id in drifted_clients,
            )
        return client_datasets

    def generate_round_data(
        self,
        round_num: int,
        t0: int,
        drifted_clients: Set[int],
        drift_magnitude: float,
    ) -> Dict[int, ClientDataset]:
        is_drift_phase = round_num >= t0
        return self.generate_static_client_data(
            drifted_clients=drifted_clients,
            generate_drifted=is_drift_phase,
            drift_magnitude=drift_magnitude,
        )
