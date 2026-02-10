"""
Diabetes (Pima Indians) dataset for federated learning.
768 samples, 8 numeric features. Binary target: diabetes onset (1) vs healthy (0).
Clients partitioned by Age bins (non-IID).
Drift: feature noise on Glucose (index 1) -- strongest predictor.
"""

from typing import Dict, List, Set, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import BaseDataGenerator, ClientDataset

DIABETES_FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
]


class DiabetesDataGenerator(BaseDataGenerator):
    """
    Pima Indians Diabetes dataset with FL partitioning by age bins.

    768 samples, 8 numeric features.  Clients are defined by equal-frequency
    binning of the Age column, giving non-IID partitions (younger patients
    have very different diabetes profiles from older ones).
    """

    def __init__(
        self,
        n_clients: int = 6,
        n_samples_per_client: int = 500,  # not used when use_all_data_per_client=True
        test_size: float = 0.2,
        seed: int = 42,
        use_all_data_per_client: bool = True,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=len(DIABETES_FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        self.use_all_data_per_client = use_all_data_per_client
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None
        self._client_to_indices: Optional[Dict[int, np.ndarray]] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = DIABETES_FEATURE_NAMES.copy()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _ensure_data(self):
        if self._X_full is not None:
            return

        # CSV ships with the repo at  <repo_root>/datasets/diabetes.csv
        csv_path = Path(__file__).resolve().parents[3] / 'datasets' / 'diabetes.csv'
        df = pd.read_csv(csv_path)

        y = df['Outcome'].values.astype(np.int64)
        X = df[DIABETES_FEATURE_NAMES].values.astype(np.float64)

        # Partition by Age bins
        age_col = df['Age'].values.astype(float)
        self._client_to_indices = self._build_client_indices(age_col)

        # Scale features
        self._scaler = StandardScaler()
        self._X_full = self._scaler.fit_transform(X).astype(np.float32)
        self._y_full = y

    def _build_client_indices(
        self, partition_col: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Bin *partition_col* into *n_clients* equal-frequency bins."""
        bins = np.percentile(partition_col, np.linspace(0, 100, self.n_clients + 1))
        bins[-1] += 1e-6
        bin_idx = np.digitize(partition_col, bins) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_clients - 1)

        client_indices: Dict[int, List[int]] = {c: [] for c in range(self.n_clients)}
        for idx, c in enumerate(bin_idx):
            client_indices[c].append(idx)

        out: Dict[int, np.ndarray] = {}
        for c in range(self.n_clients):
            ind = np.array(client_indices[c], dtype=np.int64)
            if len(ind) == 0:
                ind = self._rng.choice(
                    len(partition_col),
                    size=min(self.n_samples_per_client, len(partition_col)),
                    replace=False,
                )
            elif not self.use_all_data_per_client:
                if len(ind) >= self.n_samples_per_client:
                    ind = self._rng.choice(ind, size=self.n_samples_per_client, replace=False)
                else:
                    ind = self._rng.choice(ind, size=self.n_samples_per_client, replace=True)
            out[c] = ind
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def get_drifted_feature_indices(self) -> Set[int]:
        return {1}  # Glucose

    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
        drift_magnitude: float = 0.0,
        flip_prob_override: float = None,
    ) -> Dict[int, ClientDataset]:
        """Return clean per-client train/val splits.

        Drift (feature noise) is injected externally by ``run_drift_types.py``
        via the generic ``feature_noise`` branch in ``get_round_data_custom``.
        """
        self._ensure_data()
        client_datasets: Dict[int, ClientDataset] = {}

        for client_id in range(self.n_clients):
            ind = self._client_to_indices[client_id]
            X = self._X_full[ind]
            y = self._y_full[ind]

            if len(X) < 5:
                X_train, X_val = X, X
                y_train, y_val = y, y
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.test_size,
                    random_state=self.seed + client_id,
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
