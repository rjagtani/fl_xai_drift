"""
Wine Quality (UCI/OpenML) dataset for federated learning with feature-conditioned concept drift.
Clients partitioned by alcohol (binned). Binary target: quality >= 6 (good) vs < 6 (bad).
Drift: label flip with prob when condition (e.g. alcohol >= threshold) holds; all clients affected post-t0.
Same mechanism as Adult (feature-conditioned flip, drift_flip_prob).
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

from .base import BaseDataGenerator, ClientDataset

# Red wine quality (OpenML): 11 features + quality. Quality binarized to good (>=6) / bad (<6).
WINE_FEATURE_NAMES = [
    'fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar',
    'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol',
]
ALCOHOL_COL = 10  # Index of alcohol for drift condition (raw value before scaling)


class WineQualityDataGenerator(BaseDataGenerator):
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        test_size: float = 0.2,
        seed: int = 42,
        drift_condition_feature: str = "alcohol",
        drift_condition_threshold: float = 10.5,
        drift_flip_prob: float = 1.0,
        use_all_data_per_client: bool = False,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=len(WINE_FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        self.use_all_data_per_client = use_all_data_per_client
        self.drift_condition_feature = drift_condition_feature
        self.drift_condition_threshold = drift_condition_threshold
        self.drift_flip_prob = drift_flip_prob
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None
        self._client_to_indices: Optional[Dict[int, np.ndarray]] = None
        self._alcohol_col: int = ALCOHOL_COL
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = WINE_FEATURE_NAMES.copy()

    def _load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Red wine quality from OpenML
        data = fetch_openml("wine_quality", as_frame=True, parser="auto")
        df = data.frame
        quality_col = "quality"
        if quality_col not in df.columns:
            quality_col = [c for c in df.columns if "quality" in c.lower()][0]
        y = (df[quality_col].astype(float) >= 6).astype(np.int64).values
        feature_cols = [c for c in WINE_FEATURE_NAMES if c in df.columns]
        if len(feature_cols) != len(WINE_FEATURE_NAMES):
            feature_cols = [c for c in df.columns if c != quality_col]
        self._feature_names = feature_cols
        self.n_features = len(feature_cols)
        self._alcohol_col = feature_cols.index("alcohol") if "alcohol" in feature_cols else len(feature_cols) - 1
        X = df[feature_cols].values.astype(np.float64)
        quality_raw = df[quality_col].values.astype(float)
        partition_raw = df["alcohol"].values.astype(float)
        return X, y, quality_raw, partition_raw

    def _build_client_indices(self, X: np.ndarray, partition_raw: np.ndarray) -> Dict[int, np.ndarray]:
        # Bin partition column (alcohol) into n_clients bins for non-IID partition
        bins = np.percentile(partition_raw, np.linspace(0, 100, self.n_clients + 1))
        bins[-1] += 1e-6
        bin_idx = np.digitize(partition_raw, bins) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_clients - 1)
        client_indices: Dict[int, List[int]] = {c: [] for c in range(self.n_clients)}
        for idx, c in enumerate(bin_idx):
            client_indices[c].append(idx)
        out: Dict[int, np.ndarray] = {}
        for c in range(self.n_clients):
            ind = np.array(client_indices[c], dtype=np.int64)
            if len(ind) == 0:
                ind = self._rng.choice(len(partition_raw), size=min(self.n_samples_per_client, len(partition_raw)), replace=False)
            elif not self.use_all_data_per_client:
                if len(ind) >= self.n_samples_per_client:
                    ind = self._rng.choice(ind, size=self.n_samples_per_client, replace=False)
                else:
                    ind = self._rng.choice(ind, size=self.n_samples_per_client, replace=True)
            out[c] = ind
        return out

    def _ensure_data(self):
        if self._X_full is not None:
            return
        X, y, quality_raw, partition_raw = self._load_and_preprocess()
        self._client_to_indices = self._build_client_indices(X, partition_raw)
        # Store raw alcohol BEFORE scaling (for drift condition)
        self._raw_alcohol = X[:, self._alcohol_col].copy()
        self._scaler = StandardScaler()
        self._X_full = self._scaler.fit_transform(X.astype(np.float64))
        self._y_full = y

    def _apply_drift(self, raw_alcohol: np.ndarray, y: np.ndarray, client_id: int) -> np.ndarray:
        """Apply label flip drift for samples where raw alcohol >= threshold (same idea as Adult age)."""
        y_out = y.copy()
        cond = raw_alcohol >= self.drift_condition_threshold
        rng = np.random.default_rng(self.seed + 10000 * client_id)
        flip = rng.random(size=cond.sum()) < self.drift_flip_prob
        idx = np.where(cond)[0]
        for i in idx[flip]:
            y_out[i] = 1 - y_out[i]
        return y_out

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def get_drifted_feature_indices(self) -> Set[int]:
        return {self._alcohol_col}

    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
        drift_magnitude: float = 0.0,
    ) -> Dict[int, ClientDataset]:
        self._ensure_data()
        client_datasets = {}
        for client_id in range(self.n_clients):
            ind = self._client_to_indices[client_id]
            X = self._X_full[ind]
            y = self._y_full[ind]
            raw_alcohol = self._raw_alcohol[ind]
            if generate_drifted:
                y = self._apply_drift(raw_alcohol, y, client_id)
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
                is_drifted=generate_drifted,
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
