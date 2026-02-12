"""
ELEC2 (Electricity) dataset for federated learning with temporal streaming.

Source: Australian NSW Electricity Market, May 1996 – Dec 1998.
~45,312 records at 30-min intervals (48 per day).

FL partitioning: 7 clients, one per day-of-week (Mon=0 … Sun=6).
Each round, every client advances one week (gets the next day's 48 records).
~135 usable rounds (weeks).

Binary target: electricity price UP (1) vs DOWN (0).

Features (6 numeric): nswprice, nswdemand, vicprice, vicdemand, transfer, day_of_week_sin, day_of_week_cos
(period is encoded as sin/cos rather than raw integer)

Drift: conditional label flip on nswprice > median.
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .base import BaseDataGenerator, ClientDataset

ELEC2_FEATURE_NAMES = [
    'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer',
    'period_sin', 'period_cos',
]


class Elec2DataGenerator(BaseDataGenerator):
    """
    ELEC2 dataset with temporal day-of-week FL partitioning.

    7 clients (Mon–Sun). Each round = one week:
      client 0 (Mon) gets the next Monday's 48 half-hour records,
      client 1 (Tue) gets the next Tuesday's, etc.

    For static mode (generate_static_client_data): pool all data for each
    day-of-week and return train/val splits (ignoring temporal ordering).

    For temporal mode: use generate_temporal_round_data(round_num) which
    returns the specific week's data.
    """

    def __init__(
        self,
        n_clients: int = 7,
        n_samples_per_client: int = 500,
        test_size: float = 0.2,
        seed: int = 42,
        drift_condition_feature: str = "nswprice",
        drift_condition_threshold: Optional[float] = None,
        drift_flip_prob: float = 0.3,
    ):
        super().__init__(
            n_clients=7,
            n_samples_per_client=n_samples_per_client,
            n_features=len(ELEC2_FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        self.drift_condition_feature = drift_condition_feature
        self.drift_condition_threshold = drift_condition_threshold
        self.drift_flip_prob = drift_flip_prob

        self._loaded = False
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None
        self._day_of_week: Optional[np.ndarray] = None
        self._week_index: Optional[np.ndarray] = None
        self._raw_nswprice: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = ELEC2_FEATURE_NAMES.copy()
        self._condition_col: int = 0
        self._n_weeks: int = 0

        self._client_to_indices: Optional[Dict[int, np.ndarray]] = None

    def _load_and_preprocess(self):
        """Load ELEC2 from OpenML and preprocess."""
        from sklearn.datasets import fetch_openml

        data = fetch_openml('electricity', version=1, as_frame=True, parser='auto')
        df = data.frame

        target_col = 'class' if 'class' in df.columns else df.columns[-1]
        y = (df[target_col].astype(str).str.upper() == 'UP').astype(np.int64).values

        feature_map = {}
        for col in ['nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']:
            if col in df.columns:
                feature_map[col] = df[col].astype(float).values

        if 'period' in df.columns:
            period_raw = df['period'].astype(float).values
        else:
            period_raw = np.zeros(len(df))

        period_sin = np.sin(2 * np.pi * period_raw / 48.0)
        period_cos = np.cos(2 * np.pi * period_raw / 48.0)

        if 'day' in df.columns:
            day_raw = df['day'].astype(int).values
            day_of_week = (day_raw - 1) % 7
        else:
            day_of_week = np.repeat(np.arange(len(df) // 48 + 1), 48)[:len(df)] % 7

        X_cols = []
        for col in ['nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']:
            if col in feature_map:
                X_cols.append(feature_map[col])
        X_cols.append(period_sin)
        X_cols.append(period_cos)
        X = np.column_stack(X_cols).astype(np.float64)

        raw_nswprice = feature_map.get('nswprice', X[:, 0]).copy()

        n_records = len(y)
        day_index = np.arange(n_records) // 48
        week_index = day_index // 7

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.drift_condition_threshold is None:
            self.drift_condition_threshold = float(np.median(raw_nswprice))

        self._X_full = X_scaled.astype(np.float32)
        self._y_full = y
        self._day_of_week = day_of_week
        self._week_index = week_index
        self._raw_nswprice = raw_nswprice
        self._scaler = scaler
        self._n_weeks = int(week_index.max()) + 1
        self.n_features = X.shape[1]
        self._feature_names = ELEC2_FEATURE_NAMES[:self.n_features]

        self._client_to_indices = {}
        for dow in range(7):
            self._client_to_indices[dow] = np.where(day_of_week == dow)[0]

        self._loaded = True

    def _ensure_data(self):
        if not self._loaded:
            self._load_and_preprocess()

    def _apply_drift(self, raw_nswprice: np.ndarray, y: np.ndarray, client_id: int,
                     flip_prob_override: float = None) -> np.ndarray:
        """Apply label flip for samples where nswprice > threshold."""
        y_out = y.copy()
        flip_prob = flip_prob_override if flip_prob_override is not None else self.drift_flip_prob
        cond = raw_nswprice > self.drift_condition_threshold
        n_cond = cond.sum()
        if n_cond == 0:
            return y_out
        rng = np.random.default_rng(self.seed + 10000 * client_id)
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

    @property
    def n_weeks(self) -> int:
        self._ensure_data()
        return self._n_weeks

    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
        drift_magnitude: float = 0.0,
        flip_prob_override: float = None,
    ) -> Dict[int, ClientDataset]:
        """
        Static mode: pool all data for each day-of-week, split train/val.
        Used for non-temporal (inject-drift) experiments.
        flip_prob_override allows gradual drift (variable flip prob).
        """
        self._ensure_data()
        client_datasets = {}
        for client_id in range(self.n_clients):
            ind = self._client_to_indices[client_id]
            X = self._X_full[ind]
            y = self._y_full[ind]
            raw_nswprice = self._raw_nswprice[ind]
            if generate_drifted and client_id in drifted_clients:
                y = self._apply_drift(raw_nswprice, y, client_id,
                                      flip_prob_override=flip_prob_override)
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
        """Static mode round data (same data every round, drift switch at t0)."""
        is_drift_phase = round_num >= t0
        return self.generate_static_client_data(
            drifted_clients=drifted_clients,
            generate_drifted=is_drift_phase,
            drift_magnitude=drift_magnitude,
        )
