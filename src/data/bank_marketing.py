"""
Bank Marketing (UCI) dataset for federated learning with feature-noise drift.
Clients partitioned by job category. Categoricals are target-encoded.
Duration feature excluded (target leakage).
Drift: Gaussian noise added to specified numeric features (e.g. balance).
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

from .base import BaseDataGenerator, ClientDataset

BANK_FEATURE_NAMES = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'day', 'month',
    'campaign', 'pdays', 'previous', 'poutcome',
]
NUMERIC_COLS = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
TARGET_ENCODE_SMOOTHING = 1


class BankMarketingDataGenerator(BaseDataGenerator):
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        test_size: float = 0.2,
        seed: int = 42,
        use_all_data_per_client: bool = True,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=len(BANK_FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        self.use_all_data_per_client = use_all_data_per_client
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None
        self._client_to_indices: Optional[Dict[int, np.ndarray]] = None
        self._scaler: Optional[StandardScaler] = None
        self._target_encodings: Dict[str, Dict[str, float]] = {}
        self._feature_names: List[str] = BANK_FEATURE_NAMES.copy()

    def _target_encode_column(
        self, series: pd.Series, y: np.ndarray,
        smoothing: int = TARGET_ENCODE_SMOOTHING,
    ) -> np.ndarray:
        """Replace categories with smoothed mean of target."""
        series = series.astype(str).fillna("missing")
        global_mean = float(np.mean(y))
        agg = pd.DataFrame({"y": y, "cat": series}).groupby("cat")["y"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        mapping = smoothed.to_dict()
        self._target_encodings[series.name] = mapping
        return series.map(lambda c: mapping.get(c, global_mean)).values.astype(np.float64)

    def _load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = fetch_openml(data_id=1461, as_frame=True, parser="auto")
        df = data.frame

        # Identify target column
        target_candidates = ['y', 'Class', 'class', 'V17']
        target_col = None
        for tc in target_candidates:
            if tc in df.columns:
                target_col = tc
                break
        if target_col is None:
            target_col = df.columns[-1]

        y_raw = df[target_col].astype(str).str.strip().str.lower()
        y = (y_raw.isin(['yes', '1', '2'])).astype(np.int64)
        df = df.drop(columns=[target_col])

        # Normalise column names to lowercase for matching
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Remove duration (target leakage)
        for dur_name in ['duration', 'v12']:
            if dur_name in df.columns:
                df = df.drop(columns=[dur_name])

        # Map actual columns to canonical feature names
        # OpenML bank-marketing may use V1..V16 or actual names
        canonical_map = {}
        expected_names_lower = [f.lower() for f in BANK_FEATURE_NAMES]
        for col in df.columns:
            if col in expected_names_lower:
                canonical_map[col] = col

        if len(canonical_map) < len(BANK_FEATURE_NAMES) // 2:
            # Columns are likely V1, V2, ... positional; map by position
            # Bank Marketing order: age,job,marital,education,default,balance,
            #   housing,loan,contact,day,month, [duration removed], campaign,pdays,previous,poutcome
            positional = [c for c in df.columns if c not in (target_col,)]
            feature_cols = positional[:len(BANK_FEATURE_NAMES)]
            rename_map = {old: new for old, new in zip(feature_cols, BANK_FEATURE_NAMES)}
            df = df.rename(columns=rename_map)
        # Keep only feature columns
        feature_cols = [f for f in BANK_FEATURE_NAMES if f in df.columns]
        if not feature_cols:
            raise ValueError(f"Could not find expected Bank Marketing features in columns: {list(df.columns)}")

        # Drop rows with missing values
        df = df[feature_cols].copy()
        df = df.dropna()
        y = y.loc[df.index].values

        # Build X
        X_list = []
        for f in feature_cols:
            if f in NUMERIC_COLS:
                X_list.append(pd.to_numeric(df[f], errors='coerce').fillna(0).values.astype(np.float64))
            else:
                col_series = df[f].copy()
                col_series.name = f
                encoded = self._target_encode_column(col_series, y)
                X_list.append(encoded)
        X = np.column_stack(X_list)
        self._feature_names = feature_cols
        self.n_features = len(feature_cols)

        # Job column for client partitioning
        job_col = df['job'].astype(str).values if 'job' in df.columns else df.iloc[:, 1].astype(str).values
        return X, y, job_col

    def _build_client_indices(
        self, X: np.ndarray, partition_labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        uniq = np.unique(partition_labels)
        label_to_client = {lab: i % self.n_clients for i, lab in enumerate(uniq)}
        client_indices: Dict[int, List[int]] = {c: [] for c in range(self.n_clients)}
        for idx, lab in enumerate(partition_labels):
            c = label_to_client[lab]
            client_indices[c].append(idx)

        out: Dict[int, np.ndarray] = {}
        for c in range(self.n_clients):
            ind = np.array(client_indices[c], dtype=np.int64)
            if len(ind) == 0:
                ind = self._rng.choice(len(partition_labels),
                                       size=min(self.n_samples_per_client, len(partition_labels)),
                                       replace=False)
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
        X, y, job_labels = self._load_and_preprocess()
        self._client_to_indices = self._build_client_indices(X, job_labels)
        self._scaler = StandardScaler()
        self._X_full = self._scaler.fit_transform(X.astype(np.float64))
        self._y_full = y

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def get_drifted_feature_indices(self) -> Set[int]:
        # Default: balance (index 5)
        return {self._feature_names.index('balance')} if 'balance' in self._feature_names else {5}

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
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size,
                random_state=self.seed + client_id,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train.astype(np.float32),
                y_train=y_train,
                X_val=X_val.astype(np.float32),
                y_val=y_val,
                is_drifted=generate_drifted and (client_id in drifted_clients),
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
