"""
Adult (UCI Census) dataset for federated learning with feature-conditioned concept drift.
Clients partitioned by education. Categoricals are target-encoded. education-num excluded (redundant with education).
Drift: label flip with prob when condition (e.g. age > 30) holds; can apply to all clients.
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

from .base import BaseDataGenerator, ClientDataset

# Exclude education-num (redundant with education); education used only for client split
ADULT_FEATURE_NAMES = [
    'age', 'workclass', 'fnlwgt', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
]
COL_ALIASES = {
    'education.num': 'education-num', 'capital.gain': 'capital-gain', 'capital.loss': 'capital-loss',
    'hours.per.week': 'hours-per-week', 'marital.status': 'marital-status', 'native.country': 'native-country',
}
NUMERIC_COLS = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
TARGET_ENCODE_SMOOTHING = 1  # Blend category mean with global mean to reduce overfitting


class AdultDataGenerator(BaseDataGenerator):
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        test_size: float = 0.2,
        seed: int = 42,
        drift_condition_feature: str = "age",
        drift_condition_threshold: float = 50.0,
        drift_flip_prob: float = 0.3,
        use_all_data_per_client: bool = False,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=len(ADULT_FEATURE_NAMES),
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
        self._condition_col: int = 0  # Index of drift condition feature (resolved after loading)
        self._scaler: Optional[StandardScaler] = None
        self._target_encodings: Dict[str, Dict[str, float]] = {}  # col -> {category: encoded_value}
        self._feature_names: List[str] = ADULT_FEATURE_NAMES.copy()
        self._raw_condition: Optional[np.ndarray] = None  # Raw values of drift condition feature

    def _target_encode_column(self, series: pd.Series, y: np.ndarray, smoothing: int = TARGET_ENCODE_SMOOTHING) -> np.ndarray:
        """Replace categories with smoothed mean of target (blend with global mean)."""
        series = series.astype(str).fillna("missing")
        global_mean = float(np.mean(y))
        agg = pd.DataFrame({"y": y, "cat": series}).groupby("cat")["y"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        mapping = smoothed.to_dict()
        return series.map(lambda c: mapping.get(c, global_mean)).values.astype(np.float64)

    def _load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.frame
        df.columns = [COL_ALIASES.get(str(c), str(c)) for c in df.columns]
        target_col = "class" if "class" in df.columns else "income"
        y_series = (df[target_col].astype(str).str.contains(">50K", na=False)).astype(np.int64)
        df = df.drop(columns=[target_col])
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].replace("?", np.nan)
        df = df.dropna()
        y = y_series.loc[df.index].values
        education_col = "education" if "education" in df.columns else "education-num"
        education_series = df[education_col].astype(str)
        feature_cols = [f for f in ADULT_FEATURE_NAMES if f in df.columns]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c != education_col and c != "education-num"]
        X_list = []
        for f in feature_cols:
            if f in NUMERIC_COLS:
                X_list.append(df[f].values.astype(np.float64))
            else:
                encoded = self._target_encode_column(df[f], y)
                X_list.append(encoded)
        X = np.column_stack(X_list)
        self._feature_names = feature_cols
        self.n_features = len(feature_cols)
        # Resolve drift condition column index
        if self.drift_condition_feature in feature_cols:
            self._condition_col = feature_cols.index(self.drift_condition_feature)
        else:
            self._condition_col = feature_cols.index("age") if "age" in feature_cols else 0
        education_labels = education_series.values
        return X, y, education_labels

    def _build_client_indices(self, X: np.ndarray, education_labels: np.ndarray) -> Dict[int, np.ndarray]:
        uniq = np.unique(education_labels)
        education_to_client: Dict[str, int] = {}
        for i, ed in enumerate(uniq):
            education_to_client[ed] = i % self.n_clients
        client_indices: Dict[int, List[int]] = {c: [] for c in range(self.n_clients)}
        for idx, ed in enumerate(education_labels):
            c = education_to_client[ed]
            client_indices[c].append(idx)
        out: Dict[int, np.ndarray] = {}
        for c in range(self.n_clients):
            ind = np.array(client_indices[c], dtype=np.int64)
            if len(ind) == 0:
                ind = self._rng.choice(len(education_labels), size=min(self.n_samples_per_client, len(education_labels)), replace=False)
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
        X, y, education_labels = self._load_and_preprocess()
        self._client_to_indices = self._build_client_indices(X, education_labels)
        # Store raw condition column BEFORE scaling (for drift condition check)
        self._raw_condition = X[:, self._condition_col].copy()
        # Auto-set threshold as median if threshold is None
        if self.drift_condition_threshold is None:
            self.drift_condition_threshold = float(np.median(self._raw_condition))
        self._scaler = StandardScaler()
        self._X_full = self._scaler.fit_transform(X.astype(np.float64))
        self._y_full = y

    def _apply_drift(self, raw_condition: np.ndarray, y: np.ndarray, client_id: int,
                     flip_prob_override: Optional[float] = None) -> np.ndarray:
        """Apply label flip drift for samples where condition feature > threshold.
        
        Args:
            raw_condition: Raw (pre-scaling) values of the drift condition feature.
            y: Labels.
            client_id: Client ID (for seeding).
            flip_prob_override: If set, use this flip probability instead of self.drift_flip_prob.
                               Useful for gradual drift where the probability ramps up.
        """
        y_out = y.copy()
        flip_prob = flip_prob_override if flip_prob_override is not None else self.drift_flip_prob
        cond = raw_condition > self.drift_condition_threshold
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
        return self._feature_names.copy()

    def get_drifted_feature_indices(self) -> Set[int]:
        return {self._condition_col}

    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
        drift_magnitude: float = 0.0,
        flip_prob_override: Optional[float] = None,
    ) -> Dict[int, ClientDataset]:
        """Generate static data. flip_prob_override allows gradual drift (variable flip prob)."""
        self._ensure_data()
        client_datasets = {}
        for client_id in range(self.n_clients):
            ind = self._client_to_indices[client_id]
            X = self._X_full[ind]
            y = self._y_full[ind]
            raw_cond = self._raw_condition[ind]
            if generate_drifted:
                y = self._apply_drift(raw_cond, y, client_id, flip_prob_override=flip_prob_override)
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
