"""
Agrawal dataset generator for federated learning with concept drift.
"""

from typing import Dict, List, Set, Optional
import numpy as np
from river.datasets import synth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import BaseDataGenerator, ClientDataset


class AgrawalDataGenerator(BaseDataGenerator):
    """
    Generates Agrawal datasets for FL with configurable concept drift.
    
    The Agrawal dataset simulates salary classification based on person attributes.
    Concept drift is simulated by switching the classification function.
    
    Classification functions define different decision rules:
    - Function 0: age < 40 OR age >= 60
    - Function 1: age >= 40 AND age < 60 AND elevel >= 3
    - Function 2: age >= 40 AND age < 60 AND elevel >= 2 AND salary >= 50000
    - etc.
    
    Different functions emphasize different features, enabling controlled drift.
    """
    
    FEATURE_NAMES = ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']
    
    FUNCTION_RELEVANT_FEATURES = {
        0: {2},
        1: {2, 3},
        2: {2, 3, 0},
        3: {2, 3, 0, 1},
        4: {2, 0, 1, 6},
        5: {2, 0, 4},
        6: {2, 0, 3, 7},
        7: {2, 0, 6, 8},
        8: {2, 0, 1, 5},
        9: {2, 0, 1, 6, 8},
    }
    
    def __init__(
        self,
        n_clients: int,
        n_samples_per_client: int,
        classification_function_pre: int = 0,
        classification_function_post: int = 2,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(
            n_clients=n_clients,
            n_samples_per_client=n_samples_per_client,
            n_features=len(self.FEATURE_NAMES),
            test_size=test_size,
            seed=seed,
        )
        
        self.classification_function_pre = classification_function_pre
        self.classification_function_post = classification_function_post
        
        pre_features = self.FUNCTION_RELEVANT_FEATURES.get(classification_function_pre, set())
        post_features = self.FUNCTION_RELEVANT_FEATURES.get(classification_function_post, set())
        self._drifted_feature_indices = pre_features.symmetric_difference(post_features)
        
        self._label_encoders: Dict[str, LabelEncoder] = {}
        
        self._scaler: Optional[StandardScaler] = None
        self._scaler_fitted = False
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.FEATURE_NAMES.copy()
    
    def get_drifted_feature_indices(self) -> Set[int]:
        """Return indices of features that drift (change in importance)."""
        return self._drifted_feature_indices
    
    def _process_sample(self, x: dict, y: int) -> List:
        """Process a single sample, encoding categorical features."""
        features = []
        for fname in self.FEATURE_NAMES:
            val = x[fname]
            if fname in ['elevel', 'car', 'zipcode']:
                if fname not in self._label_encoders:
                    self._label_encoders[fname] = LabelEncoder()
                    if fname == 'elevel':
                        self._label_encoders[fname].fit([0, 1, 2, 3, 4])
                    elif fname == 'car':
                        self._label_encoders[fname].fit(list(range(20)))
                    elif fname == 'zipcode':
                        self._label_encoders[fname].fit(list(range(9)))
                try:
                    val = self._label_encoders[fname].transform([val])[0]
                except ValueError:
                    val = 0
            features.append(val)
        features.append(y)
        return features
    
    def generate_round_data(
        self,
        round_num: int,
        t0: int,
        drifted_clients: Set[int],
        drift_magnitude: float,
    ) -> Dict[int, ClientDataset]:
        """
        Generate data for all clients for a given round.
        
        For Agrawal, concept drift affects ALL clients simultaneously (global concept shift).
        The drifted_clients parameter is ignored - all clients switch classification function at t0.
        
        Args:
            round_num: Current FL round (1-indexed)
            t0: Drift onset round
            drifted_clients: Ignored for Agrawal (all clients drift)
            drift_magnitude: Not used (concept drift is function switch)
        
        Returns:
            Dictionary mapping client_id to ClientDataset
        """
        is_drift_phase = round_num >= t0
        client_datasets = {}
        
        for client_id in range(self.n_clients):
            if is_drift_phase:
                clf_fn = self.classification_function_post
            else:
                clf_fn = self.classification_function_pre
            
            stream = synth.Agrawal(
                classification_function=clf_fn,
                seed=self.seed + client_id + round_num * 1000,
            )
            
            data = []
            for x, y in stream.take(self.n_samples_per_client):
                data.append(self._process_sample(x, y))
            
            data = np.array(data, dtype=np.float64)
            X = data[:, :-1]
            y = data[:, -1].astype(int)
            
            if not self._scaler_fitted and client_id == 0:
                self._scaler = StandardScaler()
                self._scaler.fit(X)
                self._scaler_fitted = True
            
            if self._scaler is not None:
                X = self._scaler.transform(X).astype(np.float32)
            else:
                X = X.astype(np.float32)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                is_drifted=is_drift_phase,
            )
        
        return client_datasets
    
    def generate_static_client_data(
        self,
        drifted_clients: Set[int],
        generate_drifted: bool = False,
    ) -> Dict[int, ClientDataset]:
        """
        Generate static data for clients.
        
        Args:
            drifted_clients: Set of client IDs that will experience drift
            generate_drifted: If True, generate post-drift data; else pre-drift
        
        Returns:
            Dictionary mapping client_id to ClientDataset
        """
        client_datasets = {}
        
        for client_id in range(self.n_clients):
            is_drifted_client = client_id in drifted_clients
            
            if generate_drifted and is_drifted_client:
                clf_fn = self.classification_function_post
            else:
                clf_fn = self.classification_function_pre
            
            stream = synth.Agrawal(
                classification_function=clf_fn,
                seed=self.seed + client_id,
            )
            
            data = []
            for x, y in stream.take(self.n_samples_per_client):
                data.append(self._process_sample(x, y))
            
            data = np.array(data, dtype=np.float64)
            X = data[:, :-1]
            y = data[:, -1].astype(int)
            
            if not self._scaler_fitted and client_id == 0:
                self._scaler = StandardScaler()
                self._scaler.fit(X)
                self._scaler_fitted = True
            
            if self._scaler is not None:
                X = self._scaler.transform(X).astype(np.float32)
            else:
                X = X.astype(np.float32)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                is_drifted=generate_drifted and is_drifted_client,
            )
        
        return client_datasets
