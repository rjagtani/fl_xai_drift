"""
Feature importance computation modules.

Implements SAGE, PFI, and SHAP for federated learning drift diagnosis.
"""

from .sage_importance import SAGEComputer
from .pfi import PFIComputer
from .shap_importance import SHAPComputer
from .base import FeatureImportanceResult

__all__ = [
    'SAGEComputer',
    'PFIComputer',
    'SHAPComputer',
    'FeatureImportanceResult',
]
