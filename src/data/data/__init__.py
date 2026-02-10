"""
Data generation modules for synthetic and real-world drift datasets.
"""

from .hyperplane import HyperplaneDataGenerator
from .agrawal import AgrawalDataGenerator
from .adult import AdultDataGenerator
from .wine_quality import WineQualityDataGenerator
from .fed_heart import FedHeartDataGenerator
from .elec2 import Elec2DataGenerator
from .base import BaseDataGenerator, ClientDataset

__all__ = [
    'HyperplaneDataGenerator',
    'AgrawalDataGenerator',
    'AdultDataGenerator',
    'WineQualityDataGenerator',
    'FedHeartDataGenerator',
    'Elec2DataGenerator',
    'BaseDataGenerator',
    'ClientDataset',
]
