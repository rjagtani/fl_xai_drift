"""
Data generation modules for synthetic and real-world drift datasets.
"""

from .hyperplane import HyperplaneDataGenerator
from .agrawal import AgrawalDataGenerator
from .wine_quality import WineQualityDataGenerator
from .fed_heart import FedHeartDataGenerator
from .elec2 import Elec2DataGenerator
from .diabetes import DiabetesDataGenerator
from .credit import CreditDataGenerator
from .adult import AdultDataGenerator
from .bank_marketing import BankMarketingDataGenerator
from .base import BaseDataGenerator, ClientDataset

__all__ = [
    'HyperplaneDataGenerator',
    'AgrawalDataGenerator',
    'WineQualityDataGenerator',
    'FedHeartDataGenerator',
    'Elec2DataGenerator',
    'DiabetesDataGenerator',
    'CreditDataGenerator',
    'AdultDataGenerator',
    'BankMarketingDataGenerator',
    'BaseDataGenerator',
    'ClientDataset',
]
