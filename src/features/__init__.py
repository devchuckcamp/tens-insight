"""Feature engineering package initialization."""

from .accounts import build_account_features, prepare_training_data, create_synthetic_labels
from .product_areas import (
    build_product_area_features,
    calculate_priority_scores,
    prepare_for_scoring
)

__all__ = [
    'build_account_features',
    'prepare_training_data',
    'create_synthetic_labels',
    'build_product_area_features',
    'calculate_priority_scores',
    'prepare_for_scoring'
]
