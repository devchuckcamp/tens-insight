"""Product area impact scoring model.

Rule-based scoring model for prioritizing product areas based on
feedback patterns and customer segments.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from ..config import get_config

logger = logging.getLogger(__name__)


class ProductAreaModel:
    """Product area impact scoring model.
    
    This is primarily a rule-based model that uses weighted aggregation
    of feedback signals to compute priority scores. For future iterations,
    this could be replaced with a learned model.
    """
    
    def __init__(self, model_version: str = "v1"):
        """Initialize the product area model.
        
        Args:
            model_version: Version identifier for the model
        """
        self.model_version = model_version
        self.config = get_config()
        
        # Weights for priority calculation
        self.weights = {
            'feedback_volume': 0.15,
            'negative_sentiment': 0.25,
            'critical_issues': 0.20,
            'p5_issues': 0.15,
            'recent_activity': 0.10,
            'enterprise_impact': 0.15
        }
    
    def calculate_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate priority scores for product areas.
        
        Args:
            features_df: DataFrame with product area features
            
        Returns:
            DataFrame with priority scores added
        """
        logger.info(f"Calculating priority scores for {len(features_df)} product areas")
        
        # Normalize features to 0-1 range
        def normalize(series):
            """Min-max normalization."""
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - min_val) / (max_val - min_val)
        
        # Normalize key metrics
        norm_feedback = normalize(features_df['feedback_count'])
        norm_negative = normalize(features_df['negative_count'])
        norm_critical = normalize(features_df['critical_count'])
        norm_p5 = normalize(features_df['p5_count'])
        norm_recent = normalize(features_df['feedback_last_7d'])
        
        # Calculate weighted priority score (0-100 scale)
        priority_score = (
            norm_feedback * self.weights['feedback_volume'] * 100 +
            norm_negative * self.weights['negative_sentiment'] * 100 +
            norm_critical * self.weights['critical_issues'] * 100 +
            norm_p5 * self.weights['p5_issues'] * 100 +
            norm_recent * self.weights['recent_activity'] * 100 +
            features_df['enterprise_flag'] * self.weights['enterprise_impact'] * 100
        )
        
        features_df['priority_score'] = priority_score
        
        # Add risk category
        features_df['risk_category'] = pd.cut(
            features_df['priority_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['low', 'medium', 'high', 'critical'],
            include_lowest=True
        )
        
        logger.info(
            f"Scores calculated: min={priority_score.min():.2f}, "
            f"max={priority_score.max():.2f}, mean={priority_score.mean():.2f}"
        )
        
        return features_df
    
    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict priority scores for a batch of product areas.
        
        Args:
            features_df: DataFrame with product area features
            
        Returns:
            DataFrame with predictions
        """
        return self.calculate_scores(features_df)
    
    def save(self, model_name: str = "product_area_model"):
        """Save the model configuration.
        
        Args:
            model_name: Base name for saved files
        """
        models_dir = self.config.models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model configuration (weights)
        config_path = os.path.join(models_dir, f"{model_name}_{self.model_version}_config.pkl")
        joblib.dump({
            'weights': self.weights,
            'model_version': self.model_version
        }, config_path)
        logger.info(f"Saved model config to {config_path}")
    
    def load(self, model_name: str = "product_area_model"):
        """Load a saved model configuration.
        
        Args:
            model_name: Base name of saved files
        """
        models_dir = self.config.models_dir
        
        config_path = os.path.join(models_dir, f"{model_name}_{self.model_version}_config.pkl")
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.weights = config.get('weights', self.weights)
            logger.info(f"Loaded model config from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
    
    def update_weights(self, new_weights: dict):
        """Update scoring weights.
        
        Args:
            new_weights: Dictionary of weight updates
        """
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value
                logger.info(f"Updated weight '{key}' to {value}")
            else:
                logger.warning(f"Unknown weight key: {key}")
    
    def get_top_priorities(
        self,
        features_df: pd.DataFrame,
        n: int = 10,
        min_score: float = 50.0
    ) -> pd.DataFrame:
        """Get top priority product areas.
        
        Args:
            features_df: DataFrame with product area features
            n: Number of top priorities to return
            min_score: Minimum priority score threshold
            
        Returns:
            DataFrame with top n priorities above min_score
        """
        scored_df = self.calculate_scores(features_df)
        
        # Filter by minimum score and sort
        top_priorities = scored_df[
            scored_df['priority_score'] >= min_score
        ].sort_values('priority_score', ascending=False).head(n)
        
        logger.info(f"Found {len(top_priorities)} product areas above threshold {min_score}")
        
        return top_priorities


def create_product_area_model(model_version: str = "v1") -> ProductAreaModel:
    """Factory function to create a product area model instance.
    
    Args:
        model_version: Version identifier
        
    Returns:
        ProductAreaModel instance
    """
    return ProductAreaModel(model_version=model_version)
