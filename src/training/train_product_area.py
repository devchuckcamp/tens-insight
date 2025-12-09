"""Training script for the product area impact model.

Usage:
    python -m src.training.train_product_area
"""

import logging
import sys
from datetime import datetime

from ..config import get_config
from ..features.product_areas import (
    build_product_area_features,
    calculate_priority_scores
)
from ..models.product_area_model import create_product_area_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_product_area_model(
    lookback_days: int = 90,
    model_version: str = "v1"
):
    """Train (or rather, configure) the product area impact model.
    
    Since this is primarily a rule-based model, "training" consists of
    validating the scoring logic and saving the configuration.
    
    Args:
        lookback_days: Number of days to look back for features
        model_version: Version identifier for the model
    """
    logger.info("=" * 60)
    logger.info("Starting product area model training")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Step 1: Build features
    logger.info(f"Building product area features with {lookback_days} day lookback...")
    features_df = build_product_area_features(lookback_days=lookback_days)
    
    if features_df.empty:
        logger.error("No data available for training. Ensure feedback_enriched table has data.")
        sys.exit(1)
    
    logger.info(f"Built features for {len(features_df)} product-area/segment combinations")
    
    # Step 2: Create model
    logger.info(f"Creating product area model (version {model_version})...")
    model = create_product_area_model(model_version=model_version)
    
    # Step 3: Calculate scores to validate
    logger.info("Calculating priority scores...")
    scored_df = model.calculate_scores(features_df)
    
    # Step 4: Display sample results
    logger.info("=" * 60)
    logger.info("Sample Priority Scores:")
    logger.info("=" * 60)
    
    top_priorities = scored_df.nlargest(10, 'priority_score')
    
    for idx, row in top_priorities.iterrows():
        logger.info(
            f"  {row['product_area']:15s} | {row['segment']:10s} | "
            f"Score: {row['priority_score']:5.1f} | "
            f"Feedback: {row['feedback_count']:3d} | "
            f"Negative: {row['negative_count']:3d} | "
            f"Critical: {row['critical_count']:2d}"
        )
    
    logger.info("=" * 60)
    
    # Step 5: Display distribution statistics
    logger.info("Score Distribution:")
    logger.info(f"  Min:    {scored_df['priority_score'].min():.2f}")
    logger.info(f"  Q1:     {scored_df['priority_score'].quantile(0.25):.2f}")
    logger.info(f"  Median: {scored_df['priority_score'].median():.2f}")
    logger.info(f"  Q3:     {scored_df['priority_score'].quantile(0.75):.2f}")
    logger.info(f"  Max:    {scored_df['priority_score'].max():.2f}")
    logger.info(f"  Mean:   {scored_df['priority_score'].mean():.2f}")
    logger.info(f"  Std:    {scored_df['priority_score'].std():.2f}")
    logger.info("=" * 60)
    
    # Step 6: Risk category distribution
    risk_dist = scored_df['risk_category'].value_counts().sort_index()
    logger.info("Risk Category Distribution:")
    for category, count in risk_dist.items():
        pct = count / len(scored_df) * 100
        logger.info(f"  {category:10s}: {count:3d} ({pct:5.1f}%)")
    logger.info("=" * 60)
    
    # Step 7: Save model configuration
    logger.info("Saving model configuration...")
    model.save(model_name="product_area_model")
    logger.info(f"Model config saved to {config.models_dir}/")
    
    # Step 8: Summary
    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info(f"  Model version: {model_version}")
    logger.info(f"  Product areas: {scored_df['product_area'].nunique()}")
    logger.info(f"  Segments: {scored_df['segment'].nunique()}")
    logger.info(f"  Total combinations: {len(scored_df)}")
    logger.info(f"  High/Critical risk: {(scored_df['priority_score'] >= 50).sum()}")
    logger.info(f"  Trained at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("Training complete!")
    
    return model, scored_df


if __name__ == '__main__':
    """Run training when module is executed directly."""
    train_product_area_model()
