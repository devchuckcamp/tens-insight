"""Batch scoring script for product area impact predictions.

Scores all product areas by segment and writes predictions to 
product_area_impact table.

Usage:
    python -m src.scoring.score_product_areas
"""

import logging
import sys
from datetime import datetime

import pandas as pd

from ..config import get_config
from ..features.product_areas import build_product_area_features
from ..models.product_area_model import create_product_area_model
from ..db import upsert_dataframe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def score_product_areas(
    lookback_days: int = 90,
    model_version: str = "v1"
):
    """Score all product areas and write predictions to database.
    
    Args:
        lookback_days: Number of days to look back for features
        model_version: Version identifier for the model to load
    """
    logger.info("=" * 60)
    logger.info("Starting product area scoring")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Step 1: Load model
    logger.info(f"Loading product area model (version {model_version})...")
    try:
        model = create_product_area_model(model_version=model_version)
        model.load(model_name="product_area_model")
    except Exception as e:
        logger.warning(f"Could not load saved model: {e}. Using default configuration.")
        model = create_product_area_model(model_version=model_version)
    
    # Step 2: Build features
    logger.info(f"Building product area features with {lookback_days} day lookback...")
    features_df = build_product_area_features(lookback_days=lookback_days)
    
    if features_df.empty:
        logger.error("No product area data found to score")
        sys.exit(1)
    
    logger.info(f"Found {len(features_df)} product-area/segment combinations to score")
    
    # Step 3: Generate predictions
    logger.info("Generating priority scores...")
    scored_df = model.calculate_scores(features_df)
    
    # Step 4: Prepare output DataFrame
    logger.info("Preparing predictions for database write...")
    
    predictions_df = pd.DataFrame({
        'product_area': scored_df['product_area'],
        'segment': scored_df['segment'],
        'priority_score': scored_df['priority_score'],
        'feedback_count': scored_df['feedback_count'],
        'avg_sentiment_score': scored_df['avg_sentiment_score'],
        'negative_count': scored_df['negative_count'],
        'critical_count': scored_df['critical_count'],
        'predicted_at': datetime.now(),
        'model_version': model_version
    })
    
    # Step 5: Display sample predictions
    logger.info("=" * 60)
    logger.info("Top Priority Product Areas:")
    logger.info("=" * 60)
    
    top_priorities = predictions_df.nlargest(15, 'priority_score')
    for idx, row in top_priorities.iterrows():
        logger.info(
            f"  {row['product_area']:15s} | {row['segment']:10s} | "
            f"Score: {row['priority_score']:5.1f} | "
            f"Feedback: {row['feedback_count']:3d} | "
            f"Negative: {row['negative_count']:3d} | "
            f"Critical: {row['critical_count']:2d}"
        )
    
    logger.info("=" * 60)
    
    # Step 6: Display distribution statistics
    logger.info("Score Statistics:")
    logger.info(f"  Total combinations: {len(predictions_df)}")
    logger.info(f"  Product areas: {predictions_df['product_area'].nunique()}")
    logger.info(f"  Segments: {predictions_df['segment'].nunique()}")
    logger.info(f"  Mean priority score: {predictions_df['priority_score'].mean():.2f}")
    logger.info(f"  High priority (>50): {(predictions_df['priority_score'] >= 50).sum()}")
    logger.info("=" * 60)
    
    # Priority by product area
    logger.info("Priority by Product Area (average):")
    area_priority = predictions_df.groupby('product_area')['priority_score'].mean().sort_values(ascending=False)
    for area, score in area_priority.items():
        logger.info(f"  {area:15s}: {score:5.1f}")
    logger.info("=" * 60)
    
    # Priority by segment
    logger.info("Priority by Segment (average):")
    segment_priority = predictions_df.groupby('segment')['priority_score'].mean().sort_values(ascending=False)
    for segment, score in segment_priority.items():
        logger.info(f"  {segment:10s}: {score:5.1f}")
    logger.info("=" * 60)
    
    # Step 7: Write to database
    logger.info("Writing predictions to database...")
    try:
        rows_written = upsert_dataframe(
            predictions_df,
            table_name='product_area_impact',
            conflict_columns=['product_area', 'segment'],
            update_columns=['priority_score', 'feedback_count', 'avg_sentiment_score',
                          'negative_count', 'critical_count', 'predicted_at', 'model_version']
        )
        logger.info(f"Successfully wrote {rows_written} predictions to product_area_impact")
    except Exception as e:
        logger.error(f"Failed to write predictions to database: {e}")
        sys.exit(1)
    
    # Step 8: Summary
    logger.info("=" * 60)
    logger.info("Scoring Summary:")
    logger.info(f"  Product areas scored: {len(predictions_df)}")
    logger.info(f"  High priority areas: {(predictions_df['priority_score'] >= 50).sum()}")
    logger.info(f"  Model version: {model_version}")
    logger.info(f"  Scored at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("Scoring complete!")
    
    return predictions_df


if __name__ == '__main__':
    """Run scoring when module is executed directly."""
    score_product_areas()
