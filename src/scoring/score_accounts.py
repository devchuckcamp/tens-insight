"""Batch scoring script for account churn predictions.

Scores all active accounts and writes predictions to account_risk_scores table.

Usage:
    python -m src.scoring.score_accounts
"""

import logging
import sys
from datetime import datetime

import pandas as pd

from ..config import get_config
from ..features.accounts import build_account_features, prepare_training_data
from ..models.churn_model import create_churn_model
from ..db import upsert_dataframe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def score_accounts(
    lookback_days: int = 90,
    model_version: str = "v1",
    batch_size: int = 1000
):
    """Score all active accounts and write predictions to database.
    
    Args:
        lookback_days: Number of days to look back for features
        model_version: Version identifier for the model to load
        batch_size: Number of accounts to score at once
    """
    logger.info("=" * 60)
    logger.info("Starting account scoring")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Step 1: Load model
    logger.info(f"Loading churn model (version {model_version})...")
    try:
        # We need to know input_dim to create the model, so we'll build features first
        # and then load the model with the correct dimension
        features_df = build_account_features(lookback_days=lookback_days)
        
        if features_df.empty:
            logger.error("No accounts found to score")
            sys.exit(1)
        
        logger.info(f"Found {len(features_df)} accounts to score")
        
        # Prepare features
        X, _ = prepare_training_data(features_df, target_column='churned')
        
        # Create and load model
        model = create_churn_model(input_dim=X.shape[1], model_version=model_version)
        model.load(model_name="churn_model")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Step 2: Generate predictions
    logger.info("Generating predictions...")
    churn_probs, health_scores = model.predict_with_health_score(X.values)
    
    # Step 3: Prepare output DataFrame
    logger.info("Preparing predictions for database write...")
    
    predictions_df = pd.DataFrame({
        'account_id': features_df['account_id'],
        'churn_probability': churn_probs,
        'health_score': health_scores,
        'risk_category': [model.get_risk_category(p) for p in churn_probs],
        'predicted_at': datetime.now(),
        'model_version': model_version
    })
    
    # Step 4: Display sample predictions
    logger.info("=" * 60)
    logger.info("Sample Predictions:")
    logger.info("=" * 60)
    
    sample = predictions_df.nlargest(10, 'churn_probability')
    for idx, row in sample.iterrows():
        logger.info(
            f"  Account: {row['account_id']:15s} | "
            f"Churn: {row['churn_probability']:.3f} | "
            f"Health: {row['health_score']:.1f} | "
            f"Risk: {row['risk_category']}"
        )
    
    logger.info("=" * 60)
    
    # Step 5: Display distribution statistics
    logger.info("Prediction Statistics:")
    logger.info(f"  Total accounts: {len(predictions_df)}")
    logger.info(f"  Mean churn probability: {churn_probs.mean():.3f}")
    logger.info(f"  Mean health score: {health_scores.mean():.1f}")
    logger.info("=" * 60)
    
    # Risk category distribution
    risk_dist = predictions_df['risk_category'].value_counts().sort_index()
    logger.info("Risk Category Distribution:")
    for category, count in risk_dist.items():
        pct = count / len(predictions_df) * 100
        logger.info(f"  {category:10s}: {count:3d} ({pct:5.1f}%)")
    logger.info("=" * 60)
    
    # Step 6: Write to database
    logger.info("Writing predictions to database...")
    try:
        rows_written = upsert_dataframe(
            predictions_df,
            table_name='account_risk_scores',
            conflict_columns=['account_id'],
            update_columns=['churn_probability', 'health_score', 'risk_category', 
                          'predicted_at', 'model_version']
        )
        logger.info(f"Successfully wrote {rows_written} predictions to account_risk_scores")
    except Exception as e:
        logger.error(f"Failed to write predictions to database: {e}")
        sys.exit(1)
    
    # Step 7: Summary
    logger.info("=" * 60)
    logger.info("Scoring Summary:")
    logger.info(f"  Accounts scored: {len(predictions_df)}")
    logger.info(f"  High/Critical risk: {(predictions_df['churn_probability'] >= 0.5).sum()}")
    logger.info(f"  Low/Medium risk: {(predictions_df['churn_probability'] < 0.5).sum()}")
    logger.info(f"  Model version: {model_version}")
    logger.info(f"  Scored at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("Scoring complete!")
    
    return predictions_df


if __name__ == '__main__':
    """Run scoring when module is executed directly."""
    score_accounts()
