"""Training script for the account churn prediction model.

Usage:
    python -m src.training.train_churn
"""

import logging
import sys
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from ..config import get_config
from ..features.accounts import (
    build_account_features,
    create_synthetic_labels,
    prepare_training_data
)
from ..models.churn_model import create_churn_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_churn_model(
    lookback_days: int = 90,
    test_size: float = None,
    model_version: str = "v1"
):
    """Train the account churn prediction model.
    
    Args:
        lookback_days: Number of days to look back for features
        test_size: Fraction of data to use for testing (uses config default if None)
        model_version: Version identifier for the model
    """
    logger.info("=" * 60)
    logger.info("Starting churn model training")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Use config values if not explicitly provided
    if test_size is None:
        test_size = 1 - config.validation_split
    
    np.random.seed(config.random_seed)
    
    # Step 1: Build features
    logger.info(f"Building account features with {lookback_days} day lookback...")
    features_df = build_account_features(lookback_days=lookback_days)
    
    if features_df.empty:
        logger.error("No data available for training. Ensure feedback_enriched table has data.")
        sys.exit(1)
    
    logger.info(f"Built features for {len(features_df)} accounts")
    
    # Step 2: Create synthetic labels (in production, use real churn data)
    logger.info("Creating synthetic churn labels...")
    features_df['churned'] = create_synthetic_labels(features_df)
    
    # Step 3: Prepare training data
    logger.info("Preparing training data...")
    X, y = prepare_training_data(features_df, target_column='churned')
    
    if y is None:
        logger.error("Failed to extract target variable")
        sys.exit(1)
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: churned={y.sum()}, retained={len(y) - y.sum()}")
    
    # Step 4: Split data
    logger.info(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=config.random_seed,
        stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Step 5: Create and train model
    logger.info(f"Creating churn model (version {model_version})...")
    model = create_churn_model(input_dim=X_train.shape[1], model_version=model_version)
    model.build_model()
    
    logger.info("Training model...")
    history = model.train(
        X_train.values,
        y_train.values,
        X_val=X_test.values,
        y_val=y_test.values
    )
    
    # Step 6: Evaluate on test set
    logger.info("Evaluating model on test set...")
    churn_probs, health_scores = model.predict_with_health_score(X_test.values)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = (churn_probs >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, churn_probs)
    }
    
    logger.info("=" * 60)
    logger.info("Test Set Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 60)
    
    # Step 7: Save model
    logger.info("Saving model...")
    model.save(model_name="churn_model")
    logger.info(f"Model saved to {config.models_dir}/")
    
    # Step 8: Summary
    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info(f"  Model version: {model_version}")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC: {metrics['auc']:.4f}")
    logger.info(f"  Trained at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("Training complete!")
    
    return model, metrics


if __name__ == '__main__':
    """Run training when module is executed directly."""
    train_churn_model()
