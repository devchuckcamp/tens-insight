"""Training script for the account churn prediction model.

Usage:
    python -m src.training.train_churn
    python -m src.training.train_churn --incremental  # Auto-check if retraining needed
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
from ..versioning import ModelRegistry
from .incremental_trainer import IncrementalTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_churn_model(
    lookback_days: int = 90,
    test_size: float = None,
    model_version: str = None,
    incremental: bool = False,
    promotion_threshold: float = 0.01
):
    """Train the account churn prediction model.
    
    Args:
        lookback_days: Number of days to look back for features
        test_size: Fraction of data to use for testing (uses config default if None)
        model_version: Version identifier for the model (auto-generated if None)
        incremental: If True, check data freshness before training
        promotion_threshold: Minimum improvement required to promote to production (0.0-1.0)
    
    Returns:
        Tuple of (model, metrics, version_info)
    """
    logger.info("=" * 60)
    logger.info("Starting churn model training")
    logger.info("=" * 60)
    
    config = get_config()
    registry = ModelRegistry()
    trainer = IncrementalTrainer()
    
    # Check if retraining is necessary (incremental mode)
    if incremental:
        logger.info("Incremental training mode: checking data freshness...")
        should_retrain, reason = trainer.check_training_necessity(
            model_type='churn',
            min_new_data_percent=5.0,
            min_days_since_training=7,
            min_new_records=100
        )
        
        if not should_retrain:
            logger.info(f"Skipping training: {reason}")
            return None, {}, {"skipped": True, "reason": reason}
        
        logger.info(f"Proceeding with training: {reason}")
    
    # Auto-generate version if not provided
    if model_version is None:
        model_version = trainer.get_next_version('churn')
        logger.info(f"Auto-generated version: {model_version}")
    
    # Create version entry in registry
    hyperparameters = {
        "lookback_days": lookback_days,
        "test_size": test_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "validation_split": config.validation_split
    }
    
    version_entry = registry.create_version(
        model_type='churn',
        version=model_version,
        created_by='scheduler' if incremental else 'manual',
        description=f"Training with {lookback_days}d lookback",
        hyperparameters=hyperparameters
    )
    
    # Start training run tracking
    training_run = registry.start_training_run(
        model_version_id=version_entry.id,
        triggered_by='scheduler' if incremental else 'manual'
    )
    
    np.random.seed(config.random_seed)
    
    try:
        # Step 1: Build features
        logger.info(f"Building account features with {lookback_days} day lookback...")
        sys.stdout.flush()  # Ensure log is written before potential crash
        features_df = build_account_features(lookback_days=lookback_days)
        logger.info(f"Features built successfully: {features_df.shape}")
        sys.stdout.flush()
        
        if features_df.empty:
            error_msg = "No data available for training. Ensure feedback_enriched table has data."
            logger.error(error_msg)
            registry.complete_training_run(
                training_run_id=training_run.id,
                status='failed',
                training_samples=0,
                test_samples=0,
                error_message=error_msg
            )
            sys.exit(1)
        
        logger.info(f"Built features for {len(features_df)} accounts")
        
        # Check minimum dataset size
        MIN_ACCOUNTS = 20
        if len(features_df) < MIN_ACCOUNTS:
            error_msg = f"Insufficient data for training: {len(features_df)} accounts (minimum {MIN_ACCOUNTS} required)"
            logger.error(error_msg)
            logger.error("Please add more accounts to the feedback_enriched table before training.")
            registry.complete_training_run(
                training_run_id=training_run.id,
                status='failed',
                training_samples=0,
                test_samples=0,
                error_message=error_msg
            )
            sys.exit(1)
        
        # Step 2: Create synthetic labels (in production, use real churn data)
        logger.info("Creating synthetic churn labels...")
        features_df['churned'] = create_synthetic_labels(features_df)
        
        # Step 3: Prepare training data
        logger.info("Preparing training data...")
        X, y = prepare_training_data(features_df, target_column='churned')
        
        if y is None:
            error_msg = "Failed to extract target variable"
            logger.error(error_msg)
            registry.complete_training_run(
                training_run_id=training_run.id,
                status='failed',
                training_samples=0,
                test_samples=0,
                error_message=error_msg
            )
            sys.exit(1)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: churned={y.sum()}, retained={len(y) - y.sum()}")
        
        # Use config values if not explicitly provided
        if test_size is None:
            test_size = 1 - config.validation_split
        
        # For small datasets, ensure we have at least 2 samples per class in training
        min_samples_per_class = min(y.sum(), len(y) - y.sum())
        if min_samples_per_class < 4:
            # Very small dataset - use 20% test size to ensure training works
            test_size = 0.2
            logger.warning(f"Small dataset detected ({len(y)} samples, {min_samples_per_class} in minority class). Using test_size={test_size}")
        
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
        
        # Step 7: Log metrics to registry
        for metric_name, metric_value in metrics.items():
            registry.log_metric(
                model_version_id=version_entry.id,
                metric_name=metric_name,
                metric_value=metric_value,
                dataset_type='test',
                training_training_run_id=training_run.id
            )
        
        # Log prediction distribution for health checks
        registry.log_prediction_distribution(
            model_type='churn',
            model_version_id=version_entry.id,
            predictions=churn_probs.tolist()
        )
        
        # Step 8: Save model
        logger.info("Saving model...")
        model.save(model_name="churn_model")
        logger.info(f"Model saved to {config.models_dir}/")
        
        # Update version entry with paths
        version_entry.model_path = f"models/churn_model_{model_version}.keras"
        version_entry.scaler_path = f"models/churn_model_{model_version}_scaler.pkl"
        
        # Complete training run
        registry.complete_training_run(
            training_run_id=training_run.id,
            status='completed',
            training_samples=len(X_train),
            test_samples=len(X_test)
        )
        
        # Step 9: Determine if should promote to production
        should_promote = True
        comparison_result = None
        
        if incremental:
            previous_version_id = trainer.get_previous_version_id('churn')
            if previous_version_id:
                should_promote, comparison_result = trainer.should_promote_to_production(
                    model_type='churn',
                    new_version_id=version_entry.id,
                    improvement_threshold=promotion_threshold
                )
                
                if should_promote:
                    trainer.promote_version_to_production(
                        model_type='churn',
                        version_id=version_entry.id,
                        promoted_by='scheduler'
                    )
                    logger.info("New version promoted to production")
                else:
                    logger.info(f"New version NOT promoted: {comparison_result}")
            else:
                # First training, auto-promote
                trainer.promote_version_to_production(
                    model_type='churn',
                    version_id=version_entry.id,
                    promoted_by='scheduler'
                )
                logger.info("First version auto-promoted to production")
        else:
            # Manual training: auto-promote
            trainer.promote_version_to_production(
                model_type='churn',
                version_id=version_entry.id,
                promoted_by='manual'
            )
            logger.info("Manual training version promoted to production")
        
        # Step 10: Summary
        logger.info("=" * 60)
        logger.info("Training Summary:")
        logger.info(f"  Model version: {model_version}")
        logger.info(f"  Version ID: {version_entry.id}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Test AUC: {metrics['auc']:.4f}")
        logger.info(f"  Promoted to production: {should_promote}")
        logger.info(f"  Trained at: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        logger.info("Training complete!")
        
        return model, metrics, {
            "version": model_version,
            "version_id": version_entry.id,
            "promoted": should_promote,
            "comparison": comparison_result
        }
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        registry.complete_training_run(
            training_run_id=training_run.id,
            status='failed',
            training_samples=0,
            test_samples=0,
            error_message=str(e)
        )
        raise


if __name__ == '__main__':
    """Run training when module is executed directly."""
    train_churn_model()
