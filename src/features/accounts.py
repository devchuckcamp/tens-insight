"""Feature engineering for account-level models.

Builds features for account churn prediction and health scoring
from the feedback_enriched table and other account signals.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ..db import execute_query

logger = logging.getLogger(__name__)


def build_account_features(
    lookback_days: int = 90,
    account_id: Optional[str] = None
) -> pd.DataFrame:
    """Build account-level features for churn prediction.
    
    Features include:
    - Feedback volume and recency
    - Sentiment distribution
    - Priority/severity metrics
    - Product area coverage
    - Trend indicators
    
    Args:
        lookback_days: Number of days to look back for features
        account_id: Optional specific account ID to score
        
    Returns:
        DataFrame with one row per account and feature columns
    """
    logger.info(f"Building account features with {lookback_days} day lookback")
    
    # Since the goinsight schema doesn't have an explicit accounts table yet,
    # we'll aggregate from feedback_enriched using customer_tier as a proxy
    # for account segments. In production, you'd join with an actual accounts table.
    
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Query to build account features from feedback
    query = """
        WITH account_feedback AS (
            SELECT 
                customer_tier as account_id,
                region,
                COUNT(*) as total_feedback,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as feedback_last_30d,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as feedback_last_7d,
                COUNT(*) FILTER (WHERE sentiment = 'negative') as negative_feedback,
                COUNT(*) FILTER (WHERE sentiment = 'positive') as positive_feedback,
                COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral_feedback,
                COUNT(*) FILTER (WHERE priority >= 4) as critical_issues,
                COUNT(*) FILTER (WHERE priority = 5) as p5_issues,
                AVG(priority) as avg_priority,
                MAX(created_at) as last_feedback_date,
                COUNT(DISTINCT product_area) as product_areas_mentioned,
                COUNT(*) FILTER (WHERE product_area = 'billing') as billing_issues,
                COUNT(*) FILTER (WHERE product_area = 'performance') as performance_issues,
                COUNT(*) FILTER (WHERE product_area = 'security') as security_issues
            FROM feedback_enriched
            WHERE created_at >= %(cutoff_date)s
            GROUP BY customer_tier, region
        )
        SELECT 
            account_id,
            region,
            total_feedback,
            feedback_last_30d,
            feedback_last_7d,
            negative_feedback,
            positive_feedback,
            neutral_feedback,
            critical_issues,
            p5_issues,
            avg_priority,
            last_feedback_date,
            product_areas_mentioned,
            billing_issues,
            performance_issues,
            security_issues
        FROM account_feedback
    """
    
    if account_id:
        query += " WHERE account_id = %(account_id)s"
        params = {'cutoff_date': cutoff_date, 'account_id': account_id}
    else:
        params = {'cutoff_date': cutoff_date}
    
    df = execute_query(query, params)
    
    if df.empty:
        logger.warning("No account data found")
        return df
    
    # Engineer derived features
    df['negative_ratio'] = df['negative_feedback'] / (df['total_feedback'] + 1)
    df['positive_ratio'] = df['positive_feedback'] / (df['total_feedback'] + 1)
    df['critical_ratio'] = df['critical_issues'] / (df['total_feedback'] + 1)
    df['days_since_last_feedback'] = (pd.Timestamp.now(tz='UTC') - pd.to_datetime(df['last_feedback_date'], utc=True)).dt.days
    df['feedback_velocity'] = df['feedback_last_7d'] / (df['feedback_last_30d'] + 1)
    
    # Billing issues are often a churn signal
    df['billing_issue_flag'] = (df['billing_issues'] > 0).astype(int)
    
    # Encode categorical features
    df['region_encoded'] = pd.Categorical(df['region']).codes
    
    # Create a simple tier encoding (free=0, pro=1, enterprise=2)
    tier_map = {'free': 0, 'pro': 1, 'enterprise': 2}
    df['tier_encoded'] = df['account_id'].map(tier_map).fillna(0)
    
    logger.info(f"Built features for {len(df)} accounts")
    return df


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = 'churned'
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Define feature columns for the model
    feature_cols = [
        'total_feedback',
        'feedback_last_30d',
        'feedback_last_7d',
        'negative_feedback',
        'positive_feedback',
        'critical_issues',
        'p5_issues',
        'avg_priority',
        'product_areas_mentioned',
        'billing_issues',
        'performance_issues',
        'security_issues',
        'negative_ratio',
        'positive_ratio',
        'critical_ratio',
        'days_since_last_feedback',
        'feedback_velocity',
        'billing_issue_flag',
        'region_encoded',
        'tier_encoded'
    ]
    
    # Ensure all feature columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    
    # Fill missing values
    X = X.fillna(0)
    
    # Extract target if present
    if target_column in df.columns:
        y = df[target_column]
    else:
        logger.warning(f"Target column '{target_column}' not found, returning None")
        y = None
    
    return X, y


def create_synthetic_labels(df: pd.DataFrame) -> pd.Series:
    """Create synthetic churn labels for demonstration purposes.
    
    In production, you would have actual churn labels from your CRM/billing system.
    This creates synthetic labels based on negative feedback patterns.
    
    Args:
        df: DataFrame with account features
        
    Returns:
        Series with binary churn labels (1 = churned, 0 = retained)
    """
    logger.info("Creating synthetic churn labels for training")
    
    # Simple heuristic: high churn probability if:
    # - High negative ratio
    # - Multiple critical issues
    # - Billing issues present
    
    churn_score = (
        df['negative_ratio'] * 0.4 +
        (df['critical_ratio'] * 0.3) +
        (df['billing_issue_flag'] * 0.2) +
        (df['days_since_last_feedback'] / 90 * 0.1)
    )
    
    # Convert to binary with some randomness
    np.random.seed(42)
    churn_prob = 1 / (1 + np.exp(-5 * (churn_score - 0.5)))  # Sigmoid
    churned = (churn_prob + np.random.normal(0, 0.1, len(df)) > 0.5).astype(int)
    
    logger.info(f"Created labels: {churned.sum()} churned, {len(churned) - churned.sum()} retained")
    
    return pd.Series(churned, index=df.index, name='churned')
