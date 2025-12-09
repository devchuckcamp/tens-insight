"""Feature engineering for product area models.

Aggregates feedback by product area and customer segment to identify
high-impact areas that need attention.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ..db import execute_query

logger = logging.getLogger(__name__)


def build_product_area_features(
    lookback_days: int = 90,
    product_area: Optional[str] = None
) -> pd.DataFrame:
    """Build product-area level features for priority scoring.
    
    Aggregates feedback by (product_area, segment) to identify
    high-impact issues that should be prioritized.
    
    Args:
        lookback_days: Number of days to look back for features
        product_area: Optional specific product area to score
        
    Returns:
        DataFrame with one row per (product_area, segment) combination
    """
    logger.info(f"Building product area features with {lookback_days} day lookback")
    
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Query to aggregate feedback by product area and segment
    query = """
        SELECT 
            product_area,
            customer_tier as segment,
            COUNT(*) as feedback_count,
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as feedback_last_30d,
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as feedback_last_7d,
            COUNT(*) FILTER (WHERE sentiment = 'negative') as negative_count,
            COUNT(*) FILTER (WHERE sentiment = 'positive') as positive_count,
            COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral_count,
            COUNT(*) FILTER (WHERE priority >= 4) as critical_count,
            COUNT(*) FILTER (WHERE priority = 5) as p5_count,
            AVG(priority) as avg_priority,
            MAX(created_at) as last_feedback_date,
            COUNT(DISTINCT topic) as unique_topics,
            COUNT(DISTINCT region) as regions_affected
        FROM feedback_enriched
        WHERE created_at >= %(cutoff_date)s
    """
    
    if product_area:
        query += " AND product_area = %(product_area)s"
        params = {'cutoff_date': cutoff_date, 'product_area': product_area}
    else:
        params = {'cutoff_date': cutoff_date}
    
    query += " GROUP BY product_area, customer_tier"
    
    df = execute_query(query, params)
    
    if df.empty:
        logger.warning("No product area data found")
        return df
    
    # Engineer derived features
    df['negative_ratio'] = df['negative_count'] / (df['feedback_count'] + 1)
    df['critical_ratio'] = df['critical_count'] / (df['feedback_count'] + 1)
    df['p5_ratio'] = df['p5_count'] / (df['feedback_count'] + 1)
    
    # Sentiment score (-1 to 1)
    df['sentiment_score'] = (
        (df['positive_count'] - df['negative_count']) / (df['feedback_count'] + 1)
    )
    df['avg_sentiment_score'] = df['sentiment_score']  # Alias for output
    
    # Recent activity indicator
    df['recent_activity_ratio'] = df['feedback_last_7d'] / (df['feedback_last_30d'] + 1)
    
    # Encode segment
    segment_map = {'free': 0, 'pro': 1, 'enterprise': 2}
    df['segment_encoded'] = df['segment'].map(segment_map).fillna(0)
    
    # Enterprise customers get higher weight
    df['enterprise_flag'] = (df['segment'] == 'enterprise').astype(int)
    
    logger.info(f"Built features for {len(df)} product-area/segment combinations")
    return df


def calculate_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate priority scores for product areas.
    
    Priority score is a weighted combination of:
    - Feedback volume
    - Negative sentiment
    - Critical issues
    - Customer segment importance
    - Recent activity
    
    Args:
        df: DataFrame with product area features
        
    Returns:
        DataFrame with added priority_score column
    """
    logger.info("Calculating priority scores")
    
    # Normalize features to 0-1 range
    def normalize(series):
        """Min-max normalization."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    # Normalize key metrics
    norm_feedback = normalize(df['feedback_count'])
    norm_negative = normalize(df['negative_count'])
    norm_critical = normalize(df['critical_count'])
    norm_p5 = normalize(df['p5_count'])
    norm_recent = normalize(df['feedback_last_7d'])
    
    # Calculate weighted priority score (0-100 scale)
    priority_score = (
        norm_feedback * 15 +          # Volume of feedback
        norm_negative * 25 +           # Negative sentiment
        norm_critical * 20 +           # Critical issues (P4+)
        norm_p5 * 15 +                 # P5 issues
        norm_recent * 10 +             # Recent activity
        df['enterprise_flag'] * 15     # Enterprise customer impact
    )
    
    df['priority_score'] = priority_score
    
    # Add risk category
    df['risk_category'] = pd.cut(
        df['priority_score'],
        bins=[0, 30, 50, 70, 100],
        labels=['low', 'medium', 'high', 'critical']
    )
    
    logger.info(f"Priority scores calculated: min={priority_score.min():.2f}, max={priority_score.max():.2f}")
    
    return df


def prepare_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare product area features for scoring/prediction.
    
    Args:
        df: DataFrame with raw features
        
    Returns:
        DataFrame ready for model prediction
    """
    # Define feature columns
    feature_cols = [
        'feedback_count',
        'feedback_last_30d',
        'feedback_last_7d',
        'negative_count',
        'positive_count',
        'critical_count',
        'p5_count',
        'avg_priority',
        'unique_topics',
        'regions_affected',
        'negative_ratio',
        'critical_ratio',
        'p5_ratio',
        'sentiment_score',
        'recent_activity_ratio',
        'segment_encoded',
        'enterprise_flag'
    ]
    
    # Ensure all feature columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    
    # Fill missing values
    X = X.fillna(0)
    
    return X
