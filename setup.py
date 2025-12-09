#!/usr/bin/env python3
"""Setup script to initialize the tens-insight project.

This script:
1. Creates necessary directories
2. Creates the prediction tables in the database
3. Validates the connection to Postgres

Usage:
    python setup.py
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_config
from src.db import create_prediction_tables, execute_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run setup tasks."""
    logger.info("=" * 60)
    logger.info("Tens-Insight Setup")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Step 1: Create directories
    logger.info("Creating necessary directories...")
    os.makedirs(config.models_dir, exist_ok=True)
    logger.info(f"  Created {config.models_dir}/")
    
    # Step 2: Test database connection
    logger.info("Testing database connection...")
    try:
        df = execute_query("SELECT 1 as test")
        logger.info("  ✓ Database connection successful")
    except Exception as e:
        logger.error(f"  ✗ Database connection failed: {e}")
        logger.error("  Please check your DATABASE_URL environment variable")
        sys.exit(1)
    
    # Step 3: Check if feedback_enriched table exists
    logger.info("Checking for feedback_enriched table...")
    try:
        df = execute_query("SELECT COUNT(*) as count FROM feedback_enriched")
        count = df['count'].iloc[0]
        logger.info(f"  ✓ Found feedback_enriched table with {count} rows")
    except Exception as e:
        logger.warning(f"  ✗ Could not find feedback_enriched table: {e}")
        logger.warning("  Make sure goinsight migrations have run first")
    
    # Step 4: Create prediction tables
    logger.info("Creating prediction tables...")
    try:
        create_prediction_tables()
        logger.info("  ✓ Created account_risk_scores table")
        logger.info("  ✓ Created product_area_impact table")
    except Exception as e:
        logger.error(f"  ✗ Failed to create prediction tables: {e}")
        sys.exit(1)
    
    # Step 5: Summary
    logger.info("=" * 60)
    logger.info("Setup Complete!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Train models:")
    logger.info("     python -m src.training.train_churn")
    logger.info("     python -m src.training.train_product_area")
    logger.info("  2. Run scoring:")
    logger.info("     python -m src.scoring.score_accounts")
    logger.info("     python -m src.scoring.score_product_areas")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
