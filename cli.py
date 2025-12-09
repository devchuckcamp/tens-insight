#!/usr/bin/env python3
"""Command-line interface for Tens-Insight.

Provides a unified interface for all ML pipeline operations.

Usage:
    python cli.py setup
    python cli.py train [churn|product-area|all]
    python cli.py score [accounts|product-areas|all]
    python cli.py status
"""

import sys
import argparse
import logging

from src.config import get_config
from src.db import execute_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_setup():
    """Run initial setup."""
    import setup
    setup.main()


def cmd_train(args):
    """Train models."""
    if args.model in ['churn', 'all']:
        logger.info("Training churn model...")
        from src.training.train_churn import train_churn_model
        train_churn_model(
            lookback_days=args.lookback_days,
            model_version=args.version
        )
    
    if args.model in ['product-area', 'all']:
        logger.info("Training product area model...")
        from src.training.train_product_area import train_product_area_model
        train_product_area_model(
            lookback_days=args.lookback_days,
            model_version=args.version
        )


def cmd_score(args):
    """Run scoring."""
    if args.target in ['accounts', 'all']:
        logger.info("Scoring accounts...")
        from src.scoring.score_accounts import score_accounts
        score_accounts(
            lookback_days=args.lookback_days,
            model_version=args.version
        )
    
    if args.target in ['product-areas', 'all']:
        logger.info("Scoring product areas...")
        from src.scoring.score_product_areas import score_product_areas
        score_product_areas(
            lookback_days=args.lookback_days,
            model_version=args.version
        )


def cmd_status():
    """Display pipeline status."""
    logger.info("=" * 60)
    logger.info("Tens-Insight Status")
    logger.info("=" * 60)
    
    # Check database
    try:
        df = execute_query("SELECT COUNT(*) as count FROM feedback_enriched")
        feedback_count = df['count'].iloc[0]
        logger.info(f"✓ Database connected")
        logger.info(f"  Feedback records: {feedback_count}")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return
    
    # Check prediction tables
    try:
        df = execute_query("SELECT COUNT(*) as count FROM account_risk_scores")
        account_scores = df['count'].iloc[0]
        logger.info(f"✓ Account risk scores: {account_scores}")
    except Exception:
        logger.warning("✗ Account risk scores table not found")
    
    try:
        df = execute_query("SELECT COUNT(*) as count FROM product_area_impact")
        product_scores = df['count'].iloc[0]
        logger.info(f"✓ Product area scores: {product_scores}")
    except Exception:
        logger.warning("✗ Product area impact table not found")
    
    # Check for models
    import os
    config = get_config()
    if os.path.exists(config.models_dir):
        models = [f for f in os.listdir(config.models_dir) if f.endswith('.keras')]
        if models:
            logger.info(f"✓ Found {len(models)} saved models")
            for model in models:
                logger.info(f"  - {model}")
        else:
            logger.warning("✗ No trained models found")
    else:
        logger.warning("✗ Models directory not found")
    
    logger.info("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Tens-Insight ML Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup
  %(prog)s train churn --lookback-days 90
  %(prog)s score all --version v1
  %(prog)s status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    subparsers.add_parser('setup', help='Run initial setup')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument(
        'model',
        choices=['churn', 'product-area', 'all'],
        help='Which model to train'
    )
    train_parser.add_argument(
        '--lookback-days',
        type=int,
        default=90,
        help='Number of days to look back for features (default: 90)'
    )
    train_parser.add_argument(
        '--version',
        default='v1',
        help='Model version identifier (default: v1)'
    )
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Run scoring')
    score_parser.add_argument(
        'target',
        choices=['accounts', 'product-areas', 'all'],
        help='What to score'
    )
    score_parser.add_argument(
        '--lookback-days',
        type=int,
        default=90,
        help='Number of days to look back for features (default: 90)'
    )
    score_parser.add_argument(
        '--version',
        default='v1',
        help='Model version to use (default: v1)'
    )
    
    # Status command
    subparsers.add_parser('status', help='Show pipeline status')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        cmd_setup()
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'score':
        cmd_score(args)
    elif args.command == 'status':
        cmd_status()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
