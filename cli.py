#!/usr/bin/env python3
"""Command-line interface for Tens-Insight.

Provides a unified interface for all ML pipeline operations including
scheduled continuous training.

Usage:
    python cli.py setup
    python cli.py train [churn|product-area|all]
    python cli.py score [accounts|product-areas|all]
    python cli.py scheduler start
    python cli.py scheduler stop
    python cli.py scheduler status
    python cli.py status
"""

import sys
import argparse
import logging
import time

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
            model_version=args.version,
            incremental=args.incremental
        )
    
    if args.model in ['product-area', 'all']:
        logger.info("Training product area model...")
        from src.training.train_product_area import train_product_area_model
        train_product_area_model(
            lookback_days=args.lookback_days,
            model_version=args.version,
            incremental=args.incremental
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


def cmd_scheduler(args):
    """Manage continuous training scheduler."""
    from src.scheduler import TrainingScheduler
    from src.training.train_churn import train_churn_model
    from src.training.train_product_area import train_product_area_model
    
    scheduler = TrainingScheduler(timezone='UTC')
    
    if args.scheduler_action == 'start':
        logger.info("=" * 60)
        logger.info("Starting Continuous Training Scheduler")
        logger.info("=" * 60)
        
        # Schedule daily churn training at 2 AM UTC
        scheduler.schedule_daily_training(
            model_type='churn',
            hour=2,
            minute=0,
            training_function=train_churn_model,
            training_kwargs={'incremental': True, 'lookback_days': 90}
        )
        
        # Schedule weekly product area training on Sunday at 3 AM UTC
        scheduler.schedule_weekly_training(
            model_type='product_area',
            day_of_week='sunday',
            hour=3,
            minute=0,
            training_function=train_product_area_model,
            training_kwargs={'incremental': True, 'lookback_days': 90}
        )
        
        scheduler.start()
        
        logger.info("Scheduled jobs:")
        for job_id, job_info in scheduler.get_scheduled_jobs().items():
            config = job_info['config']
            next_run = job_info['next_run']
            logger.info(f"  {job_id}")
            logger.info(f"    Type: {config.get('type')}")
            logger.info(f"    Next run: {next_run}")
        
        logger.info("=" * 60)
        logger.info("Scheduler is running. Press Ctrl+C to stop.")
        logger.info("=" * 60)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down scheduler...")
            scheduler.stop()
            logger.info("Scheduler stopped")
    
    elif args.scheduler_action == 'stop':
        logger.info("Stopping scheduler...")
        scheduler.stop()
        logger.info("Scheduler stopped")
    
    elif args.scheduler_action == 'status':
        logger.info("=" * 60)
        logger.info("Training Scheduler Status")
        logger.info("=" * 60)
        
        jobs = scheduler.get_scheduled_jobs()
        
        if jobs:
            logger.info(f"Running: Yes")
            logger.info(f"Total jobs: {len(jobs)}")
            logger.info("")
            logger.info("Scheduled jobs:")
            for job_id, job_info in jobs.items():
                config = job_info['config']
                next_run = job_info['next_run']
                logger.info(f"  {job_id}")
                logger.info(f"    Type: {config.get('type')}")
                logger.info(f"    Model: {config.get('model_type')}")
                logger.info(f"    Next run: {next_run}")
        else:
            logger.info("No scheduled jobs")
        
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
  %(prog)s train all --incremental
  %(prog)s score all --version v1
  %(prog)s scheduler start
  %(prog)s scheduler status
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
        default=None,
        help='Model version identifier (auto-generated if not provided)'
    )
    train_parser.add_argument(
        '--incremental',
        action='store_true',
        help='Check data freshness before training (default: False)'
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
    
    # Scheduler command
    scheduler_parser = subparsers.add_parser(
        'scheduler',
        help='Manage continuous training scheduler'
    )
    scheduler_parser.add_argument(
        'scheduler_action',
        choices=['start', 'stop', 'status'],
        help='Scheduler action'
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
    elif args.command == 'scheduler':
        cmd_scheduler(args)
    elif args.command == 'status':
        cmd_status()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
