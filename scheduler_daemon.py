#!/usr/bin/env python3
"""
Entry point for running the Tens-Insight continuous training scheduler.

This is typically run as a long-lived background process in Docker or 
via a system scheduler (cron, systemd, etc.).

Usage:
    python scheduler_daemon.py
    python cli.py scheduler start
"""

import logging
import sys
import signal

from src.scheduler import TrainingScheduler
from src.training.train_churn import train_churn_model
from src.training.train_product_area import train_product_area_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global scheduler for signal handling
_scheduler = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    if _scheduler:
        _scheduler.stop()
    sys.exit(0)


def run_scheduler():
    """Run the continuous training scheduler."""
    global _scheduler
    
    logger.info("=" * 70)
    logger.info("Tens-Insight Continuous Training Scheduler")
    logger.info("=" * 70)
    
    _scheduler = TrainingScheduler(timezone='UTC')
    
    # Schedule churn model training daily at 2 AM UTC
    _scheduler.schedule_daily_training(
        model_type='churn',
        hour=2,
        minute=0,
        training_function=train_churn_model,
        training_kwargs={
            'incremental': True,
            'lookback_days': 90,
            'promotion_threshold': 0.01
        }
    )
    
    # Schedule product area model training weekly (Sunday 3 AM UTC)
    _scheduler.schedule_weekly_training(
        model_type='product_area',
        day_of_week='sunday',
        hour=3,
        minute=0,
        training_function=train_product_area_model,
        training_kwargs={
            'incremental': True,
            'lookback_days': 90
        }
    )
    
    # Start the scheduler
    _scheduler.start()
    
    # Log scheduled jobs
    logger.info("Scheduled Training Jobs:")
    logger.info("-" * 70)
    for job_id, job_info in _scheduler.get_scheduled_jobs().items():
        config = job_info['config']
        next_run = job_info['next_run']
        logger.info(f"  {job_id}")
        logger.info(f"    Type: {config.get('type')}")
        logger.info(f"    Model: {config.get('model_type')}")
        logger.info(f"    Next run: {next_run}")
    logger.info("-" * 70)
    
    logger.info("Scheduler started successfully")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 70)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep the scheduler running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        _scheduler.stop()


if __name__ == '__main__':
    run_scheduler()
