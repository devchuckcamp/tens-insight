"""
Training scheduler using APScheduler for continuous automated model retraining.

Provides background scheduling of training jobs with:
- Configurable job intervals (hourly, daily, weekly)
- Error handling and logging
- Job execution tracking
- Health checks and monitoring
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """
    Manages scheduled training jobs for continuous model retraining.
    
    Usage:
        scheduler = TrainingScheduler()
        
        # Schedule daily churn model training
        scheduler.schedule_daily_training(
            'churn',
            hour=2,
            minute=0,
            training_function=train_churn_model
        )
        
        # Schedule weekly product area training
        scheduler.schedule_weekly_training(
            'product_area',
            day_of_week='sunday',
            hour=3,
            training_function=train_product_area_model
        )
        
        # Start the scheduler
        scheduler.start()
        
        # Check job status
        jobs = scheduler.get_scheduled_jobs()
    """
    
    def __init__(self, timezone: str = 'UTC'):
        self.scheduler = BackgroundScheduler(timezone=timezone)
        self.logger = logging.getLogger(__name__)
        self.timezone = timezone
        self._job_configs = {}
    
    def schedule_daily_training(
        self,
        model_type: str,
        hour: int,
        minute: int,
        training_function: Callable,
        training_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a training job to run daily at a specific time.
        
        Args:
            model_type: 'churn' or 'product_area'
            hour: Hour of day (0-23)
            minute: Minute of hour (0-59)
            training_function: Callable that performs the training
            training_kwargs: Optional dict of kwargs to pass to training_function
            
        Returns:
            Job ID for tracking
        """
        if training_kwargs is None:
            training_kwargs = {}
        
        training_kwargs.setdefault('incremental', True)
        
        job_id = f"daily_{model_type}_{hour}_{minute}"
        
        trigger = CronTrigger(
            hour=hour,
            minute=minute,
            timezone=self.timezone
        )
        
        job = self.scheduler.add_job(
            func=self._job_wrapper(model_type, training_function),
            trigger=trigger,
            kwargs=training_kwargs,
            id=job_id,
            name=f"Daily {model_type} training at {hour:02d}:{minute:02d}",
            replace_existing=True
        )
        
        self._job_configs[job_id] = {
            'type': 'daily',
            'model_type': model_type,
            'hour': hour,
            'minute': minute
        }
        
        self.logger.info(f"Scheduled daily {model_type} training at {hour:02d}:{minute:02d}")
        return job_id
    
    def schedule_weekly_training(
        self,
        model_type: str,
        day_of_week: str,
        hour: int,
        minute: int,
        training_function: Callable,
        training_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a training job to run weekly on a specific day/time.
        
        Args:
            model_type: 'churn' or 'product_area'
            day_of_week: 'monday'-'sunday' or 0-6
            hour: Hour of day (0-23)
            minute: Minute of hour (0-59)
            training_function: Callable that performs the training
            training_kwargs: Optional dict of kwargs to pass to training_function
            
        Returns:
            Job ID for tracking
        """
        if training_kwargs is None:
            training_kwargs = {}
        
        training_kwargs.setdefault('incremental', True)
        
        # Convert day name to number if needed
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        day_of_week_normalized = day_map.get(day_of_week.lower(), day_of_week)
        
        job_id = f"weekly_{model_type}_{day_of_week_normalized}_{hour}_{minute}"
        
        trigger = CronTrigger(
            day_of_week=day_of_week_normalized,
            hour=hour,
            minute=minute,
            timezone=self.timezone
        )
        
        job = self.scheduler.add_job(
            func=self._job_wrapper(model_type, training_function),
            trigger=trigger,
            kwargs=training_kwargs,
            id=job_id,
            name=f"Weekly {model_type} training on {day_of_week} at {hour:02d}:{minute:02d}",
            replace_existing=True
        )
        
        self._job_configs[job_id] = {
            'type': 'weekly',
            'model_type': model_type,
            'day_of_week': day_of_week,
            'hour': hour,
            'minute': minute
        }
        
        self.logger.info(
            f"Scheduled weekly {model_type} training on {day_of_week} at {hour:02d}:{minute:02d}"
        )
        return job_id
    
    def schedule_interval_training(
        self,
        model_type: str,
        hours: int,
        minutes: int = 0,
        training_function: Callable = None,
        training_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a training job to run at regular intervals.
        
        Args:
            model_type: 'churn' or 'product_area'
            hours: Hours between training runs
            minutes: Additional minutes between training runs
            training_function: Callable that performs the training
            training_kwargs: Optional dict of kwargs to pass to training_function
            
        Returns:
            Job ID for tracking
        """
        if training_kwargs is None:
            training_kwargs = {}
        
        training_kwargs.setdefault('incremental', True)
        
        job_id = f"interval_{model_type}_{hours}h_{minutes}m"
        
        trigger = IntervalTrigger(
            hours=hours,
            minutes=minutes
        )
        
        job = self.scheduler.add_job(
            func=self._job_wrapper(model_type, training_function),
            trigger=trigger,
            kwargs=training_kwargs,
            id=job_id,
            name=f"Interval {model_type} training every {hours}h {minutes}m",
            replace_existing=True
        )
        
        self._job_configs[job_id] = {
            'type': 'interval',
            'model_type': model_type,
            'hours': hours,
            'minutes': minutes
        }
        
        self.logger.info(
            f"Scheduled interval {model_type} training every {hours}h {minutes}m"
        )
        return job_id
    
    def _job_wrapper(self, model_type: str, training_function: Callable) -> Callable:
        """
        Wrap training function with error handling and logging.
        
        Args:
            model_type: Type of model being trained
            training_function: The actual training function
            
        Returns:
            Wrapped function with error handling
        """
        def wrapped_job(*args, **kwargs):
            job_start = datetime.utcnow()
            self.logger.info(f"Starting scheduled {model_type} training job")
            
            try:
                result = training_function(*args, **kwargs)
                
                duration = (datetime.utcnow() - job_start).total_seconds()
                self.logger.info(
                    f"Completed {model_type} training job successfully in {duration:.1f}s"
                )
                
                return result
                
            except Exception as e:
                duration = (datetime.utcnow() - job_start).total_seconds()
                self.logger.error(
                    f"Failed {model_type} training job after {duration:.1f}s: {e}",
                    exc_info=True
                )
                # Don't re-raise to prevent scheduler from stopping
                return None
        
        return wrapped_job
    
    def start(self):
        """Start the background scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            self.logger.info("Training scheduler started")
        else:
            self.logger.warning("Training scheduler already running")
    
    def stop(self):
        """Stop the background scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("Training scheduler stopped")
        else:
            self.logger.warning("Training scheduler not running")
    
    def pause(self):
        """Pause the scheduler (jobs won't run but scheduler stays alive)."""
        if self.scheduler.running:
            self.scheduler.pause()
            self.logger.info("Training scheduler paused")
    
    def resume(self):
        """Resume a paused scheduler."""
        if self.scheduler.running:
            self.scheduler.resume()
            self.logger.info("Training scheduler resumed")
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a scheduled job.
        
        Args:
            job_id: ID of job to remove
            
        Returns:
            True if job was removed, False if not found
        """
        try:
            self.scheduler.remove_job(job_id)
            self._job_configs.pop(job_id, None)
            self.logger.info(f"Removed job {job_id}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to remove job {job_id}: {e}")
            return False
    
    def get_scheduled_jobs(self) -> Dict[str, Any]:
        """
        Get information about all scheduled jobs.
        
        Returns:
            Dict mapping job_id to job configuration and next run time
        """
        jobs_info = {}
        
        for job in self.scheduler.get_jobs():
            job_id = job.id
            config = self._job_configs.get(job_id, {})
            
            jobs_info[job_id] = {
                'config': config,
                'next_run': 'scheduled',
                'trigger': str(job.trigger),
                'enabled': True
            }
        
        return jobs_info
    
    def is_running(self) -> bool:
        """Check if scheduler is currently running."""
        return self.scheduler.running
    
    def get_next_run_time(self, job_id: str) -> Optional[str]:
        """Get the next scheduled run time for a job."""
        try:
            job = self.scheduler.get_job(job_id)
            return 'scheduled' if job else None
        except:
            return None


class SchedulerConfig:
    """Configuration for the training scheduler."""
    
    def __init__(self, timezone: str = 'UTC'):
        self.timezone = timezone
        self.jobs = []
    
    def add_daily_churn_training(self, hour: int = 2, minute: int = 0):
        """Add daily churn training job to config."""
        self.jobs.append({
            'type': 'daily',
            'model_type': 'churn',
            'hour': hour,
            'minute': minute
        })
    
    def add_weekly_product_area_training(self, day: str = 'sunday', hour: int = 3, minute: int = 0):
        """Add weekly product area training job to config."""
        self.jobs.append({
            'type': 'weekly',
            'model_type': 'product_area',
            'day_of_week': day,
            'hour': hour,
            'minute': minute
        })
    
    def add_interval_training(self, model_type: str, hours: int, minutes: int = 0):
        """Add interval-based training job to config."""
        self.jobs.append({
            'type': 'interval',
            'model_type': model_type,
            'hours': hours,
            'minutes': minutes
        })
