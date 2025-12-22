"""
Model Registry - Central versioning and management system for all trained models.

Manages:
- Model version creation and metadata
- Training run tracking
- Performance metrics logging
- Model comparison and promotion logic
- Active model deployment state
- Data freshness monitoring
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from sqlalchemy import text, select, desc
from sqlalchemy.orm import Session

from src.db import get_session

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a single model version with its metadata."""
    model_type: str
    version: str
    status: str = "inactive"
    created_by: Optional[str] = None
    description: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class TrainingRun:
    """Represents a single training execution."""
    model_version_id: int
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    training_samples: Optional[int] = None
    test_samples: Optional[int] = None
    data_freshness_check: Optional[Dict] = None
    duration_seconds: Optional[int] = None
    triggered_by: str = "manual"
    id: Optional[int] = None


class ModelRegistry:
    """
    Central registry for managing model versions, training runs, and metrics.
    
    Usage:
        registry = ModelRegistry()
        
        # Create a new version
        new_version = registry.create_version(
            model_type='churn',
            version='v2',
            created_by='scheduler'
        )
        
        # Log training run
        run = registry.start_training_run(
            model_version_id=new_version.id,
            triggered_by='scheduler'
        )
        
        # Log metrics
        registry.log_metric(new_version.id, run.id, 'accuracy', 0.92, 'test')
        
        # Compare with previous version
        comparison = registry.compare_versions('churn', previous_id, new_version.id)
        
        # Promote if better
        if comparison['promotion_recommended']:
            registry.deploy_version('churn', new_version.id)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_version(
        self,
        model_type: str,
        version: str,
        created_by: str = "system",
        description: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ) -> ModelVersion:
        """
        Create a new model version entry.
        
        Args:
            model_type: 'churn' or 'product_area'
            version: Version string (e.g., 'v1', 'v2', 'v1.2.3')
            created_by: User or system identifier
            description: Optional description of the version
            hyperparameters: Dict of training hyperparameters used
            model_path: Relative path to saved model
            scaler_path: Relative path to saved scaler/config
            
        Returns:
            ModelVersion object with ID set
        """
        session = next(get_session())
        try:
            query = text("""
                INSERT INTO model_versions 
                (model_type, version, status, created_by, description, hyperparameters, 
                 model_path, scaler_path, created_at)
                VALUES (:model_type, :version, :status, :created_by, :description, :hyperparameters,
                        :model_path, :scaler_path, NOW())
                RETURNING id, created_at
            """)
            
            result = session.execute(query, {
                "model_type": model_type,
                "version": version,
                "status": "inactive",
                "created_by": created_by,
                "description": description,
                "hyperparameters": json.dumps(hyperparameters) if hyperparameters else None,
                "model_path": model_path,
                "scaler_path": scaler_path
            }).fetchone()
            
            session.commit()
            
            version_obj = ModelVersion(
                id=result[0],
                model_type=model_type,
                version=version,
                status="inactive",
                created_by=created_by,
                description=description,
                hyperparameters=hyperparameters,
                model_path=model_path,
                scaler_path=scaler_path,
                created_at=result[1]
            )
            
            self.logger.info(f"Created version {model_type}:{version} (ID: {result[0]})")
            return version_obj
            
        finally:
            session.close()
    
    def start_training_run(
        self,
        model_version_id: int,
        triggered_by: str = "manual",
        data_freshness_check: Optional[Dict] = None
    ) -> TrainingRun:
        """
        Start a training run for a model version.
        
        Args:
            model_version_id: ID of the model version being trained
            triggered_by: 'scheduler', 'manual', or 'api'
            data_freshness_check: Optional dict with freshness analysis
            
        Returns:
            TrainingRun object with ID and timestamps set
        """
        session = next(get_session())
        try:
            query = text("""
                INSERT INTO training_runs 
                (model_version_id, started_at, status, triggered_by, data_freshness_check)
                VALUES (:model_version_id, NOW(), :status, :triggered_by, :data_freshness_check)
                RETURNING id, started_at
            """)
            
            result = session.execute(query, {
                "model_version_id": model_version_id,
                "status": "running",
                "triggered_by": triggered_by,
                "data_freshness_check": json.dumps(data_freshness_check) if data_freshness_check else None
            }).fetchone()
            
            session.commit()
            
            run = TrainingRun(
                id=result[0],
                model_version_id=model_version_id,
                status="running",
                started_at=result[1],
                triggered_by=triggered_by,
                data_freshness_check=data_freshness_check
            )
            
            self.logger.info(f"Started training run {run.id} for version_id {model_version_id}")
            return run
            
        finally:
            session.close()
    
    def complete_training_run(
        self,
        run_id: int,
        status: str,
        training_samples: int,
        test_samples: int,
        error_message: Optional[str] = None
    ):
        """
        Complete a training run with final status and stats.
        
        Args:
            run_id: Training run ID to complete
            status: 'completed' or 'failed'
            training_samples: Number of training samples used
            test_samples: Number of test samples used
            error_message: Optional error details if failed
        """
        session = next(get_session())
        try:
            query = text("""
                UPDATE training_runs 
                SET completed_at = NOW(),
                    status = :status,
                    training_samples = :training_samples,
                    test_samples = :test_samples,
                    duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER,
                    error_message = :error_message
                WHERE id = :run_id
            """)
            
            session.execute(query, {
                "run_id": run_id,
                "status": status,
                "training_samples": training_samples,
                "test_samples": test_samples,
                "error_message": error_message
            })
            
            session.commit()
            self.logger.info(f"Completed training run {run_id} with status {status}")
            
        finally:
            session.close()
    
    def log_metric(
        self,
        model_version_id: int,
        metric_name: str,
        metric_value: float,
        dataset_type: str = "test",
        training_run_id: Optional[int] = None
    ):
        """
        Log a performance metric for a model version.
        
        Args:
            model_version_id: Version to log metric for
            metric_name: e.g., 'accuracy', 'precision', 'recall', 'auc', 'f1'
            metric_value: The numeric metric value
            dataset_type: 'train', 'validation', or 'test'
            training_run_id: Optional reference to the training run
        """
        session = next(get_session())
        try:
            query = text("""
                INSERT INTO model_metrics 
                (model_version_id, training_run_id, metric_name, metric_value, dataset_type, created_at)
                VALUES (:model_version_id, :training_run_id, :metric_name, :metric_value, :dataset_type, NOW())
            """)
            
            session.execute(query, {
                "model_version_id": model_version_id,
                "training_run_id": training_run_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "dataset_type": dataset_type
            })
            
            session.commit()
            
        finally:
            session.close()
    
    def get_best_metrics(
        self,
        model_version_id: int,
        dataset_type: str = "test"
    ) -> Dict[str, float]:
        """
        Get all metrics for a model version.
        
        Args:
            model_version_id: Version to get metrics for
            dataset_type: Filter by dataset type
            
        Returns:
            Dict mapping metric_name to metric_value
        """
        session = next(get_session())
        try:
            query = text("""
                SELECT metric_name, metric_value
                FROM model_metrics
                WHERE model_version_id = :model_version_id
                  AND dataset_type = :dataset_type
                ORDER BY created_at DESC
            """)
            
            results = session.execute(query, {
                "model_version_id": model_version_id,
                "dataset_type": dataset_type
            }).fetchall()
            
            # Return most recent value per metric
            metrics = {}
            for metric_name, metric_value in results:
                if metric_name not in metrics:
                    metrics[metric_name] = metric_value
            
            return metrics
            
        finally:
            session.close()
    
    def get_version(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """Retrieve a specific model version."""
        session = next(get_session())
        try:
            query = text("""
                SELECT id, model_type, version, status, created_by, description, 
                       hyperparameters, model_path, scaler_path, created_at
                FROM model_versions
                WHERE model_type = :model_type AND version = :version
            """)
            
            result = session.execute(query, {
                "model_type": model_type,
                "version": version
            }).fetchone()
            
            if not result:
                return None
            
            return ModelVersion(
                id=result[0],
                model_type=result[1],
                version=result[2],
                status=result[3],
                created_by=result[4],
                description=result[5],
                hyperparameters=json.loads(result[6]) if result[6] else None,
                model_path=result[7],
                scaler_path=result[8],
                created_at=result[9]
            )
            
        finally:
            session.close()
    
    def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the currently active/deployed version of a model."""
        session = next(get_session())
        try:
            query = text("""
                SELECT mv.id, mv.model_type, mv.version, mv.status, mv.created_by, 
                       mv.description, mv.hyperparameters, mv.model_path, mv.scaler_path, mv.created_at
                FROM model_versions mv
                JOIN model_deployments md ON mv.id = md.active_version_id
                WHERE mv.model_type = :model_type AND md.model_type = :model_type
            """)
            
            result = session.execute(query, {
                "model_type": model_type
            }).fetchone()
            
            if not result:
                return None
            
            return ModelVersion(
                id=result[0],
                model_type=result[1],
                version=result[2],
                status=result[3],
                created_by=result[4],
                description=result[5],
                hyperparameters=json.loads(result[6]) if result[6] else None,
                model_path=result[7],
                scaler_path=result[8],
                created_at=result[9]
            )
            
        finally:
            session.close()
    
    def list_versions(self, model_type: str, limit: int = 10) -> List[ModelVersion]:
        """List recent versions of a model type."""
        session = next(get_session())
        try:
            query = text("""
                SELECT id, model_type, version, status, created_by, description, 
                       hyperparameters, model_path, scaler_path, created_at
                FROM model_versions
                WHERE model_type = :model_type
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            
            results = session.execute(query, {
                "model_type": model_type,
                "limit": limit
            }).fetchall()
            
            versions = []
            for result in results:
                versions.append(ModelVersion(
                    id=result[0],
                    model_type=result[1],
                    version=result[2],
                    status=result[3],
                    created_by=result[4],
                    description=result[5],
                    hyperparameters=json.loads(result[6]) if result[6] else None,
                    model_path=result[7],
                    scaler_path=result[8],
                    created_at=result[9]
                ))
            
            return versions
            
        finally:
            session.close()
    
    def compare_versions(
        self,
        model_type: str,
        previous_version_id: int,
        new_version_id: int,
        promotion_threshold: float = 0.01
    ) -> Dict:
        """
        Compare two model versions and determine if new should be promoted.
        
        Args:
            model_type: Type of model
            previous_version_id: ID of current/previous version
            new_version_id: ID of new version to compare
            promotion_threshold: Minimum improvement (e.g., 0.01 for 1% improvement)
            
        Returns:
            Dict with comparison results and promotion recommendation
        """
        session = next(get_session())
        try:
            # Get metrics for both versions
            prev_metrics = self.get_best_metrics(previous_version_id, dataset_type="test")
            new_metrics = self.get_best_metrics(new_version_id, dataset_type="test")
            
            # Calculate improvements
            improvements = {}
            promotion_recommended = True
            
            for metric_name in set(list(prev_metrics.keys()) + list(new_metrics.keys())):
                prev_val = prev_metrics.get(metric_name, 0)
                new_val = new_metrics.get(metric_name, 0)
                
                if prev_val == 0:
                    improvement = 0
                else:
                    improvement = (new_val - prev_val) / prev_val
                
                improvements[metric_name] = float(improvement)
                
                # Lower values are better for loss/error metrics
                if metric_name in ['loss', 'error', 'rmse']:
                    if improvement > -promotion_threshold:
                        promotion_recommended = False
                # Higher values are better for accuracy metrics
                else:
                    if improvement < promotion_threshold:
                        promotion_recommended = False
            
            # Store comparison in DB
            query = text("""
                INSERT INTO model_comparisons 
                (model_type, previous_version_id, new_version_id, comparison_date, 
                 metric_improvements, promotion_recommended, promotion_threshold)
                VALUES (:model_type, :prev_id, :new_id, NOW(), :improvements, 
                        :promotion_recommended, :threshold)
            """)
            
            session.execute(query, {
                "model_type": model_type,
                "prev_id": previous_version_id,
                "new_id": new_version_id,
                "improvements": json.dumps(improvements),
                "promotion_recommended": promotion_recommended,
                "threshold": promotion_threshold
            })
            session.commit()
            
            self.logger.info(
                f"Version comparison: {model_type} v{new_version_id} vs "
                f"v{previous_version_id} - Promotion: {promotion_recommended}"
            )
            
            return {
                "metric_improvements": improvements,
                "promotion_recommended": promotion_recommended,
                "promotion_threshold": promotion_threshold
            }
            
        finally:
            session.close()
    
    def deploy_version(
        self,
        model_type: str,
        new_version_id: int,
        deployed_by: str = "system",
        notes: Optional[str] = None
    ):
        """
        Deploy a new version as the active model.
        
        Args:
            model_type: Type of model
            new_version_id: ID of version to deploy
            deployed_by: User or system identifier
            notes: Optional deployment notes
        """
        session = next(get_session())
        try:
            # Get current active version if exists
            current = session.execute(
                text("""
                    SELECT active_version_id FROM model_deployments 
                    WHERE model_type = :model_type
                """),
                {"model_type": model_type}
            ).fetchone()
            
            current_id = current[0] if current else None
            
            if current_id:
                # Update existing deployment
                query = text("""
                    UPDATE model_deployments
                    SET active_version_id = :new_version_id,
                        previous_version_id = :current_id,
                        deployed_at = NOW(),
                        deployed_by = :deployed_by,
                        rollback_available = TRUE,
                        notes = :notes
                    WHERE model_type = :model_type
                """)
            else:
                # Create new deployment record
                query = text("""
                    INSERT INTO model_deployments 
                    (model_type, active_version_id, deployed_at, deployed_by, notes)
                    VALUES (:model_type, :new_version_id, NOW(), :deployed_by, :notes)
                """)
            
            session.execute(query, {
                "model_type": model_type,
                "new_version_id": new_version_id,
                "current_id": current_id,
                "deployed_by": deployed_by,
                "notes": notes
            })
            
            # Update version status to active
            session.execute(
                text("""
                    UPDATE model_versions 
                    SET status = :status 
                    WHERE id = :version_id
                """),
                {"status": "active", "version_id": new_version_id}
            )
            
            session.commit()
            self.logger.info(f"Deployed {model_type} version {new_version_id}")
            
        finally:
            session.close()
    
    def log_data_freshness(
        self,
        model_type: str,
        last_training_date: Optional[datetime],
        latest_feedback_date: datetime,
        new_feedback_count: int,
        percent_new_data: float,
        decision_to_retrain: bool,
        reason: str
    ):
        """
        Log data freshness analysis for training decision.
        
        Args:
            model_type: Type of model being checked
            last_training_date: When the model was last trained
            latest_feedback_date: Most recent feedback in database
            new_feedback_count: Number of new feedback records since last training
            percent_new_data: Percentage of new data vs. previously seen
            decision_to_retrain: Whether to proceed with retraining
            reason: Human-readable reason for the decision
        """
        session = next(get_session())
        try:
            query = text("""
                INSERT INTO data_freshness_log
                (model_type, check_timestamp, last_training_date, latest_feedback_date,
                 new_feedback_count, percent_new_data, decision_to_retrain, reason)
                VALUES (:model_type, NOW(), :last_training_date, :latest_feedback_date,
                        :new_feedback_count, :percent_new_data, :decision_to_retrain, :reason)
            """)
            
            session.execute(query, {
                "model_type": model_type,
                "last_training_date": last_training_date,
                "latest_feedback_date": latest_feedback_date,
                "new_feedback_count": new_feedback_count,
                "percent_new_data": percent_new_data,
                "decision_to_retrain": decision_to_retrain,
                "reason": reason
            })
            
            session.commit()
            
        finally:
            session.close()
    
    def log_prediction_distribution(
        self,
        model_type: str,
        model_version_id: int,
        predictions: List[float]
    ):
        """
        Log prediction distribution metrics for health checks.
        
        Args:
            model_type: Type of model
            model_version_id: Version being scored
            predictions: List of prediction values
        """
        if not predictions:
            return
        
        import numpy as np
        
        session = next(get_session())
        try:
            metrics = {
                "mean_prediction": float(np.mean(predictions)),
                "std_prediction": float(np.std(predictions)),
                "min_prediction": float(np.min(predictions)),
                "max_prediction": float(np.max(predictions)),
                "median_prediction": float(np.median(predictions))
            }
            
            for metric_name, metric_value in metrics.items():
                query = text("""
                    INSERT INTO prediction_distribution_history
                    (model_type, model_version_id, scoring_date, metric_name, metric_value, samples_count)
                    VALUES (:model_type, :model_version_id, NOW(), :metric_name, :metric_value, :samples_count)
                """)
                
                session.execute(query, {
                    "model_type": model_type,
                    "model_version_id": model_version_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "samples_count": len(predictions)
                })
            
            session.commit()
            
        finally:
            session.close()
