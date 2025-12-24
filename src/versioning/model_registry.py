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

from sqlalchemy import text

from src.db import get_connection

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
        """Create a new model version and register it in the database."""
        try:
            with get_connection() as conn:
                # Insert version
                query = text("""
                    INSERT INTO model_versions 
                    (model_type, version, status, created_by, description, hyperparameters, model_path, scaler_path, created_at)
                    VALUES (:model_type, :version, 'inactive', :created_by, :description, :hyperparameters, :model_path, :scaler_path, NOW())
                    RETURNING id, created_at
                """)
                
                result = conn.execute(query, {
                    'model_type': model_type,
                    'version': version,
                    'created_by': created_by,
                    'description': description,
                    'hyperparameters': json.dumps(hyperparameters or {}),
                    'model_path': model_path,
                    'scaler_path': scaler_path
                })
                conn.commit()
                
                row = result.fetchone()
                return ModelVersion(
                    id=row[0],
                    model_type=model_type,
                    version=version,
                    created_by=created_by,
                    description=description,
                    hyperparameters=hyperparameters,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    created_at=row[1],
                    status='inactive'
                )
        except Exception as e:
            self.logger.error(f"Failed to create model version: {e}")
            raise
    
    def start_training_run(
        self,
        model_version_id: int,
        triggered_by: str = "manual"
    ) -> TrainingRun:
        """Start a training run for a model version."""
        try:
            with get_connection() as conn:
                query = text("""
                    INSERT INTO training_runs 
                    (model_version_id, status, started_at, triggered_by)
                    VALUES (:model_version_id, 'running', NOW(), :triggered_by)
                    RETURNING id, started_at
                """)
                
                result = conn.execute(query, {
                    'model_version_id': model_version_id,
                    'triggered_by': triggered_by
                })
                conn.commit()
                
                row = result.fetchone()
                return TrainingRun(
                    id=row[0],
                    model_version_id=model_version_id,
                    status='running',
                    started_at=row[1],
                    triggered_by=triggered_by
                )
        except Exception as e:
            self.logger.error(f"Failed to start training run: {e}")
            raise
    
    def complete_training_run(
        self,
        training_run_id: int,
        status: str = "completed",
        training_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        duration_seconds: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Mark a training run as completed."""
        try:
            with get_connection() as conn:
                query = text("""
                    UPDATE training_runs 
                    SET status = :status,
                        completed_at = NOW(),
                        training_samples = :training_samples,
                        test_samples = :test_samples,
                        duration_seconds = :duration_seconds,
                        error_message = :error_message
                    WHERE id = :training_run_id
                """)
                
                conn.execute(query, {
                    'training_run_id': training_run_id,
                    'status': status,
                    'training_samples': training_samples,
                    'test_samples': test_samples,
                    'duration_seconds': duration_seconds,
                    'error_message': error_message
                })
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to complete training run: {e}")
            raise
    
    def log_metric(
        self,
        model_version_id: int,
        training_run_id: Optional[int],
        metric_name: str,
        metric_value: float,
        dataset_type: str = "test"
    ) -> bool:
        """Log a performance metric for a model version."""
        try:
            with get_connection() as conn:
                query = text("""
                    INSERT INTO model_metrics 
                    (model_version_id, training_run_id, metric_name, metric_value, dataset_type, created_at)
                    VALUES (:model_version_id, :training_run_id, :metric_name, :metric_value, :dataset_type, NOW())
                """)
                
                conn.execute(query, {
                    'model_version_id': model_version_id,
                    'training_run_id': training_run_id,
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'dataset_type': dataset_type
                })
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to log metric: {e}")
            raise
    
    def get_version_metrics(
        self,
        model_version_id: int,
        dataset_type: str = "test"
    ) -> Dict[str, float]:
        """Get all metrics for a specific model version."""
        try:
            with get_connection() as conn:
                query = text("""
                    SELECT metric_name, metric_value
                    FROM model_metrics
                    WHERE model_version_id = :model_version_id
                      AND dataset_type = :dataset_type
                    ORDER BY logged_at DESC
                """)
                
                result = conn.execute(query, {
                    'model_version_id': model_version_id,
                    'dataset_type': dataset_type
                })
                
                # Get latest value for each metric
                metrics = {}
                for row in result:
                    metric_name, metric_value = row
                    if metric_name not in metrics:  # Keep only the latest
                        metrics[metric_name] = metric_value
                
                return metrics
        except Exception as e:
            self.logger.error(f"Failed to get version metrics: {e}")
            return {}
    
    def get_latest_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the most recently created version for a model type."""
        try:
            with get_connection() as conn:
                query = text("""
                    SELECT id, model_type, version, status, created_by, description, 
                           hyperparameters, model_path, scaler_path, created_at
                    FROM model_versions
                    WHERE model_type = :model_type
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = conn.execute(query, {'model_type': model_type})
                row = result.fetchone()
                
                if row:
                    return ModelVersion(
                        id=row[0],
                        model_type=row[1],
                        version=row[2],
                        status=row[3],
                        created_by=row[4],
                        description=row[5],
                        hyperparameters=row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
                        model_path=row[7],
                        scaler_path=row[8],
                        created_at=row[9]
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get latest version: {e}")
            return None
    
    def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the currently deployed (active) version for a model type."""
        try:
            with get_connection() as conn:
                # Check model_deployments table
                query = text("""
                    SELECT mv.id, mv.model_type, mv.version, mv.status, mv.created_by,
                           mv.description, mv.hyperparameters, mv.model_path, mv.scaler_path, mv.created_at
                    FROM model_versions mv
                    JOIN model_deployments md ON mv.id = md.active_version_id
                    WHERE md.model_type = :model_type
                """)
                
                result = conn.execute(query, {'model_type': model_type})
                row = result.fetchone()
                
                if row:
                    return ModelVersion(
                        id=row[0],
                        model_type=row[1],
                        version=row[2],
                        status='active',
                        created_by=row[4],
                        description=row[5],
                        hyperparameters=row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
                        model_path=row[7],
                        scaler_path=row[8],
                        created_at=row[9]
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get active version: {e}")
            return None
    
    def compare_versions(
        self,
        model_type: str,
        previous_version_id: int,
        new_version_id: int,
        improvement_threshold: float = 0.01
    ) -> Dict:
        """Compare two model versions and determine if new version should be promoted."""
        try:
            with get_connection() as conn:
                # Get metrics for both versions
                prev_metrics = self.get_version_metrics(previous_version_id)
                new_metrics = self.get_version_metrics(new_version_id)
                
                # Calculate improvements
                improvements = {}
                metric_keys = set(prev_metrics.keys()) | set(new_metrics.keys())
                
                for metric in metric_keys:
                    prev_val = prev_metrics.get(metric, 0)
                    new_val = new_metrics.get(metric, 0)
                    
                    if prev_val > 0:
                        improvement = (new_val - prev_val) / prev_val
                    else:
                        improvement = new_val
                    
                    improvements[metric] = improvement
                
                # Determine promotion recommendation
                avg_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
                promote = avg_improvement > improvement_threshold
                
                # Log comparison to database
                query = text("""
                    INSERT INTO model_comparisons 
                    (model_type, previous_version_id, new_version_id, improvements_json, promotion_recommended, compared_at)
                    VALUES (:model_type, :previous_version_id, :new_version_id, :improvements, :promotion_recommended, NOW())
                """)
                
                conn.execute(query, {
                    'model_type': model_type,
                    'previous_version_id': previous_version_id,
                    'new_version_id': new_version_id,
                    'improvements': json.dumps(improvements),
                    'promotion_recommended': promote
                })
                conn.commit()
                
                return {
                    'previous_version_id': previous_version_id,
                    'new_version_id': new_version_id,
                    'improvements': improvements,
                    'average_improvement': avg_improvement,
                    'promotion_recommended': promote
                }
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {e}")
            return {'error': str(e), 'promotion_recommended': False}
    
    def deploy_version(
        self,
        model_type: str,
        version_id: int
    ) -> bool:
        """Deploy a model version to production (mark as active)."""
        try:
            with get_connection() as conn:
                # Get current active version
                query = text("""
                    SELECT active_version_id FROM model_deployments 
                    WHERE model_type = :model_type
                """)
                result = conn.execute(query, {'model_type': model_type})
                row = result.fetchone()
                previous_version_id = row[0] if row else None
                
                # Update or insert deployment record
                if previous_version_id:
                    query = text("""
                        UPDATE model_deployments
                        SET active_version_id = :version_id,
                            previous_version_id = :previous_version_id,
                            deployed_at = NOW()
                        WHERE model_type = :model_type
                    """)
                else:
                    query = text("""
                        INSERT INTO model_deployments 
                        (model_type, active_version_id, previous_version_id, rollback_available, deployed_at)
                        VALUES (:model_type, :version_id, NULL, FALSE, NOW())
                    """)
                
                conn.execute(query, {
                    'model_type': model_type,
                    'version_id': version_id,
                    'previous_version_id': previous_version_id
                })
                
                # Update version status
                query = text("""
                    UPDATE model_versions 
                    SET status = 'active'
                    WHERE id = :version_id
                """)
                conn.execute(query, {'version_id': version_id})
                
                conn.commit()
                self.logger.info(f"Deployed version {version_id} for {model_type}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to deploy version: {e}")
            raise
    
    def log_data_freshness_decision(
        self,
        model_type: str,
        new_feedback_count: int,
        decision_to_retrain: bool,
        reason: str
    ) -> bool:
        """Log a data freshness check and retraining decision."""
        try:
            with get_connection() as conn:
                query = text("""
                    INSERT INTO data_freshness_log 
                    (model_type, check_timestamp, new_feedback_count, decision_to_retrain, reason)
                    VALUES (:model_type, NOW(), :new_feedback_count, :decision_to_retrain, :reason)
                """)
                
                conn.execute(query, {
                    'model_type': model_type,
                    'new_feedback_count': new_feedback_count,
                    'decision_to_retrain': decision_to_retrain,
                    'reason': reason
                })
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to log data freshness decision: {e}")
            raise
    
    def log_prediction_distribution(
        self,
        model_type: str,
        model_version_id: int,
        metric_name: str,
        metric_value: float,
        samples_count: int
    ) -> bool:
        """Log prediction distribution metrics for drift detection."""
        try:
            with get_connection() as conn:
                query = text("""
                    INSERT INTO prediction_distribution_history 
                    (model_type, model_version_id, metric_name, metric_value, samples_count, scoring_date)
                    VALUES (:model_type, :model_version_id, :metric_name, :metric_value, :samples_count, NOW())
                """)
                
                conn.execute(query, {
                    'model_type': model_type,
                    'model_version_id': model_version_id,
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'samples_count': samples_count
                })
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to log prediction distribution: {e}")
            raise
    
    def get_version_history(self, model_type: str, limit: int = 10) -> List[ModelVersion]:
        """Get recent versions for a model type."""
        try:
            with get_connection() as conn:
                query = text("""
                    SELECT id, model_type, version, status, created_by, description,
                           hyperparameters, model_path, scaler_path, created_at
                    FROM model_versions
                    WHERE model_type = :model_type
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                
                result = conn.execute(query, {
                    'model_type': model_type,
                    'limit': limit
                })
                
                versions = []
                for row in result:
                    versions.append(ModelVersion(
                        id=row[0],
                        model_type=row[1],
                        version=row[2],
                        status=row[3],
                        created_by=row[4],
                        description=row[5],
                        hyperparameters=row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
                        model_path=row[7],
                        scaler_path=row[8],
                        created_at=row[9]
                    ))
                
                return versions
        except Exception as e:
            self.logger.error(f"Failed to get version history: {e}")
            return []
