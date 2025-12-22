"""
Monitoring and health checks for continuous training system.

Provides:
- Health status monitoring of training pipeline
- Alerting on training failures or performance degradation
- Prediction drift detection
- Model staleness warnings
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from src.db import get_connection
from src.versioning import ModelRegistry

logger = logging.getLogger(__name__)


class ModelHealthCheck:
    """
    Monitor and assess the health of trained models.
    
    Checks:
    - Model staleness (how long since last training)
    - Prediction distribution drift (significant changes in predictions)
    - Metric degradation (performance decline)
    - Data freshness issues
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.logger = logging.getLogger(__name__)
    
    def check_model_staleness(
        self,
        model_type: str,
        max_days_without_training: int = 7
    ) -> Tuple[bool, str]:
        """
        Check if a model is too old (hasn't been retrained recently).
        
        Args:
            model_type: 'churn' or 'product_area'
            max_days_without_training: Maximum allowed days without retraining
            
        Returns:
            Tuple of (is_stale: bool, reason: str)
        """
        active_version = self.registry.get_active_version(model_type)
        
        if not active_version:
            return False, "No active model found"
        
        if not active_version.created_at:
            return False, "Model creation date unknown"
        
        days_old = (datetime.utcnow() - active_version.created_at).days
        
        if days_old > max_days_without_training:
            reason = f"Model {model_type} is {days_old} days old (max: {max_days_without_training})"
            return True, reason
        
        return False, f"Model is {days_old} days old (acceptable)"
    
    def check_prediction_drift(
        self,
        model_type: str,
        drift_threshold: float = 0.15
    ) -> Tuple[bool, Dict]:
        """
        Detect significant changes in prediction distribution.
        
        Args:
            model_type: 'churn' or 'product_area'
            drift_threshold: Percentage change threshold (0.15 = 15%)
            
        Returns:
            Tuple of (drift_detected: bool, analysis: dict)
        """
        try:
            with get_connection() as conn:
                # Get recent prediction distributions
                query = text("""
                    SELECT mv.version, pdh.metric_name, pdh.metric_value, pdh.logged_at
                    FROM prediction_distribution_history pdh
                    JOIN model_versions mv ON pdh.model_version_id = mv.id
                    WHERE mv.model_type = :model_type
                    ORDER BY pdh.logged_at DESC
                    LIMIT 20
                """)
                
                result = conn.execute(query, {'model_type': model_type})
                rows = result.fetchall()
                
                if not rows:
                    return False, {"reason": "no_prediction_data"}
                
                # Group by metric
                metrics = {}
                for version, metric_name, metric_value, logged_at in rows:
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(float(metric_value))
                
                # Check for drift
                drift_detected = False
                drift_details = {}
                
                for metric_name, values in metrics.items():
                    if len(values) >= 2:
                        recent = values[0]
                        older = sum(values[1:]) / len(values[1:])
                        
                        if older != 0:
                            percent_change = abs((recent - older) / older)
                            if percent_change > drift_threshold:
                                drift_detected = True
                                drift_details[metric_name] = {
                                    'recent': recent,
                                    'average_older': older,
                                    'percent_change': percent_change
                                }
                
                return drift_detected, drift_details if drift_detected else {"reason": "no_significant_drift"}
        
        except Exception as e:
            self.logger.error(f"Error checking prediction drift: {e}")
            return False, {"error": str(e)}
    
    def check_metric_degradation(
        self,
        model_type: str,
        metric_name: str = "accuracy",
        degradation_threshold: float = 0.80
    ) -> Tuple[bool, Dict]:
        """
        Check if model metrics have degraded below acceptable levels.
        
        Args:
            model_type: Type of model
            metric_name: Metric to check (e.g., 'accuracy', 'auc')
            degradation_threshold: Minimum acceptable metric value
            
        Returns:
            Tuple of (degraded: bool, analysis: dict)
        """
        try:
            active_version = self.registry.get_active_version(model_type)
            
            if not active_version or not active_version.id:
                return False, {"reason": "no_active_version"}
            
            metrics = self.registry.get_version_metrics(active_version.id)
            
            if metric_name not in metrics:
                return False, {"reason": f"metric_{metric_name}_not_found"}
            
            metric_value = metrics[metric_name]
            degraded = metric_value < degradation_threshold
            
            return degraded, {
                'metric': metric_name,
                'value': metric_value,
                'threshold': degradation_threshold,
                'degraded': degraded
            }
        
        except Exception as e:
            self.logger.error(f"Error checking metric degradation: {e}")
            return False, {"error": str(e)}
    
    def get_full_health_report(self, model_type: str) -> Dict:
        """
        Get comprehensive health status for a model.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing all health check results
        """
        stale, stale_msg = self.check_model_staleness(model_type)
        drift, drift_details = self.check_prediction_drift(model_type)
        degraded, degradation_details = self.check_metric_degradation(model_type)
        
        return {
            'model_type': model_type,
            'timestamp': datetime.utcnow().isoformat(),
            'staleness': {
                'is_stale': stale,
                'message': stale_msg
            },
            'drift': {
                'drift_detected': drift,
                'details': drift_details
            },
            'degradation': {
                'is_degraded': degraded,
                'details': degradation_details
            },
            'overall_health': 'healthy' if not (stale or drift or degraded) else 'unhealthy'
        }
    
    def generate_alert_if_needed(self, model_type: str) -> Optional[str]:
        """
        Generate alert message if any health check fails.
        
        Args:
            model_type: Type of model
            
        Returns:
            Alert message if issues found, None if healthy
        """
        report = self.get_full_health_report(model_type)
        
        if report['overall_health'] == 'healthy':
            return None
        
        alerts = []
        
        if report['staleness']['is_stale']:
            alerts.append(f"STALENESS: {report['staleness']['message']}")
        
        if report['drift']['drift_detected']:
            alerts.append(f"DRIFT: Prediction distribution has changed significantly")
        
        if report['degradation']['is_degraded']:
            details = report['degradation']['details']
            alerts.append(f"DEGRADATION: {details.get('metric')} is {details.get('value'):.2%} (threshold: {details.get('threshold'):.2%})")
        
        return " | ".join(alerts)


class PipelineMonitor:
    """
    Monitor the overall continuous training pipeline health.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checker = ModelHealthCheck()
    
    def get_pipeline_status(self) -> Dict:
        """
        Get overall pipeline status for all models.
        
        Returns:
            Dictionary with status of each model type
        """
        model_types = ['churn', 'product_area']
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'models': {}
        }
        
        for model_type in model_types:
            try:
                report = self.health_checker.get_full_health_report(model_type)
                status['models'][model_type] = report
            except Exception as e:
                self.logger.error(f"Error getting status for {model_type}: {e}")
                status['models'][model_type] = {
                    'error': str(e),
                    'overall_health': 'unknown'
                }
        
        return status
    
    def get_alerts(self) -> Dict[str, List[str]]:
        """
        Get all active alerts for the pipeline.
        
        Returns:
            Dictionary mapping model types to alert messages
        """
        model_types = ['churn', 'product_area']
        alerts = {}
        
        for model_type in model_types:
            alert = self.health_checker.generate_alert_if_needed(model_type)
            if alert:
                alerts[model_type] = [alert]
        
        return alerts
