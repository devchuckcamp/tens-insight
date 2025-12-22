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

from src.db import get_session
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
        
        A large change in mean/std of predictions might indicate:
        - Data shift (different patterns in feedback)
        - Model performance degradation
        
        Args:
            model_type: 'churn' or 'product_area'
            drift_threshold: Percentage change threshold (0.15 = 15%)
            
        Returns:
            Tuple of (drift_detected: bool, analysis: dict)
        """
        session = next(get_session())
        try:
            # Get recent prediction distributions
            query = text("""
                SELECT mv.version, pdh.metric_name, pdh.metric_value, pdh.scoring_date
                FROM prediction_distribution_history pdh
                JOIN model_versions mv ON pdh.model_version_id = mv.id
                WHERE mv.model_type = :model_type
                ORDER BY pdh.scoring_date DESC
                LIMIT 20
            """)
            
            results = session.execute(query, {
                "model_type": model_type
            }).fetchall()
            
            if not results:
                return False, {"reason": "No prediction history found"}
            
            # Group by version and metric
            versions_data = {}
            for version, metric_name, metric_value, scoring_date in results:
                if version not in versions_data:
                    versions_data[version] = {}
                if metric_name not in versions_data[version]:
                    versions_data[version][metric_name] = []
                versions_data[version][metric_name].append((metric_value, scoring_date))
            
            # Get most recent 2 versions
            versions_list = list(versions_data.keys())
            if len(versions_list) < 2:
                return False, {"reason": "Not enough versions for comparison"}
            
            current_version = versions_list[0]
            previous_version = versions_list[1]
            
            drift_analysis = {
                "current_version": current_version,
                "previous_version": previous_version,
                "drift_metrics": {}
            }
            
            drift_detected = False
            
            # Compare key metrics
            for metric_name in ['mean_prediction', 'std_prediction']:
                current = versions_data[current_version].get(metric_name, [])
                previous = versions_data[previous_version].get(metric_name, [])
                
                if current and previous:
                    curr_val = current[0][0]  # Most recent value
                    prev_val = previous[0][0]
                    
                    if prev_val != 0:
                        pct_change = abs((curr_val - prev_val) / prev_val)
                        
                        drift_analysis["drift_metrics"][metric_name] = {
                            "previous": prev_val,
                            "current": curr_val,
                            "pct_change": pct_change
                        }
                        
                        if pct_change > drift_threshold:
                            drift_detected = True
            
            return drift_detected, drift_analysis
            
        finally:
            session.close()
    
    def check_metric_degradation(
        self,
        model_type: str,
        min_acceptable_metric: float = 0.80
    ) -> Tuple[bool, Dict]:
        """
        Check if model metrics have degraded below acceptable levels.
        
        Args:
            model_type: 'churn' or 'product_area'
            min_acceptable_metric: Minimum acceptable accuracy/AUC threshold
            
        Returns:
            Tuple of (degraded: bool, metrics_report: dict)
        """
        active_version = self.registry.get_active_version(model_type)
        
        if not active_version:
            return False, {"reason": "No active model"}
        
        metrics = self.registry.get_best_metrics(active_version.id, dataset_type='test')
        
        if not metrics:
            return False, {"reason": "No metrics found"}
        
        report = {
            "version": active_version.version,
            "metrics": metrics,
            "below_threshold": []
        }
        
        degraded = False
        
        # Check accuracy and AUC
        for metric_name in ['accuracy', 'auc']:
            if metric_name in metrics:
                value = metrics[metric_name]
                if value < min_acceptable_metric:
                    report["below_threshold"].append({
                        "metric": metric_name,
                        "value": value,
                        "threshold": min_acceptable_metric
                    })
                    degraded = True
        
        return degraded, report
    
    def get_full_health_report(
        self,
        model_type: str
    ) -> Dict:
        """
        Generate a comprehensive health report for a model.
        
        Returns:
            Dict with complete health status
        """
        report = {
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Staleness check
        is_stale, staleness_reason = self.check_model_staleness(model_type)
        report["checks"]["staleness"] = {
            "passed": not is_stale,
            "reason": staleness_reason
        }
        
        # Drift check
        drift_detected, drift_analysis = self.check_prediction_drift(model_type)
        report["checks"]["prediction_drift"] = {
            "passed": not drift_detected,
            "analysis": drift_analysis
        }
        
        # Degradation check
        degraded, degradation_report = self.check_metric_degradation(model_type)
        report["checks"]["metric_degradation"] = {
            "passed": not degraded,
            "report": degradation_report
        }
        
        # Overall health
        all_passed = all(check["passed"] for check in report["checks"].values())
        report["overall_status"] = "healthy" if all_passed else "degraded"
        
        return report
    
    def generate_alert_if_needed(self, model_type: str) -> Optional[str]:
        """
        Generate an alert message if health issues are detected.
        
        Args:
            model_type: 'churn' or 'product_area'
            
        Returns:
            Alert message if issues found, None otherwise
        """
        report = self.get_full_health_report(model_type)
        
        alerts = []
        
        if not report["checks"]["staleness"]["passed"]:
            alerts.append(f"ALERT: {report['checks']['staleness']['reason']}")
        
        if not report["checks"]["prediction_drift"]["passed"]:
            drift_info = report["checks"]["prediction_drift"]["analysis"]["drift_metrics"]
            for metric, data in drift_info.items():
                if data["pct_change"] > 0.15:
                    alerts.append(
                        f"ALERT: Prediction drift detected in {metric}: "
                        f"{data['pct_change']:.1%} change"
                    )
        
        if not report["checks"]["metric_degradation"]["passed"]:
            degradation = report["checks"]["metric_degradation"]["report"]
            for below_threshold in degradation.get("below_threshold", []):
                alerts.append(
                    f"ALERT: {below_threshold['metric']} degraded to "
                    f"{below_threshold['value']:.4f} (threshold: {below_threshold['threshold']})"
                )
        
        return "\n".join(alerts) if alerts else None


class PipelineMonitor:
    """
    Monitor overall pipeline health and performance.
    """
    
    def __init__(self):
        self.health_check = ModelHealthCheck()
        self.logger = logging.getLogger(__name__)
    
    def check_all_models(self) -> Dict[str, Dict]:
        """Check health of all model types."""
        results = {}
        
        for model_type in ['churn', 'product_area']:
            report = self.health_check.get_full_health_report(model_type)
            results[model_type] = report
            
            # Log alerts if any
            alert = self.health_check.generate_alert_if_needed(model_type)
            if alert:
                self.logger.warning(f"Health issues detected for {model_type}:\n{alert}")
        
        return results
    
    def get_pipeline_status(self) -> str:
        """Get overall pipeline status as a string."""
        results = self.check_all_models()
        
        all_healthy = all(
            r["overall_status"] == "healthy" 
            for r in results.values()
        )
        
        if all_healthy:
            return "healthy"
        else:
            return "degraded"
