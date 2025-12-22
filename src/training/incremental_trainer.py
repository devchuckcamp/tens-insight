"""
Incremental training logic for deciding when and how to retrain models.

Provides functionality for:
- Data freshness checking
- Training necessity evaluation
- Automatic version incrementing
- Training decision logging
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy import text

from src.db import get_connection
from src.versioning import ModelRegistry

logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """
    Manages incremental training decisions based on data freshness and performance metrics.
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.logger = logging.getLogger(__name__)
    
    def check_training_necessity(
        self,
        model_type: str,
        min_new_data_percent: float = 5.0,
        min_days_since_training: int = 7,
        min_new_records: int = 100
    ) -> Tuple[bool, str]:
        """
        Determine if a model needs retraining based on data freshness.
        
        Args:
            model_type: 'churn' or 'product_area'
            min_new_data_percent: Minimum % of new data to trigger retraining
            min_days_since_training: Minimum days since last training
            min_new_records: Minimum absolute number of new records required
            
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        try:
            with get_connection() as conn:
                # Get last training date
                query = text("""
                    SELECT mv.created_at
                    FROM model_versions mv
                    WHERE mv.model_type = :model_type
                    ORDER BY mv.created_at DESC
                    LIMIT 1
                """)
                result = conn.execute(query, {"model_type": model_type})
                last_training = result.fetchone()
                last_training_date = last_training[0] if last_training else None
                
                # Get latest feedback date and total count
                query = text("""
                    SELECT MAX(created_at) as latest_date, COUNT(*) as total_count
                    FROM feedback_enriched
                """)
                result = conn.execute(query)
                feedback_info = result.fetchone()
                latest_date = feedback_info[0] if feedback_info else None
                total_count = feedback_info[1] if feedback_info else 0
                
                if not latest_date or total_count == 0:
                    return False, "No feedback data available"
                
                # Calculate new data since last training
                if last_training_date:
                    query = text("""
                        SELECT COUNT(*) FROM feedback_enriched
                        WHERE created_at > :last_training_date
                    """)
                    result = conn.execute(query, {"last_training_date": last_training_date})
                    new_records = result.scalar() or 0
                    
                    percent_new = (new_records / total_count * 100) if total_count > 0 else 0
                    # Make datetime timezone-aware for comparison
                    now_utc = datetime.now(timezone.utc)
                    days_since_training = (now_utc - last_training_date).days
                else:
                    # First training
                    new_records = total_count
                    percent_new = 100.0
                    days_since_training = float('inf')
                
                # Decision logic
                reasons = []
                
                if days_since_training >= min_days_since_training:
                    reasons.append(f"{days_since_training} days since last training (min: {min_days_since_training})")
                
                if new_records >= min_new_records:
                    reasons.append(f"{new_records} new records (min: {min_new_records})")
                
                if percent_new >= min_new_data_percent:
                    reasons.append(f"{percent_new:.1f}% new data (min: {min_new_data_percent}%)")
                
                should_retrain = len(reasons) >= 2  # Require at least 2 conditions
                
                if should_retrain:
                    reason = " AND ".join(reasons)
                else:
                    reason = f"Insufficient data freshness: {new_records} new records ({percent_new:.1f}%), "
                    reason += f"{days_since_training} days since training"
                
                # Log the decision
                self.registry.log_data_freshness_decision(
                    model_type=model_type,
                    new_feedback_count=new_records,
                    decision_to_retrain=should_retrain,
                    reason=reason
                )
                
                return should_retrain, reason
        
        except Exception as e:
            self.logger.error(f"Error checking training necessity: {e}")
            return False, f"Error: {str(e)}"
    
    def get_next_version(self, model_type: str) -> str:
        """
        Generate the next version identifier for a model.
        
        Versions follow semantic versioning (v1, v2, etc.)
        
        Args:
            model_type: 'churn' or 'product_area'
            
        Returns:
            Next version string (e.g., 'v2', 'v3')
        """
        versions = self.registry.get_version_history(model_type, limit=100)
        
        if not versions:
            return "v1"
        
        # Parse existing versions
        max_major = 0
        for v in versions:
            version_str = v.version
            if version_str.startswith('v'):
                version_str = version_str[1:]
            
            if '.' in version_str:
                parts = version_str.split('.')
                major = int(parts[0])
            else:
                major = int(version_str)
            
            if major > max_major:
                max_major = major
        
        # Return next major version
        return f"v{max_major + 1}"
    
    def get_previous_version_id(self, model_type: str) -> Optional[int]:
        """
        Get the ID of the previously trained version.
        
        Args:
            model_type: Type of model
            
        Returns:
            Version ID of previous version, or None if this is first training
        """
        versions = self.registry.get_version_history(model_type, limit=2)
        
        if len(versions) >= 2:
            return versions[1].id  # Second most recent
        elif len(versions) == 1:
            return versions[0].id  # Only version
        
        return None
    
    def should_promote_to_production(
        self,
        model_type: str,
        new_version_id: int,
        improvement_threshold: float = 0.01
    ) -> Tuple[bool, Dict]:
        """
        Determine if a newly trained model should be promoted to production.
        
        Args:
            model_type: Type of model
            new_version_id: ID of the newly trained version
            improvement_threshold: Minimum improvement required (e.g., 0.01 for 1%)
            
        Returns:
            Tuple of (should_promote: bool, comparison_result: dict)
        """
        # Get active version
        active_version = self.registry.get_active_version(model_type)
        
        if not active_version:
            self.logger.info(f"No active version found for {model_type}, promoting new version")
            return True, {"reason": "first_deployment"}
        
        # Compare versions
        comparison = self.registry.compare_versions(
            model_type=model_type,
            previous_version_id=active_version.id,
            new_version_id=new_version_id,
            improvement_threshold=improvement_threshold
        )
        
        return comparison['promotion_recommended'], comparison
    
    def promote_version_to_production(
        self,
        model_type: str,
        version_id: int,
        promoted_by: str = "scheduler"
    ) -> bool:
        """
        Promote a version to production (make it active).
        
        Args:
            model_type: Type of model
            version_id: ID of version to promote
            promoted_by: User or system identifier
            
        Returns:
            True if promotion was successful
        """
        try:
            self.registry.deploy_version(
                model_type=model_type,
                version_id=version_id
            )
            self.logger.info(f"Successfully promoted {model_type} version {version_id} to production")
            return True
        except Exception as e:
            self.logger.error(f"Failed to promote version: {e}")
            return False
