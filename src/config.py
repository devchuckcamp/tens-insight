"""Configuration management for tens-insight.

Loads configuration from environment variables with sensible defaults.
"""

import os
from typing import Optional


class Config:
    """Application configuration."""
    
    def __init__(self):
        # Database configuration - required, no default
        self.database_url: str = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set it in .env file or environment."
            )
        
        # Model configuration
        self.models_dir: str = os.getenv('MODELS_DIR', 'models')
        
        # Training configuration
        self.random_seed: int = int(os.getenv('RANDOM_SEED', '42'))
        self.batch_size: int = int(os.getenv('BATCH_SIZE', '32'))
        self.epochs: int = int(os.getenv('EPOCHS', '50'))
        self.validation_split: float = float(os.getenv('VALIDATION_SPLIT', '0.2'))
        
        # Scoring configuration
        self.score_batch_size: int = int(os.getenv('SCORE_BATCH_SIZE', '1000'))
        
        # Logging
        self.log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    def __repr__(self) -> str:
        """String representation (masks sensitive data)."""
        return (
            f"Config(database_url=*****, "
            f"models_dir={self.models_dir}, "
            f"random_seed={self.random_seed})"
        )


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
