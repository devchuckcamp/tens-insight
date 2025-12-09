"""Account churn prediction model using TensorFlow.

Simple feed-forward neural network for predicting account churn risk
based on feedback patterns and account characteristics.
"""

import logging
import os
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib

from ..config import get_config

logger = logging.getLogger(__name__)


class ChurnModel:
    """Account churn prediction model."""
    
    def __init__(self, input_dim: int, model_version: str = "v1"):
        """Initialize the churn model.
        
        Args:
            input_dim: Number of input features
            model_version: Version identifier for the model
        """
        self.input_dim = input_dim
        self.model_version = model_version
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.config = get_config()
        
    def build_model(self) -> keras.Model:
        """Build the neural network architecture."""
        # Get hyperparameters from config
        batch_size = self.config.batch_size
        
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        logger.info(f"Built churn model with {self.input_dim} input features")
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> keras.callbacks.History:
        """Train the churn model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Set up callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        logger.info(f"Training churn model on {len(X_train)} samples")
        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log final metrics
        final_metrics = {k: v[-1] for k, v in history.history.items()}
        logger.info(f"Training complete. Final metrics: {final_metrics}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict churn probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of churn probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def predict_with_health_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict churn probability and health score.
        
        Health score is the inverse of churn probability (0-100 scale).
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (churn_probabilities, health_scores)
        """
        churn_probs = self.predict(X)
        health_scores = (1 - churn_probs) * 100  # 0-100 scale
        return churn_probs, health_scores
    
    def save(self, model_name: str = "churn_model"):
        """Save the model and preprocessing artifacts.
        
        Args:
            model_name: Base name for saved files
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        models_dir = self.config.models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(models_dir, f"{model_name}_{self.model_version}.keras")
        self.model.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(models_dir, f"{model_name}_{self.model_version}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    
    def load(self, model_name: str = "churn_model"):
        """Load a saved model and preprocessing artifacts.
        
        Args:
            model_name: Base name of saved files
        """
        models_dir = self.config.models_dir
        
        # Load Keras model
        model_path = os.path.join(models_dir, f"{model_name}_{self.model_version}.keras")
        self.model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, f"{model_name}_{self.model_version}_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning("Scaler file not found")
    
    def get_risk_category(self, churn_prob: float) -> str:
        """Convert churn probability to risk category.
        
        Args:
            churn_prob: Churn probability (0-1)
            
        Returns:
            Risk category: 'low', 'medium', 'high', or 'critical'
        """
        if churn_prob < 0.25:
            return 'low'
        elif churn_prob < 0.5:
            return 'medium'
        elif churn_prob < 0.75:
            return 'high'
        else:
            return 'critical'


def create_churn_model(input_dim: int, model_version: str = "v1") -> ChurnModel:
    """Factory function to create a churn model instance.
    
    Args:
        input_dim: Number of input features
        model_version: Version identifier
        
    Returns:
        ChurnModel instance
    """
    return ChurnModel(input_dim=input_dim, model_version=model_version)
