import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import pandas as pd
import joblib
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WineQualityTrainer:
    """Handles model training with MLflow experiment tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary containing model and MLflow settings
        """
        self.config = config
        self.model = None
        self.feature_names = None
        
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        logger.info(f"Initialized trainer with experiment: {config['mlflow']['experiment_name']}")
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Train model with MLflow tracking.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.feature_names = list(X_train.columns)
        
        with mlflow.start_run():
            logger.info("Starting model training...")
            
            model_params = self.config['model']['params']
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", self.config['model']['type'])
            
            self.model = RandomForestRegressor(**model_params)
            self.model.fit(X_train, y_train)
            
            logger.info("Model training completed")
            
            metrics = self._evaluate_model(X_train, X_test, y_train, y_test)
            
            mlflow.log_metrics(metrics)
            
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                input_example=X_train.iloc[:5]
            )
            
            self._log_feature_importance(X_train)
            
            logger.info(f"Training metrics: {metrics}")
            
            return metrics
    
    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on train and test sets.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        
        Returns:
            Dictionary of metrics
        """
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred)
        }
        
        return metrics
    
    def _log_feature_importance(self, X: pd.DataFrame):
        """
        Log feature importance to MLflow.
        
        Args:
            X: Features DataFrame
        """
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_dict = dict(zip(
                feature_importance['feature'],
                feature_importance['importance']
            ))
            
            mlflow.log_dict(importance_dict, "feature_importance.json")
            logger.info(f"Top 5 important features:\n{feature_importance.head()}")
    
    def save_model(self, output_path: str = "models/latest_model.pkl"):
        """
        Save trained model to disk.
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, output_file)
        logger.info(f"Model saved to {output_file}")
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[Any, list]:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to the saved model
        
        Returns:
            Tuple of (model, feature_names)
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model_data = joblib.load(model_file)
        logger.info(f"Model loaded from {model_path}")
        
        return model_data['model'], model_data['feature_names']
