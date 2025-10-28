#!/usr/bin/env python3
"""
Training Pipeline for Wine Quality Prediction Model

This script orchestrates the complete ML training pipeline:
1. Data ingestion and preprocessing
2. Model training with hyperparameters
3. Experiment tracking with MLflow
4. Model evaluation and artifact saving
"""

import sys
from pathlib import Path

from src.data.data_loader import DataLoader
from src.models.trainer import WineQualityTrainer
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Execute the complete training pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting Wine Quality Model Training Pipeline")
        logger.info("=" * 60)
        
        config = load_config()
        
        logger.info("Step 1: Loading and preprocessing data")
        data_loader = DataLoader(
            data_path=config['data']['raw_data_path'],
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        X_train, X_test, y_train, y_test = data_loader.split_data()
        
        logger.info("Step 2: Training model with MLflow tracking")
        trainer = WineQualityTrainer(config)
        metrics = trainer.train(X_train, X_test, y_train, y_test)
        
        logger.info("Step 3: Saving trained model")
        trainer.save_model(config['api']['model_path'])
        
        logger.info("=" * 60)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Model Performance:")
        logger.info(f"  - Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  - Test MAE: {metrics['test_mae']:.4f}")
        logger.info(f"  - Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Model saved to: {config['api']['model_path']}")
        logger.info(f"MLflow tracking URI: {config['mlflow']['tracking_uri']}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
