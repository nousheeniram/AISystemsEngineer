from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import pandas as pd
from pathlib import Path
from src.models.trainer import WineQualityTrainer
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)

app = FastAPI(
    title="Wine Quality Prediction API",
    description="ML model serving endpoint for predicting wine quality",
    version="1.0.0"
)


class WineFeatures(BaseModel):
    """Input features for wine quality prediction."""
    fixed_acidity: float = Field(..., ge=0, description="Fixed acidity")
    volatile_acidity: float = Field(..., ge=0, description="Volatile acidity")
    citric_acid: float = Field(..., ge=0, description="Citric acid")
    residual_sugar: float = Field(..., ge=0, description="Residual sugar")
    chlorides: float = Field(..., ge=0, description="Chlorides")
    free_sulfur_dioxide: float = Field(..., ge=0, description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., ge=0, description="Total sulfur dioxide")
    density: float = Field(..., gt=0, description="Density")
    pH: float = Field(..., ge=0, le=14, description="pH level")
    sulphates: float = Field(..., ge=0, description="Sulphates")
    alcohol: float = Field(..., ge=0, description="Alcohol percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    quality: float
    message: str = "Prediction successful"


model = None
feature_names = None


@app.on_event("startup")
async def load_model():
    """Load model on API startup."""
    global model, feature_names
    
    try:
        config = load_config()
        model_path = config['api']['model_path']
        
        if not Path(model_path).exists():
            logger.warning(f"Model file not found at {model_path}. API will start but predictions will fail.")
            return
        
        model, feature_names = WineQualityTrainer.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Expected features: {feature_names}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("API started without model. Run training pipeline first.")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": "API is running" if model_loaded else "Model not loaded. Run training first."
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures):
    """
    Predict wine quality based on input features.
    
    Args:
        features: Wine chemical properties
    
    Returns:
        Predicted quality score
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: python train_pipeline.py"
        )
    
    try:
        input_data = pd.DataFrame([features.dict()])
        
        input_data.columns = [col.replace('_', ' ') for col in input_data.columns]
        
        if feature_names:
            input_data = input_data[feature_names]
        
        prediction = model.predict(input_data)[0]
        
        logger.info(f"Prediction made: {prediction:.2f}")
        
        return PredictionResponse(
            quality=round(float(prediction), 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(features_list: List[WineFeatures]):
    """
    Predict wine quality for multiple samples.
    
    Args:
        features_list: List of wine chemical properties
    
    Returns:
        List of predicted quality scores
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        input_data = pd.DataFrame([f.dict() for f in features_list])
        
        input_data.columns = [col.replace('_', ' ') for col in input_data.columns]
        
        if feature_names:
            input_data = input_data[feature_names]
        
        predictions = model.predict(input_data)
        
        logger.info(f"Batch prediction for {len(predictions)} samples")
        
        return [
            PredictionResponse(quality=round(float(pred), 2))
            for pred in predictions
        ]
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )
