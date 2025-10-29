# Wine Quality Prediction – MLOps Project
**Author:** Nousheen Iram  
This project demonstrates how I built a lightweight end-to-end MLOps pipeline for predicting wine quality using the UCI dataset.
It covers the full model lifecycle — data ingestion, training, experiment tracking, deployment, and CI/CD automation.


##  Highlights

**Preprocessing:** Validates and loads data with a modular pipeline.

**Training:** Random Forest Regressor with evaluation metrics (RMSE, MAE, R²).

**Tracking:** MLflow for experiment logging and artifact storage.

**Serving:** FastAPI endpoint for predictions with health checks.

**Automation:** GitHub Actions workflow for testing and retraining.

## Dataset

The Wine Quality dataset from UCI Machine Learning Repository contains physicochemical test results for Portuguese "Vinho Verde" red wine with quality scores (0-10).

**Features** (11 physicochemical attributes):
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free/Total sulfur dioxide
- Density, pH, Sulphates, Alcohol

**Target**: Quality score (regression task)

##  Project Structure

```
AISystemsEngineer/
├── src/                # data, models, api, utils
├── tests/              # API & pipeline tests
├── config/             # YAML configs
├── models/             # Saved artifacts
├── data/               # Input CSV
├── .github/workflows/  # CI/CD automation
└── train_pipeline.py   # Main training script
 script
```

### Design Decisions & Trade-offs

#### **Model Choice: Random Forest Regressor**
- **Why**: Robust, interpretable, handles non-linear relationships
- **Trade-off**: Slightly slower than linear models but better accuracy

#### **FastAPI for Serving**
- **Why**: Modern, fast, automatic OpenAPI documentation

#### **Configuration via YML**
- **Why**: Easy to read, modify parameters without code changes

#### **GitHub Actions for CI/CD**
GitHub Actions handled automation well for a small project

 
### Prerequisites
- Python 3.11+
- pip

### Installation

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 1. Train the Model

```bash
python train_pipeline.py
```

-  The trained model is Saved to `models/latest_model.pkl`

**Expected Output:**
```
Training Pipeline Completed Successfully!
Model Performance:
  - Test RMSE: ~0.65
  - Test MAE: ~0.50
  - Test R²: ~0.35
```

#### 2. Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or simply:
```bash
python src/api/main.py
```

#### 3. Test the API

**Health Check:**
```bash
curl http://localhost:5000/health
```


**Interactive API Documentation:**
Visit `http://localhost:5000/docs` for Swagger UI with interactive testing.

#### 4. View MLflow Experiments

```bash
mlflow ui --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` to view experiment metrics, parameters, and artifacts.

##  CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) automatically:


##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | Interactive API documentation |

### Example API Response

```json
{
  "quality": 5.85,
  "message": "Prediction successful"
}
```

##  Model Performance

**Current Baseline (Random Forest):**
- Test RMSE: ~0.65
- Test MAE: ~0.50
- Test R²: ~0.35

## Housekeeping
.venv/
mlruns/
models/
__pycache__/
.DS_Store
.local/


## Closing Note

This project shows how to turn a simple ML model into a reproducible, trackable, and deployable pipeline using real MLOps concepts.
It’s lightweight, easy to extend, and ready for production-style experimentation.
