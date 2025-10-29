# Wine Quality Prediction – MLOps Project

This project demonstrates how I built a simple but complete MLOps pipeline for predicting wine quality using machine learning.  
It includes data loading, model training, experiment tracking, and deployment through a FastAPI REST service.


##  Overview

This project implements an end-to-end MLOps system for wine quality prediction, demonstrating best practices in:
- **Modular Architecture**: Clean separation of data, models, and serving layers
- **Experiment Tracking**: MLflow integration for metrics and artifact management
- **API Deployment**: FastAPI REST endpoint with input validation
- **CI/CD Automation**: GitHub Actions workflow for continuous training and deployment
- **Observability**: Comprehensive logging across all pipeline components

## Dataset

The Wine Quality dataset from UCI Machine Learning Repository contains physicochemical test results for Portuguese "Vinho Verde" red wine with quality scores (0-10).

**Features** (11 physicochemical attributes):
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free/Total sulfur dioxide
- Density, pH, Sulphates, Alcohol

**Target**: Quality score (regression task)

##  Architecture

```
wine-quality-mlops/
├── src/
│   ├── data/           # Data ingestion and preprocessing
│   ├── models/         # Model training with MLflow
│   ├── api/            # FastAPI serving endpoint
│   └── utils/          # Logging and configuration utilities
├── tests/              # Unit and integration tests
├── config/             # Configuration files (YAML)
├── data/               # Raw dataset
├── models/             # Saved model artifacts
├── mlruns/             # MLflow experiment tracking
├── .github/workflows/  # CI/CD automation
└── train_pipeline.py   # Main training orchestration script
```

### Design Decisions & Trade-offs

#### 1. **Model Choice: Random Forest Regressor**
- **Why**: Robust, interpretable, handles non-linear relationships
- **Trade-off**: Slightly slower than linear models but better accuracy


#### 2. **MLflow for Experiment Tracking**
- **Why**: Industry standard, comprehensive tracking, model registry
- **Trade-off**: Adds dependency but essential for reproducibility
- **Alternative**: Weights & Biases for more advanced features

#### 3. **FastAPI for Serving**
- **Why**: Modern, fast, automatic OpenAPI documentation
- **Trade-off**: Less mature than Flask but better performance
- **Alternative**: Flask for simpler deployment, TorchServe for PyTorch models

#### 4. **Configuration via YAML**
- **Why**: Easy to read, modify parameters without code changes
- **Trade-off**: Less dynamic than database config but simpler
- **Alternative**: Environment variables or config server

#### 5. **GitHub Actions for CI/CD**
GitHub Actions handled automation well for a small project

 
### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd wine-quality-mlops

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 1. Train the Model

```bash
python train_pipeline.py
```

This will:
- Load and preprocess the wine quality dataset
- Train a Random Forest model with configured hyperparameters
- Log metrics, parameters, and artifacts to MLflow
- Save the trained model to `models/latest_model.pkl`

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
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
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

**Make a Prediction:**
```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Interactive API Documentation:**
Visit `http://localhost:5000/docs` for Swagger UI with interactive testing.

#### 4. View MLflow Experiments

```bash
mlflow ui --host 0.0.0.0 --port 5001
```

Visit `http://localhost:5001` to view experiment metrics, parameters, and artifacts.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) automatically:

1. **On Push/PR to main**:
   - Sets up Python environment
   - Installs dependencies
   - Runs test suite
   - Trains model with latest data
   - Uploads model artifacts

2. **Deployment Check**:
   - Downloads trained model
   - Starts API server
   - Validates health endpoint

### Triggering the Workflow

```bash
# Push to main branch
git push origin main

# Or manually trigger
# Go to Actions tab → ML Training Pipeline → Run workflow
```

## 📝 Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  params:
    n_estimators: 100      # Number of trees
    max_depth: 10          # Tree depth
    min_samples_split: 5   # Min samples for split

data:
  test_size: 0.2          # Train/test split ratio
  random_state: 42        # Reproducibility seed
```

## 🧪 API Endpoints

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

## 📊 Model Performance

**Current Baseline (Random Forest):**
- Test RMSE: ~0.65
- Test MAE: ~0.50
- Test R²: ~0.35

**Feature Importance (Top 5):**
1. Alcohol
2. Sulphates
3. Volatile acidity
4. Total sulfur dioxide
5. Density

##  Extending the Pipeline

### Adding a New Model

1. Implement in `src/models/trainer.py`
2. Update `config/config.yaml` with model type
3. Retrain: `python train_pipeline.py`

### Adding New Features

1. Modify `src/data/data_loader.py` for feature engineering
2. Update `src/api/main.py` input schema
3. Retrain and redeploy

### Monitoring & Alerts

**Recommendations for Production:**
- Add Prometheus metrics export
- Implement data drift detection
- Set up alerting for model performance degradation
- Add automated retraining triggers

##  Development

### Project Structure Explanation

- **`src/data/`**: Data loading with validation and preprocessing
- **`src/models/`**: Training logic with MLflow integration
- **`src/api/`**: FastAPI application with input validation
- **`src/utils/`**: Shared utilities (logging, config)
- **`tests/`**: Unit tests for all components
- **`train_pipeline.py`**: Main orchestration script

### Adding Tests

```python
# tests/test_your_module.py
def test_your_feature():
    # Your test implementation
    assert True
```

##  Dependencies

Core libraries:
- **scikit-learn**: Model training
- **pandas/numpy**: Data manipulation
- **mlflow**: Experiment tracking
- **fastapi/uvicorn**: API serving
- **pydantic**: Data validation
- **pytest**: Testing

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit: `git commit -am 'Add feature'`
6. Push: `git push origin feature/your-feature`
7. Submit a Pull Request

##  License

This project is for assessment purposes.