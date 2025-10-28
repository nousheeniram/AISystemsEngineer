import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_without_model():
    """Test prediction endpoint behavior when model not loaded."""
    payload = {
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
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code in [200, 503]


def test_predict_invalid_input():
    """Test prediction with invalid input."""
    invalid_payload = {
        "fixed_acidity": -1.0,
        "volatile_acidity": 0.7,
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422
