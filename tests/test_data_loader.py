import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader


@pytest.fixture
def data_loader():
    """Fixture for DataLoader instance."""
    return DataLoader(
        data_path="data/winequality-red.csv",
        test_size=0.2,
        random_state=42
    )


def test_load_data(data_loader):
    """Test data loading functionality."""
    df = data_loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'quality' in df.columns


def test_validate_data(data_loader):
    """Test data validation."""
    data_loader.load_data()
    
    assert data_loader.data is not None
    assert 'quality' in data_loader.data.columns


def test_preprocess_data(data_loader):
    """Test data preprocessing."""
    X, y = data_loader.preprocess_data()
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'quality' not in X.columns
    assert len(X) == len(y)


def test_split_data(data_loader):
    """Test train/test split."""
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples
    assert 0.15 < test_ratio < 0.25


def test_get_feature_names(data_loader):
    """Test feature names retrieval."""
    feature_names = data_loader.get_feature_names()
    
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert 'quality' not in feature_names
