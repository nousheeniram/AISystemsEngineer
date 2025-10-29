import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Handles data loading, validation, and preprocessing for Wine Quality dataset."""

    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the dataset CSV file
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.data: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with validation.
        Tries the default comma first, falls back to semicolon if needed.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")

        # Try comma-separated first (your file is now comma-separated)
        try:
            df = pd.read_csv(self.data_path)  # default sep=','
            if df.shape[1] > 1:
                self.data = df
            else:
                # One giant column → likely semicolon file
                raise ValueError("Single-column read with ',', trying ';'")
        except Exception:
            # Fallback to original UCI semicolon format
            self.data = pd.read_csv(self.data_path, sep=';')

        logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} features")
        self._validate_data()
        return self.data

    def _validate_data(self):
        """Validate loaded data for missing values and expected columns."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # If delimiter was wrong, you'd see a single wide column; catch early.
        if self.data.shape[1] == 1:
            raise ValueError(
                "Detected a single-column dataframe. CSV delimiter may be wrong. "
                "Ensure the file is comma-separated or fallback to semicolon."
            )

        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in dataset")

        if 'quality' not in self.data.columns:
            raise ValueError("Target column 'quality' not found in dataset")

        logger.info("Data validation passed")

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data by separating features and target.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.data is None:
            self.load_data()

        X = self.data.drop('quality', axis=1)
        y = self.data['quality']

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Feature columns: {list(X.columns)}")
        # For regression this is just a histogram-like count of label values (still fine)
        try:
            logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        except Exception:
            pass

        return X, y

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.preprocess_data()

        # quality is integer-valued; stratify keeps class balance if you treat it as classification
        # If your trainer is regression, this still runs, but you can remove stratify if needed.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> list:
        """
        Get list of feature names.

        Returns:
            List of feature column names
        """
        if self.data is None:
            self.load_data()

        return [col for col in self.data.columns if col != 'quality']
