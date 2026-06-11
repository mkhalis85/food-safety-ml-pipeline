"""
Data preprocessing module for the Food Safety ML Pipeline.
Handles feature engineering, scaling, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessor for food safety datasets.
    Handles all preprocessing steps including imputation, scaling, and encoding.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: str = "safety_status",
        scaling_method: str = "standard",
        random_state: int = 42,
    ):
        """
        Initialize the preprocessor.

        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_column: Name of the target column
            scaling_method: Scaling method ('standard', 'robust', or 'none')
            random_state: Random seed for reproducibility
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.scaling_method = scaling_method
        self.random_state = random_state

        # Initialize preprocessing components
        self.numeric_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
        self.label_encoder = None

        # Fit state
        self._is_fitted = False

    def _create_numeric_pipeline(self) -> Pipeline:
        """Create pipeline for numeric features."""
        steps = []

        # Imputation
        if self.scaling_method == "robust":
            imputer = SimpleImputer(strategy="median")
        else:
            imputer = SimpleImputer(strategy="mean")
        steps.append(("imputer", imputer))

        # Scaling
        if self.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = None

        if scaler:
            steps.append(("scaler", scaler))

        return Pipeline(steps)

    def _create_categorical_pipeline(self) -> Pipeline:
        """Create pipeline for categorical features."""
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            df: Training DataFrame

        Returns:
            Fitted preprocessor
        """
        logger.info("Fitting preprocessor...")

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Identify feature types if not provided
        if not self.numeric_features:
            self.numeric_features = X.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        logger.info(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        logger.info(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")

        # Create pipelines
        self.numeric_pipeline = self._create_numeric_pipeline()
        self.categorical_pipeline = self._create_categorical_pipeline()

        # Create column transformer
        transformers = []

        if self.numeric_features:
            transformers.append(("num", self.numeric_pipeline, self.numeric_features))

        if self.categorical_features:
            transformers.append(("cat", self.categorical_pipeline, self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        # Fit the preprocessor
        self.preprocessor.fit(X)

        # Encode target variable
        self.label_encoder = {
            "safe": 0,
            "unsafe": 1,
        }
        self.classes_ = ["safe", "unsafe"]

        self._is_fitted = True
        logger.info("Preprocessor fitted successfully")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed feature array
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming")

        X = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df
        return self.preprocessor.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data.

        Args:
            df: Training DataFrame

        Returns:
            Tuple of (transformed features, encoded target)
        """
        self.fit(df)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].map(self.label_encoder)

        X_transformed = self.transform(df)

        return X_transformed, y.values

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.0,
        stratify: bool = True,
    ) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation (from remaining after test split)
            stratify: Whether to stratify splits by target variable

        Returns:
            Dictionary with train_df, val_df (optional), test_df, and their features/targets
        """
        logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        stratify_col = y if stratify else None

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col,
        )

        result = {
            "X_train": X_train_val,
            "y_train": y_train_val,
            "X_test": X_test,
            "y_test": y_test,
            "train_df": pd.concat([X_train_val, y_train_val], axis=1),
            "test_df": pd.concat([X_test, y_test], axis=1),
        }

        # Second split: train vs val (if val_size specified)
        if val_size > 0:
            # Calculate val_size relative to remaining data
            val_size_adj = val_size / (1 - test_size)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adj,
                random_state=self.random_state,
                stratify=y_train_val if stratify else None,
            )

            result["X_val"] = X_val
            result["y_val"] = y_val
            result["val_df"] = pd.concat([X_val, y_val], axis=1)
            result["X_train"] = X_train
            result["y_train"] = y_train
            result["train_df"] = pd.concat([X_train, y_train], axis=1)

            logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        else:
            logger.info(f"Split sizes: train={len(X_train_val)}, test={len(X_test)}")

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.

        Returns:
            List of feature names
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        feature_names = []

        if self.numeric_features:
            feature_names.extend(self.numeric_features)

        if self.categorical_features and hasattr(self.preprocessor, "named_transformers_"):
            cat_transformer = self.preprocessor.named_transformers_["cat"]
            ohe = cat_transformer.named_steps["encoder"]
            cat_feature_names = ohe.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)

        return feature_names

    def save(self, output_dir: str):
        """
        Save fitted preprocessor to disk.

        Args:
            output_dir: Directory to save preprocessor files
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save preprocessor
        joblib.dump(self.preprocessor, output_path / "preprocessor.joblib")

        # Save metadata
        metadata = {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
            "scaling_method": self.scaling_method,
            "random_state": self.random_state,
            "label_encoder": self.label_encoder,
            "classes_": self.classes_,
        }
        joblib.dump(metadata, output_path / "preprocessor_metadata.joblib")

        logger.info(f"Preprocessor saved to {output_dir}")

    @classmethod
    def load(cls, input_dir: str) -> "DataPreprocessor":
        """
        Load fitted preprocessor from disk.

        Args:
            input_dir: Directory containing preprocessor files

        Returns:
            Loaded DataPreprocessor instance
        """
        input_path = Path(input_dir)

        # Load preprocessor
        preprocessor = joblib.load(input_path / "preprocessor.joblib")

        # Load metadata
        metadata = joblib.load(input_path / "preprocessor_metadata.joblib")

        # Create instance and restore state
        instance = cls(
            numeric_features=metadata["numeric_features"],
            categorical_features=metadata["categorical_features"],
            target_column=metadata["target_column"],
            scaling_method=metadata["scaling_method"],
            random_state=metadata["random_state"],
        )
        instance.preprocessor = preprocessor
        instance.label_encoder = metadata["label_encoder"]
        instance.classes_ = metadata["classes_"]
        instance._is_fitted = True

        logger.info(f"Preprocessor loaded from {input_dir}")

        return instance

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded target values.

        Args:
            y_encoded: Encoded target values

        Returns:
            Original target labels
        """
        reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        return np.array([reverse_encoder[val] for val in y_encoded])
