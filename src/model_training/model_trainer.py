"""
Model training module for the Food Safety ML Pipeline.
Supports multiple model types including XGBoost, LightGBM, Random Forest, and Logistic Regression.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import joblib
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified model trainer supporting multiple classification algorithms.
    Provides training, evaluation, and model persistence capabilities.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
        use_gpu: bool = False,
        **model_kwargs,
    ):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'logistic_regression')
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration (if available)
            **model_kwargs: Additional model-specific parameters
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.model_kwargs = model_kwargs
        self.model = None
        self._is_fitted = False
        self.feature_names = None

        # Set default parameters
        self._set_default_params()

    def _set_default_params(self):
        """Set default parameters based on model type."""
        defaults = {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": self.random_state,
                "n_jobs": -1,
            },
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": self.random_state,
                "n_jobs": -1,
            },
        }

        # Merge defaults with provided kwargs
        model_defaults = defaults.get(self.model_type, {})
        for key, value in model_defaults.items():
            if key not in self.model_kwargs:
                self.model_kwargs[key] = value

    def _create_model(self) -> BaseEstimator:
        """Create model instance based on model type."""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                if self.use_gpu:
                    self.model_kwargs["tree_method"] = "gpu_hist"
                return XGBClassifier(**self.model_kwargs)
            except ImportError:
                logger.warning("XGBoost not installed, falling back to RandomForest")
                self.model_type = "random_forest"
                return self._create_model()

        elif self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                if self.use_gpu:
                    self.model_kwargs["device"] = "gpu"
                return LGBMClassifier(**self.model_kwargs)
            except ImportError:
                logger.warning("LightGBM not installed, falling back to RandomForest")
                self.model_type = "random_forest"
                return self._create_model()

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.model_kwargs)

        elif self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            # Add multi_class parameter for older sklearn versions
            if "multi_class" not in self.model_kwargs:
                self.model_kwargs["multi_class"] = "auto"
            return LogisticRegression(**self.model_kwargs)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        class_weight: Optional[str] = "balanced",
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
    ) -> "ModelTrainer":
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights (optional)
            class_weight: How to balance classes ('balanced' or None)
            early_stopping_rounds: Early stopping rounds (for boosting models)
            verbose: Whether to print training progress

        Returns:
            Fitted ModelTrainer
        """
        logger.info(f"Training {self.model_type} model...")

        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            X = np.asarray(X)

        y = np.asarray(y)

        # Handle class imbalance
        if class_weight == "balanced" and self.model_type in ["random_forest", "logistic_regression"]:
            self.model_kwargs["class_weight"] = "balanced"

        # Create and fit model
        self.model = self._create_model()

        # Prepare fit arguments
        fit_args = {"X": X, "y": y}

        if sample_weight is not None:
            fit_args["sample_weight"] = sample_weight

        # Handle validation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None and early_stopping_rounds:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            eval_set = [(X_val, np.asarray(y_val))]

            if self.model_type in ["xgboost", "lightgbm"]:
                fit_args["eval_set"] = eval_set
                fit_args["early_stopping_rounds"] = early_stopping_rounds
                if verbose:
                    fit_args["verbose"] = verbose

        # Suppress convergence warnings for logistic regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(**fit_args)

        self._is_fitted = True

        # Log training results
        train_pred = self.predict(X)
        train_acc = accuracy_score(y, train_pred)
        logger.info(f"Training accuracy: {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_acc = accuracy_score(np.asarray(y_val), val_pred)
            logger.info(f"Validation accuracy: {val_acc:.4f}")

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted class labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Prediction probabilities for each class
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of evaluation metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluating")

        y = np.asarray(y)
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Probability of positive class

        # Apply custom threshold if needed
        if threshold != 0.5:
            y_pred = (y_proba >= threshold).astype(int)
            # Map back to original labels if needed
            if hasattr(self, "classes_"):
                y_pred = np.array([self.classes_[pred] for pred in y_pred])

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(
                y, y_pred, output_dict=True, zero_division=0
            ),
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv_folds: int = 5,
        scoring: str = "f1",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")

        if isinstance(X, pd.DataFrame):
            X = X.values
        y = np.asarray(y)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(
            self._create_model(),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results = {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "scores": scores.tolist(),
            "cv_folds": cv_folds,
            "scoring_metric": scoring,
        }

        logger.info(f"Cross-validation {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance rankings
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        if self.model_type in ["xgboost", "lightgbm", "random_forest"]:
            importance = self.model.feature_importances_
        else:
            # For logistic regression, use absolute coefficient values
            importance = np.abs(self.model.coef_[0])

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def save(self, output_path: Union[str, Path]):
        """
        Save trained model to disk.

        Args:
            output_path: Path to save model file
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "model_kwargs": self.model_kwargs,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
        }

        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to {output_path}")

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "ModelTrainer":
        """
        Load trained model from disk.

        Args:
            input_path: Path to model file

        Returns:
            Loaded ModelTrainer instance
        """
        input_path = Path(input_path)
        model_data = joblib.load(input_path)

        instance = cls(
            model_type=model_data["model_type"],
            random_state=model_data["random_state"],
            **model_data["model_kwargs"],
        )
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance._is_fitted = True

        logger.info(f"Model loaded from {input_path}")

        return instance

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "use_gpu": self.use_gpu,
            **self.model_kwargs,
        }

    def set_params(self, **params) -> "ModelTrainer":
        """Set model parameters."""
        self.model_kwargs.update(params)
        if self._is_fitted:
            self.model.set_params(**params)
        return self
