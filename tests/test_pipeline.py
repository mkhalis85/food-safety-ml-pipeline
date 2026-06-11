"""
Test suite for the Food Safety ML Pipeline.
Run with: pytest tests/ -v --cov=src
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataValidator:
    """Tests for data validation module."""

    def test_validate_schema_valid(self):
        """Test schema validation with valid data."""
        from src.utils.data_validator import DataValidator

        df = pd.DataFrame({
            "temperature": [4.0, 5.0, 6.0],
            "ph_level": [6.5, 7.0, 6.8],
            "moisture_content": [80, 75, 85],
            "total_plate_count": [1000, 2000, 1500],
            "coliform_count": [10, 20, 15],
            "storage_time_days": [5, 10, 7],
            "water_activity": [0.85, 0.88, 0.82],
            "food_type": ["dairy", "meat", "produce"],
            "packaging_type": ["refrigerated", "vacuum", "ambient"],
            "supplier_id": ["SUP001", "SUP002", "SUP003"],
            "safety_status": ["safe", "safe", "unsafe"],
        })

        validator = DataValidator()
        is_valid, errors = validator.validate_schema(df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        from src.utils.data_validator import DataValidator

        df = pd.DataFrame({
            "temperature": [4.0, 5.0],
            "ph_level": [6.5, 7.0],
        })

        validator = DataValidator()
        is_valid, errors = validator.validate_schema(df)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_ranges(self):
        """Test range validation."""
        from src.utils.data_validator import DataValidator

        df = pd.DataFrame({
            "temperature": [4.0, 150.0, -30.0],  # Out of range values
            "ph_level": [6.5, 7.0, 8.0],
        })

        validator = DataValidator()
        is_valid, out_of_range = validator.validate_ranges(df)

        assert is_valid is False
        assert out_of_range["temperature"].sum() == 2


class TestDataLoader:
    """Tests for data loading module."""

    def test_generate_sample_data(self):
        """Test synthetic data generation."""
        from src.data_ingestion import generate_sample_data

        df = generate_sample_data(n_samples=100, random_state=42)

        assert len(df) == 100
        assert "safety_status" in df.columns
        assert "temperature" in df.columns
        assert "food_type" in df.columns

    def test_generate_data_contamination_ratio(self):
        """Test that generated data has expected contamination ratio."""
        from src.data_ingestion import generate_sample_data

        df = generate_sample_data(n_samples=1000, contamination_ratio=0.3)
        actual_ratio = (df["safety_status"] == "unsafe").mean()

        # Allow some variance
        assert 0.2 <= actual_ratio <= 0.4


class TestDataPreprocessor:
    """Tests for data preprocessing module."""

    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        from src.data_ingestion import DataPreprocessor
        from src.data_ingestion import generate_sample_data

        df = generate_sample_data(n_samples=100)

        preprocessor = DataPreprocessor(
            numeric_features=["temperature", "ph_level"],
            categorical_features=["food_type"],
        )

        X_transformed, y = preprocessor.fit_transform(df)

        assert X_transformed.shape[0] == 100
        assert len(y) == 100

    def test_preprocessor_split_data(self):
        """Test data splitting."""
        from src.data_ingestion import DataPreprocessor
        from src.data_ingestion import generate_sample_data

        df = generate_sample_data(n_samples=1000)

        preprocessor = DataPreprocessor()
        splits = preprocessor.split_data(df, test_size=0.2, val_size=0.1)

        assert "train_df" in splits
        assert "val_df" in splits
        assert "test_df" in splits
        assert len(splits["train_df"]) + len(splits["val_df"]) + len(splits["test_df"]) == 1000


class TestModelTrainer:
    """Tests for model training module."""

    def test_trainer_xgboost(self):
        """Test XGBoost model training."""
        from src.model_training import ModelTrainer
        from src.data_ingestion import generate_sample_data, DataPreprocessor

        df = generate_sample_data(n_samples=200)

        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df)

        trainer = ModelTrainer(model_type="xgboost", n_estimators=10)
        trainer.fit(X, y)

        predictions = trainer.predict(X)
        assert len(predictions) == len(y)

    def test_trainer_evaluate(self):
        """Test model evaluation."""
        from src.model_training import ModelTrainer
        from src.data_ingestion import generate_sample_data, DataPreprocessor

        df = generate_sample_data(n_samples=200)

        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df)

        trainer = ModelTrainer(model_type="random_forest", n_estimators=10)
        trainer.fit(X, y)

        metrics = trainer.evaluate(X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics


class TestModelEvaluator:
    """Tests for model evaluation module."""

    def test_evaluator_metrics(self):
        """Test evaluation metrics calculation."""
        from src.evaluation import ModelEvaluator

        y_true = ["safe", "unsafe", "safe", "unsafe", "safe"]
        y_pred = ["safe", "unsafe", "safe", "safe", "unsafe"]
        y_proba = [0.9, 0.8, 0.7, 0.4, 0.3]

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics

    def test_optimal_threshold(self):
        """Test optimal threshold finding."""
        from src.evaluation import ModelEvaluator

        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

        evaluator = ModelEvaluator()
        result = evaluator.find_optimal_threshold(y_true, y_proba, metric="f1")

        assert "optimal_threshold" in result
        assert 0 < result["optimal_threshold"] < 1


class TestPipeline:
    """Tests for the main pipeline."""

    def test_pipeline_run(self):
        """Test running the full pipeline."""
        from src.preprocessing import FoodSafetyPipeline

        pipeline = FoodSafetyPipeline(output_dir="test_models")

        results = pipeline.run_full_pipeline(
            data_path=None,
            model_type="random_forest",
            use_tuning=False,
            n_samples=500,
        )

        assert "evaluation" in results
        assert "pipeline_summary" in results
        assert results["pipeline_summary"]["n_samples"] == 500


class TestPredictor:
    """Tests for inference predictor."""

    def test_predictor_single_prediction(self):
        """Test single sample prediction."""
        from src.inference import FoodSafetyPredictor
        from src.preprocessing import FoodSafetyPipeline

        # Train a quick model
        pipeline = FoodSafetyPipeline(output_dir="test_predictor_models")
        pipeline.run_full_pipeline(
            model_type="random_forest",
            use_tuning=False,
            n_samples=200,
        )

        # Load predictor
        predictor = FoodSafetyPredictor(
            model_path="test_predictor_models/best_model.joblib",
            preprocessor_path="test_predictor_models/preprocessor",
        )

        # Make prediction
        sample = {
            "temperature": 4.0,
            "ph_level": 6.5,
            "moisture_content": 80,
            "total_plate_count": 1000,
            "coliform_count": 10,
            "storage_time_days": 5,
            "water_activity": 0.85,
            "food_type": "dairy",
            "packaging_type": "refrigerated",
            "supplier_id": "SUP001",
        }

        result = predictor.predict(sample, return_proba=True)

        assert "prediction" in result
        assert "probability" in result
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
