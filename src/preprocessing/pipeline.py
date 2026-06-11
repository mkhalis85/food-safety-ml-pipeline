"""
End-to-end pipeline for the Food Safety ML system.
Orchestrates data loading, preprocessing, training, evaluation, and model persistence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import logging
from datetime import datetime

from src.utils import setup_logger, load_config
from src.data_ingestion import DataLoader, DataPreprocessor, generate_sample_data
from src.model_training import ModelTrainer, HyperparameterTuner
from src.evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class FoodSafetyPipeline:
    """
    Complete ML pipeline for food safety classification.
    Manages the entire workflow from raw data to production-ready model.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "models",
    ):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration YAML file
            output_dir: Directory to save models and artifacts
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = None

        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = None
        self.trainer = None
        self.evaluator = ModelEvaluator()

        # Pipeline state
        self.data = None
        self.results = {}
        self._is_trained = False

    def load_or_generate_data(
        self,
        data_path: Optional[str] = None,
        generate_synthetic: bool = True,
        n_samples: int = 5000,
    ) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data.

        Args:
            data_path: Path to data file (optional)
            generate_synthetic: Whether to generate synthetic data if no file provided
            n_samples: Number of synthetic samples to generate

        Returns:
            Loaded or generated DataFrame
        """
        if data_path and Path(data_path).exists():
            logger.info(f"Loading data from {data_path}")
            self.data = self.data_loader.load(data_path)
        elif generate_synthetic:
            logger.info(f"Generating {n_samples} synthetic samples")
            self.data = generate_sample_data(n_samples=n_samples)

            # Save generated data
            generated_path = self.output_dir.parent / "data" / "raw" / "food_safety_data.csv"
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(generated_path, index=False)
            logger.info(f"Generated data saved to {generated_path}")
        else:
            raise ValueError("No data provided and synthetic generation disabled")

        logger.info(f"Data loaded: {len(self.data)} samples, {len(self.data.columns)} columns")
        return self.data

    def preprocess_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Preprocess and split data.

        Args:
            test_size: Proportion for testing
            val_size: Proportion for validation

        Returns:
            Dictionary with train/val/test splits
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded first")

        logger.info("Preprocessing data...")

        # Get config values or use defaults
        if self.config:
            numeric_features = self.config.preprocessing.numeric_features
            categorical_features = self.config.preprocessing.categorical_features
            scaling_method = self.config.preprocessing.scaling_method
            random_state = self.config.data.random_state
        else:
            numeric_features = [
                "temperature", "ph_level", "moisture_content",
                "total_plate_count", "coliform_count",
                "storage_time_days", "water_activity"
            ]
            categorical_features = ["food_type", "packaging_type", "supplier_id"]
            scaling_method = "standard"
            random_state = 42

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaling_method=scaling_method,
            random_state=random_state,
        )

        # Split and transform
        splits = self.preprocessor.split_data(
            self.data,
            test_size=test_size,
            val_size=val_size,
        )

        # Fit on training data
        X_train_transformed, y_train = self.preprocessor.fit_transform(splits["train_df"])

        # Transform validation and test sets
        if "val_df" in splits:
            X_val_transformed = self.preprocessor.transform(splits["val_df"])
            y_val = splits["val_df"][self.preprocessor.target_column].map(
                self.preprocessor.label_encoder
            ).values
            splits["X_val_transformed"] = X_val_transformed
            splits["y_val"] = y_val

        X_test_transformed = self.preprocessor.transform(splits["test_df"])
        y_test = splits["test_df"][self.preprocessor.target_column].map(
            self.preprocessor.label_encoder
        ).values

        splits["X_train_transformed"] = X_train_transformed
        splits["y_train"] = y_train
        splits["X_test_transformed"] = X_test_transformed
        splits["y_test"] = y_test

        # Save preprocessor
        self.preprocessor.save(self.output_dir / "preprocessor")

        logger.info("Data preprocessing complete")
        return splits

    def train_model(
        self,
        splits: Dict[str, Any],
        model_type: str = "xgboost",
        use_tuning: bool = True,
        n_trials: int = 30,
    ) -> ModelTrainer:
        """
        Train the model with optional hyperparameter tuning.

        Args:
            splits: Data splits from preprocessing
            model_type: Type of model to train
            use_tuning: Whether to perform hyperparameter tuning
            n_trials: Number of tuning trials

        Returns:
            Trained ModelTrainer
        """
        logger.info(f"Training {model_type} model...")

        # Get config values
        if self.config:
            early_stopping = self.config.training.early_stopping_rounds
            class_weight = self.config.training.class_weight_balance
        else:
            early_stopping = 10
            class_weight = True

        # Initialize trainer
        self.trainer = ModelTrainer(
            model_type=model_type,
            random_state=42,
        )

        X_train = splits["X_train_transformed"]
        y_train = splits["y_train"]

        # Validation set for early stopping
        X_val = splits.get("X_val_transformed")
        y_val = splits.get("y_val")

        # Hyperparameter tuning
        if use_tuning:
            logger.info("Performing hyperparameter tuning...")
            tuner = HyperparameterTuner(
                model_type=model_type,
                n_trials=n_trials,
                cv_folds=5,
                scoring_metric="f1",
            )

            tuning_results = tuner.tune(X_train, y_train, show_progress=True)
            best_params = tuning_results["best_params"]

            logger.info(f"Best parameters: {best_params}")

            # Update trainer with best params
            self.trainer.set_params(**best_params)

        # Train final model
        self.trainer.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=early_stopping if use_tuning else None,
            class_weight="balanced" if class_weight else None,
        )

        # Save model
        self.trainer.save(self.output_dir / "best_model.joblib")

        self._is_trained = True
        logger.info("Model training complete")

        return self.trainer

    def evaluate_model(
        self,
        splits: Dict[str, Any],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            splits: Data splits
            threshold: Classification threshold

        Returns:
            Evaluation metrics dictionary
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        logger.info("Evaluating model...")

        X_test = splits["X_test_transformed"]
        y_test = splits["y_test"]

        # Get predictions
        y_pred = self.trainer.predict(X_test)
        y_proba = self.trainer.predict_proba(X_test)[:, 1]

        # Evaluate
        metrics = self.evaluator.evaluate(
            y_test,
            y_pred,
            y_proba,
            threshold=threshold,
        )

        # Feature importance
        feature_importance = self.trainer.get_feature_importance(top_n=10)
        metrics["feature_importance"] = feature_importance.to_dict("records")

        # Generate report
        report = self.evaluator.generate_report(metrics)
        print(report)

        # Save results
        self.evaluator.save(self.output_dir / "evaluation_results.joblib")

        # Save report
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        self.results["evaluation"] = metrics
        logger.info("Model evaluation complete")

        return metrics

    def run_full_pipeline(
        self,
        data_path: Optional[str] = None,
        model_type: str = "xgboost",
        use_tuning: bool = True,
        n_trials: int = 30,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to evaluation.

        Args:
            data_path: Path to data file (or generate synthetic)
            model_type: Model type to train
            use_tuning: Whether to tune hyperparameters
            n_trials: Number of tuning trials
            test_size: Test set proportion
            val_size: Validation set proportion

        Returns:
            Complete pipeline results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("STARTING FOOD SAFETY ML PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load/generate data
        self.load_or_generate_data(data_path=data_path)

        # Step 2: Preprocess
        splits = self.preprocess_data(test_size=test_size, val_size=val_size)

        # Step 3: Train
        self.train_model(
            splits,
            model_type=model_type,
            use_tuning=use_tuning,
            n_trials=n_trials,
        )

        # Step 4: Evaluate
        metrics = self.evaluate_model(splits, threshold=0.5)

        # Compile results
        end_time = datetime.now()
        self.results["pipeline_summary"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "n_samples": len(self.data),
            "n_features": len(self.preprocessor.numeric_features) + len(self.preprocessor.categorical_features),
            "model_type": model_type,
            "hyperparameter_tuning": use_tuning,
            "n_tuning_trials": n_trials if use_tuning else 0,
            "output_directory": str(self.output_dir),
        }

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total duration: {self.results['pipeline_summary']['duration_seconds']:.2f} seconds")
        logger.info("=" * 60)

        return self.results

    def save_pipeline_state(self, path: Optional[str] = None):
        """Save complete pipeline state."""
        if path is None:
            path = self.output_dir / "pipeline_state.joblib"
        else:
            path = Path(path)

        state = {
            "results": self.results,
            "config_path": self.config_path,
            "output_dir": str(self.output_dir),
            "_is_trained": self._is_trained,
        }

        joblib.dump(state, path)
        logger.info(f"Pipeline state saved to {path}")

    @classmethod
    def load_from_saved(cls, model_dir: str, config_path: Optional[str] = None) -> "FoodSafetyPipeline":
        """
        Load a trained pipeline from saved artifacts.

        Args:
            model_dir: Directory containing saved model and preprocessor
            config_path: Path to config file

        Returns:
            Loaded FoodSafetyPipeline instance
        """
        pipeline = cls(config_path=config_path, output_dir=model_dir)

        # Load model
        model_path = Path(model_dir) / "best_model.joblib"
        if model_path.exists():
            pipeline.trainer = ModelTrainer.load(model_path)
            pipeline._is_trained = True

        # Load preprocessor
        preprocessor_dir = Path(model_dir) / "preprocessor"
        if preprocessor_dir.exists():
            pipeline.preprocessor = DataPreprocessor.load(str(preprocessor_dir))

        # Load evaluation results
        eval_path = Path(model_dir) / "evaluation_results.joblib"
        if eval_path.exists():
            pipeline.evaluator = ModelEvaluator.load(eval_path)
            pipeline.results["evaluation"] = pipeline.evaluator.results

        logger.info(f"Pipeline loaded from {model_dir}")
        return pipeline
