"""
Inference module for the Food Safety ML Pipeline.
Provides production-ready prediction capabilities with batch processing support.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FoodSafetyPredictor:
    """
    Production-ready predictor for food safety classification.
    Handles model loading, preprocessing, and batch predictions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model file
            preprocessor_path: Path to saved preprocessor directory
            threshold: Classification threshold
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.threshold = threshold

        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self._loaded = False

        # Load if paths provided
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)

    def load_model(self, model_path: Union[str, Path]) -> "FoodSafetyPredictor":
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Self for method chaining
        """
        model_path = Path(model_path)
        model_data = joblib.load(model_path)

        if isinstance(model_data, dict):
            self.model = model_data.get("model")
            self.model_metadata = {
                "model_type": model_data.get("model_type"),
                "feature_names": model_data.get("feature_names"),
            }
        else:
            self.model = model_data
            self.model_metadata = {}

        self.model_path = str(model_path)
        logger.info(f"Model loaded from {model_path}")

        return self

    def load_preprocessor(self, preprocessor_path: Union[str, Path]) -> "FoodSafetyPredictor":
        """
        Load preprocessor from disk.

        Args:
            preprocessor_path: Path to preprocessor directory

        Returns:
            Self for method chaining
        """
        from src.data_ingestion import DataPreprocessor

        preprocessor_path = Path(preprocessor_path)
        self.preprocessor = DataPreprocessor.load(str(preprocessor_path))
        self.preprocessor_path = str(preprocessor_path)

        logger.info(f"Preprocessor loaded from {preprocessor_path}")

        return self

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame has required columns.

        Args:
            df: Input DataFrame

        Returns:
            True if valid

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            "temperature", "ph_level", "moisture_content",
            "total_plate_count", "coliform_count", "storage_time_days",
            "water_activity", "food_type", "packaging_type", "supplier_id"
        ]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return True

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_proba: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Make prediction(s) on input data.

        Args:
            data: Input data (DataFrame or dictionary for single sample)
            return_proba: Whether to return probabilities

        Returns:
            Prediction(s) with optional probabilities
        """
        if not self._is_loaded and self.model is None:
            raise RuntimeError("Model must be loaded before predicting")

        # Handle single sample as dictionary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
            single_sample = True
        else:
            df = data.copy()
            single_sample = False

        # Validate input
        self._validate_input(df)

        # Handle missing target column
        if "safety_status" in df.columns:
            df_for_pred = df.drop(columns=["safety_status"])
        else:
            df_for_pred = df

        # Transform features if preprocessor available
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(df_for_pred)
        else:
            # Simple fallback: just use numeric columns
            numeric_cols = df_for_pred.select_dtypes(include=[np.number]).columns
            X_transformed = df_for_pred[numeric_cols].values

        # Make predictions
        y_pred = self.model.predict(X_transformed)
        y_proba = self.model.predict_proba(X_transformed)

        # Convert predictions to labels
        label_map = {0: "safe", 1: "unsafe"}
        predictions = [label_map.get(p, "unknown") for p in y_pred]

        if single_sample:
            result = {"prediction": predictions[0]}
            if return_proba:
                result["probability"] = {
                    "safe": float(y_proba[0][0]),
                    "unsafe": float(y_proba[0][1]),
                }
                result["confidence"] = float(max(y_proba[0]))
            return result
        else:
            result_df = pd.DataFrame({
                "prediction": predictions,
            })

            if return_proba:
                result_df["prob_safe"] = y_proba[:, 0]
                result_df["prob_unsafe"] = y_proba[:, 1]
                result_df["confidence"] = np.max(y_proba, axis=1)

            return result_df

    def predict_batch(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 1000,
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """
        Process a batch of samples from file.

        Args:
            input_path: Path to input CSV file
            output_path: Path to save results (optional)
            batch_size: Batch size for processing
            return_proba: Whether to include probabilities

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading batch data from {input_path}")

        # Load data in chunks
        chunks = pd.read_csv(input_path, chunksize=batch_size)
        all_results = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing batch {i + 1} ({len(chunk)} samples)")

            try:
                results = self.predict(chunk, return_proba=return_proba)
                results["batch_id"] = i + 1
                results["sample_id"] = range(i * batch_size, i * batch_size + len(chunk))
                all_results.append(results)

            except Exception as e:
                logger.error(f"Error processing batch {i + 1}: {e}")
                continue

        if not all_results:
            raise RuntimeError("No batches were successfully processed")

        # Combine results
        results_df = pd.concat(all_results, ignore_index=True)

        # Add metadata
        results_df["prediction_timestamp"] = datetime.now().isoformat()
        results_df["model_path"] = self.model_path

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        return results_df

    def explain_prediction(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        n_features: int = 5,
    ) -> Dict[str, Any]:
        """
        Provide simple feature-based explanation for prediction.

        Args:
            data: Input data
            n_features: Number of top features to show

        Returns:
            Dictionary with prediction and explanation
        """
        # Get prediction
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        prediction_result = self.predict(df, return_proba=True)

        if isinstance(prediction_result, dict):
            prediction = prediction_result["prediction"]
            confidence = prediction_result.get("confidence", 0)
        else:
            prediction = prediction_result["prediction"].iloc[0]
            confidence = prediction_result["confidence"].iloc[0]

        # Simple rule-based explanation
        explanations = []

        if isinstance(data, dict):
            sample = data
        else:
            sample = df.iloc[0].to_dict()

        # Check risk factors
        if sample.get("temperature", 0) > 10:
            explanations.append(f"High temperature ({sample['temperature']}°C) increases contamination risk")

        if sample.get("storage_time_days", 0) > 30:
            explanations.append(f"Extended storage time ({sample['storage_time_days']} days)")

        if sample.get("total_plate_count", 0) > 1e5:
            explanations.append(f"Elevated microbial count ({sample['total_plate_count']:.0f} CFU/g)")

        if sample.get("coliform_count", 0) > 100:
            explanations.append(f"High coliform count ({sample['coliform_count']:.0f} CFU/g)")

        if sample.get("water_activity", 0) > 0.85:
            explanations.append(f"High water activity ({sample['water_activity']:.2f}) supports microbial growth")

        if sample.get("ph_level", 7) > 7:
            explanations.append(f"Alkaline pH ({sample['ph_level']}) may indicate spoilage")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_factors": explanations[:n_features],
            "recommendation": "REJECT - Unsafe for consumption" if prediction == "unsafe" else "ACCEPT - Safe for consumption",
        }

    @property
    def _is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "preprocessor_path": self.preprocessor_path,
            "threshold": self.threshold,
            "is_loaded": self._is_loaded,
            "metadata": self.model_metadata or {},
        }
