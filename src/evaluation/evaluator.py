"""
Model evaluation module for the Food Safety ML Pipeline.
Provides comprehensive evaluation metrics, visualizations, and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import joblib
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for food safety classification.
    Generates detailed metrics, visualizations, and reports.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the evaluator.

        Args:
            class_names: Names of classes (default: ['safe', 'unsafe'])
        """
        self.class_names = class_names or ["safe", "unsafe"]
        self.results = {}

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            threshold: Classification threshold

        Returns:
            Dictionary of all evaluation metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Convert to binary if needed
        if y_true.dtype == object or isinstance(y_true[0], str):
            label_map = {label: i for i, label in enumerate(self.class_names)}
            y_true_binary = np.array([label_map.get(label, 0) for label in y_true])
        else:
            y_true_binary = y_true

        if y_pred.dtype == object or isinstance(y_pred[0], str):
            label_map = {label: i for i, label in enumerate(self.class_names)}
            y_pred_binary = np.array([label_map.get(label, 0) for label in y_pred])
        else:
            y_pred_binary = y_pred

        # Apply custom threshold if probabilities provided
        if y_proba is not None and threshold != 0.5:
            y_pred_binary = (y_proba >= threshold).astype(int)

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true_binary, y_pred_binary),
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "matthews_corrcoef": matthews_corrcoef(y_true_binary, y_pred_binary),
            "cohen_kappa": cohen_kappa_score(y_true_binary, y_pred_binary),
        }

        # ROC AUC if probabilities provided
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true_binary, y_proba)

            # Calculate ROC curve points
            fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_proba)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            }

            # Calculate Precision-Recall curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                y_true_binary, y_proba
            )
            metrics["pr_curve"] = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }

            # Calculate optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            metrics["optimal_threshold"] = float(roc_thresholds[optimal_idx])

        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["confusion_matrix_labels"] = self.class_names

        # Per-class metrics
        report = classification_report(
            y_true_binary, y_pred_binary,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        metrics["per_class_metrics"] = report

        # Store results
        self.results = metrics

        return metrics

    def find_optimal_threshold(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_proba: np.ndarray,
        metric: str = "f1",
    ) -> Dict[str, float]:
        """
        Find optimal classification threshold for a given metric.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')

        Returns:
            Dictionary with optimal threshold and metric value
        """
        y_true = np.asarray(y_true)
        if y_true.dtype == object or isinstance(y_true[0], str):
            label_map = {label: i for i, label in enumerate(self.class_names)}
            y_true = np.array([label_map.get(label, 0) for label in y_true])

        thresholds = np.linspace(0.01, 0.99, 100)
        scores = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "youden":
                # Youden's J statistic
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = tpr + tnr - 1
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            scores.append(score)

        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        best_score = scores[best_idx]

        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} (score: {best_score:.4f})")

        return {
            "optimal_threshold": float(optimal_threshold),
            "best_score": float(best_score),
            "metric": metric,
        }

    def analyze_errors(
        self,
        X: pd.DataFrame,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_proba: Optional[np.ndarray] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Analyze misclassified samples.

        Args:
            X: Feature DataFrame
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            top_n: Number of top errors to return

        Returns:
            DataFrame with misclassified samples analysis
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            logger.info("No misclassified samples found!")
            return pd.DataFrame()

        # Create error analysis DataFrame
        error_df = X.iloc[misclassified_indices].copy()
        error_df["true_label"] = y_true[misclassified_indices]
        error_df["predicted_label"] = y_pred[misclassified_indices]

        if y_proba is not None:
            error_df["prediction_confidence"] = y_proba[misclassified_indices]
            # Sort by confidence (lowest first = most confident wrong predictions)
            error_df = error_df.sort_values("prediction_confidence")

        # Add error type
        error_df["error_type"] = error_df.apply(
            lambda row: "false_positive" if row["predicted_label"] == 1 else "false_negative",
            axis=1
        )

        logger.info(f"Found {len(error_df)} misclassified samples out of {len(y_true)}")

        # Summary statistics
        error_summary = {
            "total_samples": len(y_true),
            "misclassified": len(error_df),
            "error_rate": len(error_df) / len(y_true),
            "false_positives": (error_df["error_type"] == "false_positive").sum(),
            "false_negatives": (error_df["error_type"] == "false_negative").sum(),
        }

        logger.info(f"Error summary: {error_summary}")

        return error_df.head(top_n)

    def generate_report(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate human-readable evaluation report.

        Args:
            metrics: Metrics dictionary (uses self.results if None)
            save_path: Path to save report (optional)

        Returns:
            Report string
        """
        if metrics is None:
            metrics = self.results

        if not metrics:
            return "No evaluation results available."

        report_lines = [
            "=" * 60,
            "FOOD SAFETY MODEL EVALUATION REPORT",
            "=" * 60,
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"Accuracy:           {metrics.get('accuracy', 'N/A'):.4f}",
            f"Precision:          {metrics.get('precision', 'N/A'):.4f}",
            f"Recall:             {metrics.get('recall', 'N/A'):.4f}",
            f"F1 Score:           {metrics.get('f1_score', 'N/A'):.4f}",
            f"ROC AUC:            {metrics.get('roc_auc', 'N/A'):.4f}",
            f"Matthews Corr:      {metrics.get('matthews_corrcoef', 'N/A'):.4f}",
            f"Cohen's Kappa:      {metrics.get('cohen_kappa', 'N/A'):.4f}",
            "",
        ]

        if "optimal_threshold" in metrics:
            report_lines.append(
                f"Optimal Threshold:  {metrics['optimal_threshold']:.4f}"
            )
            report_lines.append("")

        # Confusion Matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            report_lines.extend([
                "CONFUSION MATRIX",
                "-" * 40,
                f"                    Predicted",
                f"                    {self.class_names[0]:>10}  {self.class_names[1]:>10}",
                f"Actual {self.class_names[0]:>8}     {cm[0][0]:>10}  {cm[0][1]:>10}",
                f"       {self.class_names[1]:>8}     {cm[1][0]:>10}  {cm[1][1]:>10}",
                "",
            ])

        # Per-class metrics
        if "per_class_metrics" in metrics:
            report_lines.extend([
                "PER-CLASS METRICS",
                "-" * 40,
            ])
            for class_name in self.class_names:
                if class_name in metrics["per_class_metrics"]:
                    class_metrics = metrics["per_class_metrics"][class_name]
                    report_lines.append(f"{class_name}:")
                    report_lines.append(
                        f"  Precision: {class_metrics.get('precision', 'N/A'):.4f}, "
                        f"Recall: {class_metrics.get('recall', 'N/A'):.4f}, "
                        f"F1: {class_metrics.get('f1-score', 'N/A'):.4f}"
                    )
            report_lines.append("")

        report_lines.extend([
            "=" * 60,
        ])

        report = "\n".join(report_lines)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")

        return report

    def save(self, output_path: Union[str, Path]):
        """
        Save evaluation results to disk.

        Args:
            output_path: Path to save results
        """
        if not self.results:
            raise RuntimeError("No results to save")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.results, output_path)
        logger.info(f"Evaluation results saved to {output_path}")

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "ModelEvaluator":
        """
        Load evaluation results from disk.

        Args:
            input_path: Path to results file

        Returns:
            ModelEvaluator instance with loaded results
        """
        input_path = Path(input_path)
        results = joblib.load(input_path)

        instance = cls()
        instance.results = results

        logger.info(f"Evaluation results loaded from {input_path}")

        return instance
