"""
Hyperparameter tuning module for the Food Safety ML Pipeline.
Uses Optuna for efficient Bayesian optimization of model hyperparameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import joblib
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter optimization using Optuna.
    Supports multiple model types and custom objective functions.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        cv_folds: int = 5,
        scoring_metric: str = "f1",
        random_state: int = 42,
        direction: str = "maximize",
    ):
        """
        Initialize the hyperparameter tuner.

        Args:
            model_type: Type of model to tune
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (None for no timeout)
            cv_folds: Number of cross-validation folds
            scoring_metric: Metric to optimize
            random_state: Random seed
            direction: Optimization direction ('maximize' or 'minimize')
        """
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.random_state = random_state
        self.direction = direction

        self.best_params = None
        self.best_score = None
        self.study = None
        self.trial_history = []

    def _get_search_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space based on model type."""
        spaces = {
            "xgboost": {
                "n_estimators": (50, 500),
                "max_depth": (3, 12),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "gamma": (0, 10),
                "reg_alpha": (0, 10),
                "reg_lambda": (0, 10),
            },
            "lightgbm": {
                "n_estimators": (50, 500),
                "max_depth": (3, 12),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "min_child_samples": (5, 100),
                "reg_alpha": (0, 10),
                "reg_lambda": (0, 10),
            },
            "random_forest": {
                "n_estimators": (50, 500),
                "max_depth": (5, 30),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
                "max_features": ("sqrt", "log2"),
            },
            "logistic_regression": {
                "C": (0.001, 100),
                "penalty": ("l1", "l2", "elasticnet"),
                "solver": ("liblinear", "saga"),
            },
        }

        return spaces.get(self.model_type, {})

    def _objective(
        self,
        trial,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            X: Training features
            y: Training labels

        Returns:
            Cross-validation score
        """
        params = self._suggest_params(trial)

        # Create model with suggested parameters
        model = self._create_model_with_params(params)

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=self.scoring_metric,
            n_jobs=-1,
        )

        mean_score = scores.mean()

        # Store trial info
        self.trial_history.append({
            "trial_number": trial.number,
            "params": params,
            "score": mean_score,
        })

        return mean_score

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest parameters for current trial."""
        search_space = self._get_search_space()
        params = {}

        for param_name, param_range in search_space.items():
            if isinstance(param_range[0], (int, float)):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, param_range[0], param_range[1]
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], log=True
                    )
            elif isinstance(param_range[0], str):
                params[param_name] = trial.suggest_categorical(param_name, param_range)

        # Add model-specific defaults
        params["random_state"] = self.random_state

        if self.model_type == "xgboost":
            params["use_label_encoder"] = False
            params["eval_metric"] = "logloss"
        elif self.model_type == "lightgbm":
            params["verbose"] = -1
        elif self.model_type == "logistic_regression":
            params["max_iter"] = 1000

        return params

    def _create_model_with_params(self, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training labels
            show_progress: Whether to show optimization progress

        Returns:
            Dictionary with best parameters and score
        """
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise ImportError("Optuna is required for hyperparameter tuning")

        logger.info(f"Starting hyperparameter optimization for {self.model_type}")
        logger.info(f"Running {self.n_trials} trials with {self.cv_folds}-fold CV")
        logger.info(f"Optimizing for {self.scoring_metric} metric")

        # Reset trial history
        self.trial_history = []

        # Create study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress,
        )

        # Extract best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Best {self.scoring_metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.trial_history),
            "model_type": self.model_type,
        }

    def get_trial_history(self) -> pd.DataFrame:
        """
        Get history of all trials as DataFrame.

        Returns:
            DataFrame with trial history
        """
        return pd.DataFrame(self.trial_history)

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history (requires matplotlib).

        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not installed for plotting")
            return

        if not self.trial_history:
            logger.warning("No trial history available")
            return

        df = pd.DataFrame(self.trial_history)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot score over trials
        axes[0].plot(df["trial_number"], df["score"], alpha=0.5)
        axes[0].plot(
            df["trial_number"],
            df["score"].cummax(),
            color="red",
            linewidth=2,
            label="Best so far",
        )
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel(f"{self.scoring_metric} Score")
        axes[0].set_title("Optimization History")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot best score distribution
        axes[1].hist(df["score"], bins=20, edgecolor="black", alpha=0.7)
        axes[1].axvline(
            self.best_score, color="red", linestyle="--",
            linewidth=2, label=f"Best: {self.best_score:.4f}"
        )
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Score Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def save(self, output_path: Union[str, Path]):
        """
        Save tuning results to disk.

        Args:
            output_path: Path to save results
        """
        if self.best_params is None:
            raise RuntimeError("No tuning results to save")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trial_history": self.trial_history,
            "model_type": self.model_type,
            "config": {
                "n_trials": self.n_trials,
                "cv_folds": self.cv_folds,
                "scoring_metric": self.scoring_metric,
                "random_state": self.random_state,
            },
        }

        joblib.dump(results, output_path)
        logger.info(f"Tuning results saved to {output_path}")

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "HyperparameterTuner":
        """
        Load tuning results from disk.

        Args:
            input_path: Path to results file

        Returns:
            HyperparameterTuner instance with loaded results
        """
        input_path = Path(input_path)
        results = joblib.load(input_path)

        instance = cls(
            model_type=results["model_type"],
            n_trials=results["config"]["n_trials"],
            cv_folds=results["config"]["cv_folds"],
            scoring_metric=results["config"]["scoring_metric"],
            random_state=results["config"]["random_state"],
        )

        instance.best_params = results["best_params"]
        instance.best_score = results["best_score"]
        instance.trial_history = results["trial_history"]

        logger.info(f"Tuning results loaded from {input_path}")

        return instance
