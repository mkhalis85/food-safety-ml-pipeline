#!/usr/bin/env python3
"""
Main entry point for the Food Safety ML Pipeline.
Run this script to train a model and evaluate performance.

Usage:
    python run_pipeline.py [--config CONFIG] [--model MODEL_TYPE] [--no-tuning] [--samples N]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logger, load_config
from src.preprocessing import FoodSafetyPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Food Safety ML Pipeline - Train and evaluate food safety classification models"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to input data CSV file (optional, will generate synthetic if not provided)"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest", "logistic_regression"],
        help="Model type to train"
    )

    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning"
    )

    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=30,
        help="Number of hyperparameter tuning trials"
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Output directory for models and artifacts"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logger(
        name="food_safety_pipeline",
        log_level=args.log_level,
        log_dir="logs/",
    )

    logger.info("=" * 70)
    logger.info("FOOD SAFETY ML PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model type: {args.model}")
    logger.info(f"Hyperparameter tuning: {'enabled' if not args.no_tuning else 'disabled'}")
    logger.info(f"Output directory: {args.output}")

    # Check config file exists
    config_path = args.config if Path(args.config).exists() else None
    if config_path is None:
        logger.warning(f"Config file {args.config} not found, using defaults")

    try:
        # Initialize pipeline
        pipeline = FoodSafetyPipeline(
            config_path=config_path,
            output_dir=args.output,
        )

        # Run full pipeline
        results = pipeline.run_full_pipeline(
            data_path=args.data,
            model_type=args.model,
            use_tuning=not args.no_tuning,
            n_trials=args.trials,
        )

        # Print summary
        logger.info("")
        logger.info("PIPELINE RESULTS SUMMARY")
        logger.info("-" * 40)

        if "evaluation" in results:
            eval_metrics = results["evaluation"]
            logger.info(f"Accuracy:  {eval_metrics.get('accuracy', 0):.4f}")
            logger.info(f"Precision: {eval_metrics.get('precision', 0):.4f}")
            logger.info(f"Recall:    {eval_metrics.get('recall', 0):.4f}")
            logger.info(f"F1 Score:  {eval_metrics.get('f1_score', 0):.4f}")
            logger.info(f"ROC AUC:   {eval_metrics.get('roc_auc', 0):.4f}")

        if "pipeline_summary" in results:
            summary = results["pipeline_summary"]
            logger.info(f"Samples processed: {summary['n_samples']}")
            logger.info(f"Total duration: {summary['duration_seconds']:.2f} seconds")

        logger.info("")
        logger.info(f"Models and artifacts saved to: {args.output}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
