"""
Data loading module for the Food Safety ML Pipeline.
Handles loading data from various sources and generating synthetic data for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal data loader for food safety datasets.
    Supports loading from CSV, Excel, JSON, and database sources.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data loader.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.supported_formats = [".csv", ".xlsx", ".xls", ".json", ".parquet"]

    def load(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from a file.

        Args:
            file_path: Path to the data file
            file_format: File format (auto-detected if not provided)
            **kwargs: Additional arguments passed to the underlying loader

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Auto-detect format if not provided
        if file_format is None:
            file_format = file_path.suffix.lower()
        else:
            file_format = f".{file_format.lstrip('.')}"

        if file_format not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Supported formats: {self.supported_formats}"
            )

        logger.info(f"Loading data from {file_path} (format: {file_format})")

        try:
            if file_format == ".csv":
                df = pd.read_csv(file_path, **kwargs)
            elif file_format in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, **kwargs)
            elif file_format == ".json":
                df = pd.read_json(file_path, **kwargs)
            elif file_format == ".parquet":
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unknown format: {file_format}")

            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def load_from_multiple(
        self,
        file_paths: List[Union[str, Path]],
        combine: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load data from multiple files.

        Args:
            file_paths: List of file paths
            combine: If True, concatenate all DataFrames
            **kwargs: Additional arguments for loading

        Returns:
            Single combined DataFrame or list of DataFrames
        """
        dataframes = []

        for file_path in file_paths:
            try:
                df = self.load(file_path, **kwargs)
                dataframes.append(df)
                logger.info(f"Loaded {file_path}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if combine and dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined dataset: {len(combined_df)} rows")
            return combined_df

        return dataframes

    def save(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        file_format: str = "csv",
        **kwargs,
    ):
        """
        Save DataFrame to a file.

        Args:
            df: DataFrame to save
            output_path: Output file path
            file_format: Output format (csv, excel, json, parquet)
            **kwargs: Additional arguments for saving
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = file_format.lower().lstrip(".")

        logger.info(f"Saving data to {output_path} (format: {file_format})")

        if file_format == "csv":
            df.to_csv(output_path, index=False, **kwargs)
        elif file_format in ["xlsx", "excel"]:
            df.to_excel(output_path, index=False, **kwargs)
        elif file_format == "json":
            df.to_json(output_path, orient="records", **kwargs)
        elif file_format == "parquet":
            df.to_parquet(output_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported output format: {file_format}")

        logger.info(f"Successfully saved {len(df)} rows to {output_path}")


def generate_sample_data(
    n_samples: int = 1000,
    random_state: int = 42,
    contamination_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Generate synthetic food safety data for testing and development.

    This function creates realistic-looking food safety data with appropriate
    correlations between features and the target variable.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        contamination_ratio: Ratio of unsafe/contaminated samples

    Returns:
        DataFrame with synthetic food safety data
    """
    np.random.seed(random_state)

    logger.info(f"Generating {n_samples} synthetic food safety samples")

    # Generate categorical features
    food_types = np.random.choice(
        ["dairy", "meat", "seafood", "produce", "grain", "processed"],
        size=n_samples,
        p=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1],
    )

    packaging_types = np.random.choice(
        ["vacuum", "aerated", "canned", "frozen", "refrigerated", "ambient"],
        size=n_samples,
        p=[0.2, 0.1, 0.15, 0.2, 0.25, 0.1],
    )

    supplier_ids = [f"SUP{i:03d}" for i in range(1, 21)]
    suppliers = np.random.choice(supplier_ids, size=n_samples)

    # Generate numeric features with realistic distributions
    # Temperature: affected by packaging type
    base_temp = np.random.normal(4, 3, n_samples)  # Refrigerated baseline
    temp_adjustments = {
        "frozen": -18,
        "refrigerated": 0,
        "ambient": 15,
        "canned": 10,
        "vacuum": 2,
        "aerated": 5,
    }
    temperature = np.array([
        base_temp[i] + temp_adjustments.get(packaging_types[i], 0)
        for i in range(n_samples)
    ])
    temperature = np.clip(temperature, -20, 30)

    # pH level: varies by food type
    ph_means = {
        "dairy": 6.5,
        "meat": 5.8,
        "seafood": 7.0,
        "produce": 5.5,
        "grain": 6.0,
        "processed": 5.0,
    }
    ph_level = np.array([
        np.random.normal(ph_means.get(ft, 6.0), 0.8)
        for ft in food_types
    ])
    ph_level = np.clip(ph_level, 3.0, 9.0)

    # Moisture content: varies by food type
    moisture_means = {
        "dairy": 85,
        "meat": 70,
        "seafood": 80,
        "produce": 90,
        "grain": 12,
        "processed": 50,
    }
    moisture_content = np.array([
        np.random.normal(moisture_means.get(ft, 50), 10)
        for ft in food_types
    ])
    moisture_content = np.clip(moisture_content, 5, 98)

    # Water activity: correlated with moisture
    water_activity = 0.3 + (moisture_content / 100) * 0.65
    water_activity += np.random.normal(0, 0.05, n_samples)
    water_activity = np.clip(water_activity, 0.3, 0.99)

    # Storage time: varies by packaging
    storage_means = {
        "frozen": 180,
        "refrigerated": 14,
        "ambient": 30,
        "canned": 365,
        "vacuum": 45,
        "aerated": 7,
    }
    storage_time_days = np.array([
        np.random.exponential(storage_means.get(pt, 30))
        for pt in packaging_types
    ])
    storage_time_days = np.clip(storage_time_days, 0, 730).astype(int)

    # Microbial counts: influenced by multiple factors
    # Base microbial growth potential
    growth_potential = (
        (temperature + 5) / 35 * 0.3 +  # Higher temp = more growth
        (water_activity - 0.3) / 0.69 * 0.3 +  # Higher aw = more growth
        (storage_time_days / 730) * 0.4  # Longer storage = more growth
    )
    growth_potential = np.clip(growth_potential, 0, 1)

    # Total plate count (CFU/g)
    total_plate_count = np.random.lognormal(
        mean=np.log(1000) + growth_potential * 5,
        sigma=1.5,
        size=n_samples,
    )
    total_plate_count = np.clip(total_plate_count, 0, 1e8)

    # Coliform count (CFU/g) - correlated with total plate count
    coliform_ratio = np.random.beta(2, 20, n_samples)  # Typically small fraction
    coliform_count = total_plate_count * coliform_ratio
    coliform_count = np.clip(coliform_count, 0, 1e7)

    # Add some noise and outliers
    outlier_mask = np.random.random(n_samples) < 0.02
    total_plate_count[outlier_mask] *= np.random.uniform(5, 20, outlier_mask.sum())
    coliform_count[outlier_mask] *= np.random.uniform(5, 20, outlier_mask.sum())

    # Generate target variable based on safety criteria
    # Unsafe if any of these conditions are met:
    # 1. Total plate count > 1e6 CFU/g
    # 2. Coliform count > 1000 CFU/g
    # 3. Temperature abuse (too high for too long)
    # 4. Combination of factors

    unsafe_conditions = (
        (total_plate_count > 1e6) |
        (coliform_count > 1000) |
        ((temperature > 10) & (storage_time_days > 7)) |
        ((water_activity > 0.85) & (ph_level > 6.5) & (storage_time_days > 14))
    )

    # Add some randomness to make it not perfectly deterministic
    randomness = np.random.random(n_samples) < 0.05
    unsafe_conditions = unsafe_conditions ^ randomness

    # Ensure we have approximately the desired contamination ratio
    current_ratio = unsafe_conditions.sum() / n_samples
    if abs(current_ratio - contamination_ratio) > 0.05:
        # Adjust threshold to get closer to desired ratio
        risk_score = (
            np.log1p(total_plate_count) / 20 +
            np.log1p(coliform_count) / 15 +
            (temperature + 5) / 35 +
            (storage_time_days / 730) +
            (water_activity - 0.3) / 0.69
        )
        threshold = np.percentile(risk_score, (1 - contamination_ratio) * 100)
        unsafe_conditions = risk_score > threshold

    safety_status = np.where(unsafe_conditions, "unsafe", "safe")

    # Create DataFrame
    df = pd.DataFrame({
        "temperature": np.round(temperature, 2),
        "ph_level": np.round(ph_level, 2),
        "moisture_content": np.round(moisture_content, 2),
        "total_plate_count": np.round(total_plate_count, 2),
        "coliform_count": np.round(coliform_count, 2),
        "storage_time_days": storage_time_days,
        "water_activity": np.round(water_activity, 4),
        "food_type": food_types,
        "packaging_type": packaging_types,
        "supplier_id": suppliers,
        "safety_status": safety_status,
    })

    # Add some missing values randomly (about 2-5% per column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_mask = np.random.random(n_samples) < np.random.uniform(0.02, 0.05)
        df.loc[missing_mask, col] = np.nan

    logger.info(f"Generated dataset with {unsafe_conditions.sum()} unsafe samples "
                f"({unsafe_conditions.sum()/n_samples*100:.1f}%)")

    return df
