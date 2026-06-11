"""
Data validation module for the Food Safety ML Pipeline.
Uses Pydantic for schema validation and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class SafetyStatus(str, Enum):
    """Enum for safety status labels."""
    SAFE = "safe"
    UNSAFE = "unsafe"


class FoodType(str, Enum):
    """Common food types in the dataset."""
    DAIRY = "dairy"
    MEAT = "meat"
    SEAFOOD = "seafood"
    PRODUCE = "produce"
    GRAIN = "grain"
    PROCESSED = "processed"
    OTHER = "other"


class PackagingType(str, Enum):
    """Common packaging types."""
    VACUUM = "vacuum"
    AERATED = "aerated"
    CANNED = "canned"
    FROZEN = "frozen"
    REFRESHED = "refrigerated"
    AMBIENT = "ambient"
    OTHER = "other"


class FoodSafetySchema(BaseModel):
    """
    Pydantic schema for validating individual food safety records.
    Defines expected ranges and constraints for each feature.
    """

    # Numeric features with realistic ranges
    temperature: float = Field(
        ge=-20.0, le=100.0,
        description="Storage temperature in Celsius"
    )
    ph_level: float = Field(
        ge=0.0, le=14.0,
        description="pH level of the food sample"
    )
    moisture_content: float = Field(
        ge=0.0, le=100.0,
        description="Moisture content percentage"
    )
    total_plate_count: float = Field(
        ge=0.0,
        description="Total plate count (CFU/g)"
    )
    coliform_count: float = Field(
        ge=0.0,
        description="Coliform count (CFU/g)"
    )
    storage_time_days: int = Field(
        ge=0, le=3650,
        description="Storage time in days"
    )
    water_activity: float = Field(
        ge=0.0, le=1.0,
        description="Water activity (aw)"
    )

    # Categorical features
    food_type: FoodType = Field(description="Type of food product")
    packaging_type: PackagingType = Field(description="Packaging type")
    supplier_id: str = Field(description="Supplier identifier")

    # Target variable
    safety_status: SafetyStatus = Field(description="Safety classification label")

    @field_validator("ph_level")
    @classmethod
    def check_ph_extremes(cls, v):
        """Warn about extreme pH values."""
        if v < 2.0 or v > 12.0:
            pass  # Could log a warning here
        return v

    @field_validator("total_plate_count", "coliform_count")
    @classmethod
    def validate_microbial_counts(cls, v):
        """Ensure microbial counts are non-negative."""
        if v < 0:
            raise ValueError("Microbial counts cannot be negative")
        return v


class DataValidator:
    """
    Comprehensive data validator for food safety datasets.
    Performs schema validation, range checks, and data quality assessments.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.

        Args:
            config: Optional configuration dictionary with validation rules
        """
        self.config = config or {}
        self.validation_errors: List[Dict[str, Any]] = []
        self.validation_warnings: List[Dict[str, Any]] = []

        # Define expected columns
        self.expected_columns = [
            "temperature",
            "ph_level",
            "moisture_content",
            "total_plate_count",
            "coliform_count",
            "storage_time_days",
            "water_activity",
            "food_type",
            "packaging_type",
            "supplier_id",
            "safety_status",
        ]

        # Define valid ranges for numeric features
        self.numeric_ranges = {
            "temperature": (-20.0, 100.0),
            "ph_level": (0.0, 14.0),
            "moisture_content": (0.0, 100.0),
            "total_plate_count": (0.0, float("inf")),
            "coliform_count": (0.0, float("inf")),
            "storage_time_days": (0, 3650),
            "water_activity": (0.0, 1.0),
        }

        # Define valid categorical values
        self.categorical_values = {
            "food_type": ["dairy", "meat", "seafood", "produce", "grain", "processed", "other"],
            "packaging_type": ["vacuum", "aerated", "canned", "frozen", "refrigerated", "ambient", "other"],
            "safety_status": ["safe", "unsafe"],
        }

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the DataFrame has all required columns.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        missing_cols = set(self.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_columns)

        errors = []
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        if extra_cols:
            self.validation_warnings.append({
                "type": "extra_columns",
                "message": f"Extra columns found (will be ignored): {extra_cols}"
            })

        return len(missing_cols) == 0, errors

    def validate_ranges(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        Validate that numeric features are within expected ranges.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (all_valid, DataFrame with out-of-range flags)
        """
        out_of_range = pd.DataFrame(index=df.index)

        for col, (min_val, max_val) in self.numeric_ranges.items():
            if col in df.columns:
                mask = df[col].notna()
                if max_val == float("inf"):
                    out_of_range[col] = (df[col] < min_val) & mask
                else:
                    out_of_range[col] = ((df[col] < min_val) | (df[col] > max_val)) & mask

                if out_of_range[col].any():
                    n_violations = out_of_range[col].sum()
                    self.validation_warnings.append({
                        "type": "range_violation",
                        "column": col,
                        "count": int(n_violations),
                        "expected_range": (min_val, max_val),
                    })

        return not out_of_range.any().any(), out_of_range

    def validate_categorical(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        Validate categorical features have expected values.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (all_valid, dict of invalid values per column)
        """
        invalid_values = {}

        for col, valid_vals in self.categorical_values.items():
            if col in df.columns:
                invalid_mask = ~df[col].str.lower().isin(valid_vals) & df[col].notna()
                if invalid_mask.any():
                    unique_invalid = df.loc[invalid_mask, col].unique().tolist()
                    invalid_values[col] = unique_invalid
                    self.validation_warnings.append({
                        "type": "invalid_categorical",
                        "column": col,
                        "invalid_values": unique_invalid[:10],  # Limit to first 10
                        "count": int(invalid_mask.sum()),
                    })

        return len(invalid_values) == 0, invalid_values

    def check_missing_values(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.

        Args:
            df: Input DataFrame
            threshold: Maximum acceptable missing ratio per column

        Returns:
            Dictionary with missing value statistics
        """
        missing_stats = {}
        total_rows = len(df)

        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / total_rows if total_rows > 0 else 0

            missing_stats[col] = {
                "missing_count": int(missing_count),
                "missing_ratio": float(missing_ratio),
                "acceptable": missing_ratio <= threshold,
            }

            if missing_ratio > threshold:
                self.validation_errors.append({
                    "type": "excessive_missing",
                    "column": col,
                    "missing_ratio": float(missing_ratio),
                    "threshold": threshold,
                })

        return missing_stats

    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for duplicate rows in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with duplicate statistics
        """
        total_duplicates = df.duplicated().sum()
        subset_duplicates = {}

        # Check duplicates on key identifiers
        if "supplier_id" in df.columns:
            subset_duplicates["supplier_id"] = int(
                df.duplicated(subset=["supplier_id"]).sum()
            )

        return {
            "total_duplicates": int(total_duplicates),
            "subset_duplicates": subset_duplicates,
        }

    def validate_full(
        self, df: pd.DataFrame, strict: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the dataset.

        Args:
            df: Input DataFrame
            strict: If True, treat warnings as errors

        Returns:
            Comprehensive validation report
        """
        # Reset validation logs
        self.validation_errors = []
        self.validation_warnings = []

        report = {
            "is_valid": True,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "schema_validation": {},
            "range_validation": {},
            "categorical_validation": {},
            "missing_value_analysis": {},
            "duplicate_analysis": {},
            "errors": [],
            "warnings": [],
        }

        # Schema validation
        schema_valid, schema_errors = self.validate_schema(df)
        report["schema_validation"] = {
            "is_valid": schema_valid,
            "errors": schema_errors,
        }
        if not schema_valid:
            report["errors"].extend(schema_errors)
            report["is_valid"] = False

        # Range validation
        range_valid, out_of_range_df = self.validate_ranges(df)
        report["range_validation"] = {
            "is_valid": range_valid,
            "out_of_range_columns": out_of_range_df.columns[
                out_of_range_df.any()
            ].tolist(),
        }
        if not range_valid and strict:
            report["is_valid"] = False

        # Categorical validation
        cat_valid, invalid_cat_values = self.validate_categorical(df)
        report["categorical_validation"] = {
            "is_valid": cat_valid,
            "invalid_values": invalid_cat_values,
        }
        if not cat_valid and strict:
            report["is_valid"] = False

        # Missing value analysis
        report["missing_value_analysis"] = self.check_missing_values(df)

        # Duplicate analysis
        report["duplicate_analysis"] = self.check_duplicates(df)

        # Add warnings and errors
        report["warnings"] = self.validation_warnings
        report["errors"] = self.validation_errors

        return report

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        fill_missing: str = "median",
    ) -> pd.DataFrame:
        """
        Clean the dataset based on validation results.

        Args:
            df: Input DataFrame
            remove_outliers: Whether to remove rows with out-of-range values
            fill_missing: Strategy for filling missing values

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Remove out-of-range values (set to NaN)
        if remove_outliers:
            for col, (min_val, max_val) in self.numeric_ranges.items():
                if col in df_clean.columns:
                    if max_val == float("inf"):
                        mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                    else:
                        mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                    df_clean.loc[mask, col] = np.nan

        # Fill missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                if fill_missing == "median":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif fill_missing == "mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif fill_missing == "drop":
                    df_clean = df_clean.dropna(subset=[col])

        return df_clean
