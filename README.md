# food-safety-ml-pipeline

A complete, production-ready machine learning pipeline for food safety classification. This system predicts whether food samples/batches are safe or unsafe (contaminated) based on tabular features such as temperature, pH, moisture, microbial counts, storage time, and more.

## Features

- **Data Generation**: Synthetic data generator with realistic correlations between features and contamination risk
- **Data Validation**: Comprehensive schema validation, range checking, and data quality assessment using Pydantic
- **Preprocessing**: Automated handling of missing values, feature scaling, and categorical encoding
- **Multiple Model Support**: XGBoost, LightGBM, Random Forest, and Logistic Regression
- **Hyperparameter Tuning**: Bayesian optimization using Optuna
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1, ROC AUC, confusion matrix, and more
- **Production Inference**: Batch processing, single-sample prediction, and explainable AI features
- **Configuration Management**: YAML-based configuration for easy customization
- **Logging**: Structured logging with file and console output

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Full Pipeline

```bash
# Run with default settings (generates synthetic data)
python run_pipeline.py

# Use specific model type
python run_pipeline.py --model xgboost

# Disable hyperparameter tuning for faster execution
python run_pipeline.py --no-tuning

# Use your own data
python run_pipeline.py --data path/to/your/data.csv
```

### Programmatic Usage

```python
from src.preprocessing import FoodSafetyPipeline

# Initialize and run pipeline
pipeline = FoodSafetyPipeline(output_dir='models')
results = pipeline.run_full_pipeline(
    model_type='xgboost',
    use_tuning=True,
    n_trials=50,
)

# Access results
print(f"Accuracy: {results['evaluation']['accuracy']:.4f}")
print(f"F1 Score: {results['evaluation']['f1_score']:.4f}")
```

### Inference/Prediction

```python
from src.inference import FoodSafetyPredictor

# Load trained model
predictor = FoodSafetyPredictor(
    model_path='models/best_model.joblib',
    preprocessor_path='models/preprocessor',
)

# Single sample prediction
sample = {
    'temperature': 4.0,
    'ph_level': 6.5,
    'moisture_content': 80,
    'total_plate_count': 1000,
    'coliform_count': 10,
    'storage_time_days': 5,
    'water_activity': 0.85,
    'food_type': 'dairy',
    'packaging_type': 'refrigerated',
    'supplier_id': 'SUP001',
}

result = predictor.predict(sample, return_proba=True)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Get explanation
explanation = predictor.explain_prediction(sample)
print(f"Recommendation: {explanation['recommendation']}")
```

## Project Structure

```
food-safety-ml-pipeline/
├── config/
│   └── config.yaml          # Pipeline configuration
├── data/
│   ├── raw/                 # Raw input data
│   └── processed/           # Processed data
├── models/                  # Trained models and artifacts
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── data_ingestion/      # Data loading and preprocessing
│   ├── model_training/      # Model training and tuning
│   ├── evaluation/          # Model evaluation metrics
│   ├── inference/           # Production inference
│   ├── preprocessing/       # Main pipeline orchestration
│   └── utils/               # Utilities (logging, config, validation)
├── tests/                   # Unit and integration tests
├── logs/                    # Log files
├── requirements.txt         # Python dependencies
└── run_pipeline.py          # Main entry point
```

## Configuration

Edit `config/config.yaml` to customize:

- Feature columns (numeric and categorical)
- Model hyperparameters
- Train/validation/test split ratios
- Evaluation metrics
- Logging settings

## Features

The pipeline uses the following features for prediction:

| Feature | Type | Description |
|---------|------|-------------|
| temperature | Numeric | Storage temperature (°C) |
| ph_level | Numeric | pH level of the sample |
| moisture_content | Numeric | Moisture percentage |
| total_plate_count | Numeric | Total microbial count (CFU/g) |
| coliform_count | Numeric | Coliform bacteria count (CFU/g) |
| storage_time_days | Numeric | Days in storage |
| water_activity | Numeric | Water activity (aw) |
| food_type | Categorical | Type of food product |
| packaging_type | Categorical | Packaging method |
| supplier_id | Categorical | Supplier identifier |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src
```

## License

MIT License
