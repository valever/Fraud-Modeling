# Fraud Detection Model Notebooks

This directory contains a series of Jupyter notebooks that implement a fraud detection model using machine learning techniques. The notebooks are organized in a logical sequence that covers data preparation, model training, and evaluation.

## Directory Structure

```
notebooks/
├── data_preparation/
│   ├── 01_data_loading.ipynb        # Initial data loading and exploration
│   └── 02_feature_engineering.ipynb  # Feature engineering and preprocessing
├── utils/
│   └── custom_classes.ipynb         # Custom implementations of classes and utilities
├── model_training/
│   ├── 01_data_splitting.ipynb      # Data splitting strategy implementation
│   └── 02_model_training.ipynb      # Model training with cross-validation
└── evaluation/
    └── 01_model_evaluation.ipynb    # Model evaluation and performance analysis
```

## Notebook Descriptions

### Data Preparation
- `01_data_loading.ipynb`: Handles initial data loading, basic exploration, and preliminary analysis of the dataset.
- `02_feature_engineering.ipynb`: Implements feature engineering steps, including encoding categorical variables and scaling numerical features.

### Utils
- `custom_classes.ipynb`: Contains custom implementations of classes used across the project, including:
  - CustomStratifiedKFold: Enhanced k-fold cross-validation with sampling support
  - MultiColumnEncoder: Efficient categorical variable encoder

### Model Training
- `01_data_splitting.ipynb`: Implements the data splitting strategy, including:
  - Out-of-time (OOT) split
  - Train/validation split
  - Sampling approaches
- `02_model_training.ipynb`: Handles model training with:
  - Cross-validation implementation
  - Different sampling techniques
  - Model parameter configuration

### Evaluation
- `01_model_evaluation.ipynb`: Provides comprehensive model evaluation, including:
  - Performance metrics analysis
  - Comparison of different sampling approaches
  - Visualization of results

## Usage

The notebooks should be executed in the following order:

1. Data Preparation notebooks
2. Utils notebook (as needed)
3. Model Training notebooks
4. Evaluation notebook

Each notebook is self-contained but depends on the outputs of previous notebooks in the sequence.

## Requirements

The notebooks require the following main dependencies:
- pandas
- numpy
- scikit-learn
- lightgbm
- imbalanced-learn
- seaborn
- matplotlib
- plotly 