"""
Utility module for creating and running a complete model pipeline.

This module provides a pipeline for preparing data, encoding categorical variables,
and training a fraud detection model with undersampling.
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import sys
from sklearn.model_selection import train_test_split
sys.path.append('../../')
from utils.multicolumn_encoder import MultiColumnEncoder

class ModelPipeline:
    """Pipeline for preparing and training a fraud detection model.
    
    This class provides a complete pipeline for:
    - Splitting data into train, holdout, and OOT sets
    - Encoding categorical variables
    - Training a model with undersampling
    
    The pipeline uses LightGBM as the base model and includes RandomUnderSampler
    to handle class imbalance.
    
    Attributes:
        data (pd.DataFrame): Input dataset containing transaction data
        metadata_columns (list): List of metadata columns to exclude from training
        oot_X (pd.DataFrame): Out-of-time test features
        oot_y (pd.Series): Out-of-time test labels
        holdout_X (pd.DataFrame): Holdout validation features
        holdout_y (pd.Series): Holdout validation labels
        train_X (pd.DataFrame): Training features
        train_y (pd.Series): Training labels
        model_score: Trained model pipeline
    """

    def __init__(self, data: pd.DataFrame, metadata_columns: list) -> None:
        """
        Initialize the ModelPipeline.
        
        Args:
            data (pd.DataFrame): Input dataset containing transaction data
            metadata_columns (list): List of metadata columns to exclude from training
        """
        self.data = data
        self.metadata_columns = metadata_columns
        self.oot_X = pd.DataFrame()
        self.oot_y = pd.Series()
        self.holdout_X = pd.DataFrame()
        self.holdout_y = pd.Series()
        self.train_X = pd.DataFrame()
        self.train_y = pd.Series()

    def prepare_dataset(self):
        """
        Prepare and split the dataset into train, holdout, and OOT sets.
        
        This method:
        1. Splits data into training and OOT sets based on transaction date
        2. Further splits training data into train and holdout sets
        3. Removes metadata columns and age_at_purchase from features
        4. Stores all splits as class attributes
        
        The split is based on the transaction date, with data before July 2020
        used for training and data after that used for OOT testing.
        """
        _expression = "trans_date_trans_time < '2020-07-01 00:00:00'"
        print(f"Splitting dataframe based on expression {_expression!r}.")
        self.data.index = self.data.trans_num
        train = self.data.query(_expression)
        oot = self.data.query(f"~({_expression})")
        print(f"Split dataframe into two dataframes with shapes {train.shape} and {oot.shape}.")

        oot_y = oot.is_fraud
        oot_X = oot.drop(columns=['is_fraud'])
        self.oot_y = oot_y
        self.oot_X = oot_X

        # train test split
        X = train.drop(columns=['is_fraud'])
        X.drop(self.metadata_columns, axis=1, inplace=True)
        y = train['is_fraud']

        train_X, holdout_X, train_y, holdout_y = train_test_split(X, y, test_size=0.2, random_state=42)
        train_X.drop(columns=['age_at_purchase'], inplace=True)
        holdout_X.drop(columns=['age_at_purchase'], inplace=True)

        holdout_y = train.is_fraud
        holdout_X = train.drop(columns=['is_fraud'])
        self.holdout_y = holdout_y
        self.holdout_X = holdout_X
        self.train_X = train_X
        self.train_y = train_y

    def encode_categorical_variables(self):
        """
        Encode categorical variables in all datasets.
        
        This method:
        1. Identifies categorical columns in the training data
        2. Creates and fits a MultiColumnEncoder on training data
        3. Applies the fitted encoder to holdout and OOT sets
        """
        categorical_columns = self.train_X.select_dtypes(include=['object']).columns.tolist()
        encoder = MultiColumnEncoder(categorical_columns)
        self.train_X = encoder.fit_transform(self.train_X)
        self.holdout_X = encoder.transform(self.holdout_X)
        self.oot_X = encoder.transform(self.oot_X)

    def run_model(self):
        """
        Train the model pipeline with undersampling.
        
        This method:
        1. Creates a pipeline with RandomUnderSampler and LightGBM
        2. Fits the pipeline on training data
        3. Stores the trained model as a class attribute
        
        The undersampling strategy is set to 0.2 (20% of majority class) to handle
        class imbalance in the fraud detection task.
        """
        undersample_pipe = make_pipeline(RandomUnderSampler(sampling_strategy=0.2, random_state=42)
                                        ,LGBMClassifier(objective='binary'))
        self.model_score = undersample_pipe.fit(self.train_X, self.train_y
                                            , lgbmclassifier__eval_metric='average_precision'
                                            )
        
    def main(self):
        """
        Run the complete model pipeline.
        
        This method executes all pipeline steps in sequence:
        1. Prepare and split the dataset
        2. Encode categorical variables
        3. Train the model
        
        Returns:
            tuple: Contains:
                - model_score: Trained model pipeline
                - train_X: Training features
                - train_y: Training labels
                - holdout_X: Holdout validation features
                - holdout_y: Holdout validation labels
                - oot_X: Out-of-time test features
                - oot_y: Out-of-time test labels
        """
        self.prepare_dataset()
        self.encode_categorical_variables()
        self.run_model()
        return self.model_score, self.train_X, self.train_y, self.holdout_X, self.holdout_y, self.oot_X, self.oot_y
