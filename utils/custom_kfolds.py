"""
Utility module for custom cross-validation with undersampling support.

This module provides a custom implementation of stratified k-fold cross-validation
that supports undersampling within each fold during the training process.
"""

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd
class CustomStratifiedKFold:
    """Custom implementation of StratifiedKFold with optional undersampling support.
    
    This class extends the functionality of sklearn's StratifiedKFold by adding
    undersampling within each fold during cross-validation. This is particularly
    useful for imbalanced datasets where you want to maintain class distribution
    in the validation set while undersampling the training data.
    
    Attributes:
        n_splits (int): Number of folds for cross-validation
        undersample_func: Function or object implementing fit_resample for undersampling
        random_state (int): Random seed for reproducibility
        skf (StratifiedKFold): Base sklearn StratifiedKFold instance
    """
    
    def __init__(self, n_splits=5, undersample_func=None, shuffle=True, random_state=42):
        """
        Initialize the CustomStratifiedKFold class.
        
        Args:
            n_splits (int, optional): Number of folds. Defaults to 5.
            undersample_func (callable, optional): Function for undersampling training data.
                Must implement fit_resample method. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data before splitting.
                Defaults to True.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.
        """
        self.n_splits = n_splits
        self.undersample_func = undersample_func
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=self.random_state)

    def split(self, dataframe: pd.DataFrame, y: pd.Series) -> list[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]]:
        """
        Generate indices to split data into training and validation sets.
        
        This method creates k folds of the data, where each fold maintains the
        class distribution in the validation set. If undersampling is specified,
        it is applied only to the training data within each fold.
        
        Args:
            dataframe (pd.DataFrame): Features dataframe containing the input data
            y (pd.Series): Target variable with class labels
            
        Returns:
            list: List of tuples, where each tuple contains:
                - (train_df, y_train): Training data and labels for the fold
                - (test_df, y_test): Validation data and labels for the fold
                
        Example:
            >>> kf = CustomStratifiedKFold(n_splits=5, undersample_func=RandomUnderSampler())
            >>> folds = kf.split(X, y)
            >>> for (train_df, y_train), (test_df, y_test) in folds:
            ...     # Process each fold
        """
        folds = []
        for train_index, test_index in tqdm(self.skf.split(X=dataframe, y=y), desc="Generating K-Folds", total=self.n_splits):
            train_df, y_train = dataframe.iloc[train_index], y.iloc[train_index]
            test_df, y_test = dataframe.iloc[test_index], y[test_index]

            # Apply undersampling only to training data if specified
            if self.undersample_func is not None:
                train_df, y_train = self.undersample_func.fit_resample(train_df, y_train)

            folds.append(((train_df, y_train), (test_df, y_test)))
        return folds