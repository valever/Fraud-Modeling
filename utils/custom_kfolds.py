import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

class CustomStratifiedKFold:
    """Custom implementation of StratifiedKFold with optional undersampling support.
    
    This class extends the functionality of sklearn's StratifiedKFold by adding
    support for undersampling within each fold during cross-validation.
    """
    
    def __init__(self, n_splits=5, undersample_func=None, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.undersample_func = undersample_func
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=self.random_state)

    def split(self, dataframe, y):
        """Generate indices to split data into training and validation sets.
        
        Args:
            dataframe: Features dataframe
            y: Target variable
            
        Returns:
            List of tuples containing train and validation data for each fold
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