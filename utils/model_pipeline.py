import pandas as pd
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import sys
sys.path.append('../../')
from utils.multicolumn_encoder import MultiColumnEncoder

class ModelPipeline():
    def __init__(self, data, metadata_columns) -> None:
        self.data = data
        self.metadata_columns = metadata_columns
        self.oot_X = pd.DataFrame()
        self.oot_y = pd.Series()
        self.holdout_X = pd.DataFrame()
        self.holdout_y = pd.Series()
        self.train_X = pd.DataFrame()
        self.train_y = pd.Series()

    def prepare_dataset(self):
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
        categorical_columns = self.train_X.select_dtypes(include=['object']).columns.tolist()
        encoder = MultiColumnEncoder(categorical_columns)
        self.train_X = encoder.fit_transform(self.train_X)
        self.holdout_X = encoder.transform(self.holdout_X)
        self.oot_X = encoder.transform(self.oot_X)

    def run_model(self):
        undersample_pipe = make_pipeline(RandomUnderSampler(sampling_strategy=0.2, random_state=42)
                                    ,LGBMClassifier(objective='binary'))
        self.score_balanced_model = undersample_pipe.fit(self.train_X, self.train_y
                                            , lgbmclassifier__eval_metric='average_precision'
                                            )
        
    def main(self):
        self.prepare_dataset()
        self.encode_categorical_variables()
        self.run_model()
        return self.score_balanced_model, self.train_X, self.train_y, self.holdout_X, self.holdout_y, self.oot_X, self.oot_y
