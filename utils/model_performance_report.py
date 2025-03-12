## Build model performance report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
import sys
sys.path.append('../utils ')
from utils.eval_plots import EvalPlots
import pandas as pd

class ModelPerformanceReport(EvalPlots):
    def __init__(self, train_X,train_y, holdout_X,holdout_y,oot_X, oot_y):
        self.train_X = train_X
        self.train_y = train_y
        self.holdout_X = holdout_X
        self.holdout_y = holdout_y
        self.oot_X = oot_X
        self.oot_y = oot_y
        super().__init__()

    def predictions(self, model):
        y_train_pred = model.predict(self.train_X)
        y_train_true = self.train_y
        y_holdout_pred = model.predict(self.holdout_X)
        y_holdout_true = self.holdout_y
        y_oot_true = self.oot_y
        y_oot_pred = model.predict(self.oot_X[self.train_X.columns])

        return y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_pred, y_oot_true

    def produce_report(self, model): 
        
        y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_pred, y_oot_true = self.predictions(model)

        # confusion_matrix(y_train_true, y_train_pred)


        results_df = pd.DataFrame()
        results_df['train'] = [accuracy_score(y_train_true, y_train_pred), precision_score(y_train_true, y_train_pred), recall_score(y_train_true, y_train_pred), f1_score(y_train_true, y_train_pred)]
        results_df['holdout'] = [accuracy_score(y_holdout_true, y_holdout_pred), precision_score(y_holdout_true, y_holdout_pred), recall_score(y_holdout_true, y_holdout_pred), f1_score(y_holdout_true, y_holdout_pred)]   
        results_df['oot'] = [accuracy_score(y_oot_true, y_oot_pred), precision_score(y_oot_true, y_oot_pred), recall_score(y_oot_true, y_oot_pred), f1_score(y_oot_true, y_oot_pred)]
        results_df.index = ['accuracy', 'precision', 'recall', 'f1']
        return results_df
    

    def proba_predictions(self, model):
        y_train_pred = model.predict_proba(self.train_X)[:, 1]
        y_train_true = self.train_y
        y_holdout_pred = model.predict_proba(self.holdout_X)[:, 1]
        y_holdout_true = self.holdout_y
        y_oot_true = self.oot_y
        y_oot_pred = model.predict_proba(self.oot_X[self.train_X.columns])[:, 1]

        return y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_true, y_oot_pred
    
    def produce_proba_report(self, model):
        y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred = self.proba_predictions(model)
        return self.plot_eval_pred_dist(y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred)

    def precision_recall_calc(self, y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred):
        precision_train, recall_train, _ = precision_recall_curve(y_train_true, y_train_pred)
        precision_holdout, recall_holdout, _ = precision_recall_curve(y_holdout_true, y_holdout_pred)
        precision_oot, recall_oot, _ = precision_recall_curve(y_oot_true, y_oot_pred)
        return precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot

    def produce_pr_auc_report(self, model):
        y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_true, y_oot_pred = self.proba_predictions(model)
        precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot = self.precision_recall_calc(y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred)
        return self.plot_eval_pr_auc(precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot) 
