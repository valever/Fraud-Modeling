import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EvalPlots():

    def __init__(self):
        pass

    def plot_eval_basic(self, y_true, y_score):

        '''
        y_score = model.predict_proba(X)[:, 1]
        '''

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # The histogram of scores compared to true labels
        fig_hist = px.histogram(
            x=y_score, color=y_true, nbins=50,
            labels=dict(color='True Labels', x='Score')
            , histnorm='probability density'
        )

        fig_hist.show()


        # Evaluating model performance on PR curve

        fig_thresh = px.area(
            x=recall, y=precision,
            title=f'Precision-Recall Curve (AUC={auc(recall, precision):.4f})',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        fig_thresh.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_thresh.update_xaxes(constrain='domain')

        fig_thresh.show()

        return fig_hist, fig_thresh
    
    
    
    def plot_eval_pred_dist(self, y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred):
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Train", "Holdout", "OOT"))
        print('inside plot_eval_pred_dist')

        trace0 = px.histogram(
                    x=y_train_pred, color=y_train_true, nbins=50,
                    histnorm='probability density',
                    labels=dict(color='True Labels', x='Score')
                )
        print(  'trace0')
        trace1 = px.histogram(
                    x=y_holdout_pred, color=y_holdout_true, nbins=50,
                    labels=dict(color='True Labels', x='Score')
                    , histnorm='probability density'
                )
        trace2 = px.histogram(
                    x=y_oot_pred, color=y_oot_true, nbins=50,
                    labels=dict(color='True Labels', x='Score')
                    , histnorm='probability density'
                )

        # add each trace (or traces) to its specific subplot
        pl_nr = 0
        for plot_ in [trace0, trace1, trace2]:
            pl_nr += 1
            for trace in plot_.data:
                fig.add_trace(trace, row=pl_nr, col=1)

        fig.update_layout(title_text="Model Performance", showlegend=True)
        return fig

    def plot_eval_pr_auc(self, precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot):
        # Evaluating model performance on PR curve

        tr_title = f'Train (AUC={auc(recall_train, precision_train):.4f})' 
        ho_title = f'Holdout (AUC={auc(recall_holdout, precision_holdout):.4f})'
        oot_title = f'OOT (AUC={auc(recall_oot, precision_oot):.4f})'

        fig = make_subplots(rows=1, cols=3, subplot_titles=(tr_title, ho_title, oot_title)) 

        trace0 = px.area(
            x=recall_train, y=precision_train,
            title=f'Training (AUC={auc(recall_train, precision_train):.4f})',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        trace0.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        trace0.update_yaxes(scaleanchor="x", scaleratio=1)
        trace0.update_xaxes(constrain='domain')

        trace1 = px.area(
            x=recall_holdout, y=precision_holdout,
            title=f'Holdout (AUC={auc(recall_holdout, precision_holdout):.4f})',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        trace1.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        trace1.update_yaxes(scaleanchor="x", scaleratio=1)
        trace1.update_xaxes(constrain='domain')

        trace2 = px.area(
            x=recall_oot, y=precision_oot,
            title=f'OOT AUC={auc(recall_oot, precision_oot):.4f})',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        trace2.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        trace2.update_yaxes(scaleanchor="x", scaleratio=1)
        trace2.update_xaxes(constrain='domain')

        pl_nr = 0
        for plot_ in [trace0, trace1, trace2]:
            pl_nr += 1
            for trace in plot_.data:
                fig.add_trace(trace, row=1, col=pl_nr)

        fig.update_layout(title_text="Model Precision-Recall Curve", showlegend=True)

        return fig