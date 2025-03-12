import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# Load the original data
def load_data():
    with open('models/oot_X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('models/oot_y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open('models/score_balanced_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return X, y, model

def create_altered_datasets(X, y, fraud_rates=[0.005, 0.01, 0.02, 0.05]):
    """
    Create datasets with different fraud rates by undersampling non-fraud cases
    """
    datasets = []
    fraud_indices = np.where(y == 1)[0]
    non_fraud_indices = np.where(y == 0)[0]
    
    for rate in fraud_rates:
        # Calculate how many non-fraud cases we need
        n_fraud = len(fraud_indices)
        n_non_fraud = int(n_fraud / rate - n_fraud)
        
        # Randomly sample non-fraud cases
        sampled_non_fraud = np.random.choice(non_fraud_indices, size=n_non_fraud, replace=True)
        
        # Combine indices
        combined_indices = np.concatenate([fraud_indices, sampled_non_fraud])
        
        # Create new dataset
        X_new = X.iloc[combined_indices]
        y_new = y.iloc[combined_indices]
        
        datasets.append((X_new, y_new, f"Fraud Rate: {rate:.1%}"))
    
    return datasets

def plot_curves(datasets, model):
    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Precision-Recall Curves', 'ROC Curves'),
                       horizontal_spacing=0.15)
    
    # Colors for different curves
    colors = ['blue', 'red', 'green', 'purple']
    
    for (X, y, label), color in zip(datasets, colors):
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        # Add PR curve
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name=f"{label} (PR AUC: {pr_auc:.3f})",
                      line=dict(color=color)),
            row=1, col=1
        )
        
        # Add ROC curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f"{label} (ROC AUC: {roc_auc:.3f})",
                      line=dict(color=color)),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        showlegend=True,
        title_text="Model Performance Across Different Fraud Rates"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Recall", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
    
    # Add diagonal line for ROC plot
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                  line=dict(color='black', dash='dash'),
                  showlegend=False),
        row=1, col=2
    )
    
    return fig

def main():
    # Load data
    print("Loading data...")
    X, y, model = load_data()
    
    # Create datasets with different fraud rates
    print("Creating altered datasets...")
    datasets = create_altered_datasets(X, y)
    
    # Plot curves
    print("Generating plots...")
    fig = plot_curves(datasets, model)
    print("Analysis complete! Check fraud_rate_analysis.html for the results.")

if __name__ == "__main__":
    main() 