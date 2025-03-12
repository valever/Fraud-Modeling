import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import plotly.express as px
def create_shifted_datasets(X, y, n_shifts=4):
    """
    Create datasets with shifted fraud populations by applying different transformations
    to the fraud cases while maintaining the same fraud rate
    """
    datasets = []
    
    # Ensure X is a DataFrame and y is a numpy array
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, pd.Series):
        y = y.values
    
    # Reset index to ensure we have clean integer indices
    X = X.reset_index(drop=True)
    
    fraud_mask = y == 1
    
    # Get numeric columns for shifting
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    # Create different shifts
    shifts = [
        ("Original", 0),  # No shift
        ("Positive Shift", 1),  # Shift all numeric features up
        ("Negative Shift", -1),  # Shift all numeric features down
        ("Mixed Shift", 0.5)  # Mixed shift (some up, some down)
    ]
    
    for shift_name, shift_factor in shifts:
        # Create a deep copy to avoid modifying the original data
        X_new = pd.DataFrame()
        for col in X.columns:
            X_new[col] = X[col].copy()
        y_new = y.copy()
        
        if shift_name != "Original":
            # Apply different shifts based on the type
            if shift_name == "Mixed Shift":
                # Randomly shift some features up and some down
                for col in numeric_cols:
                    # Convert to float64 and create new array
                    col_data = X_new[col].astype(float).values
                    shift_direction = np.random.choice([-1, 1], size=sum(fraud_mask))
                    col_data[fraud_mask] += shift_factor * shift_direction
                    X_new[col] = col_data
            else:
                # Apply uniform shift to all numeric features
                for col in numeric_cols:
                    # Convert to float64 and create new array
                    col_data = X_new[col].astype(float).values
                    col_data[fraud_mask] += shift_factor
                    X_new[col] = col_data
            
            # Standardize the shifted features to maintain similar scale
            scaler = StandardScaler()
            X_new[numeric_cols] = scaler.fit_transform(X_new[numeric_cols].astype(float))
        
        datasets.append((X_new, y_new, shift_name))
    
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
        title_text="Model Performance Across Different Fraud Population Shifts"
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

def plot_prediction_distribution(datasets, model):
    
    # Colors for different curves
    df_list = []    
    for X, y, name in datasets:
        y_pred_proba = model.predict_proba(X)[:, 1]
        df_list.append(pd.DataFrame({'score': y_pred_proba, 'label': y, 'dataset': [name]*len(y_pred_proba)}))

    df = pd.concat(df_list)
    fig = px.histogram(df, x='score', color='label'
                       , nbins=50, facet_col='dataset',
                        histnorm='probability density',
                        labels=dict(color='True Labels', x='Score'),
                        title='Model Performance')

    return fig

def main():
    # Load data
    print("Loading data...")
    X, y, model = load_data()
    
    # Create shifted datasets
    print("Creating shifted datasets...")
    datasets = create_shifted_datasets(X, y)
    
    # Plot curves
    print("Generating plots...")
    fig = plot_curves(datasets, model)
    print("Analysis complete! Check fraud_population_shift.html for the results.")

if __name__ == "__main__":
    main()