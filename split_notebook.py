import json
import os

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_notebook(cells, output_path):
    """Create a new notebook with the given cells."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)

def main():
    # Read the original notebook
    with open('ModelTraining_compareStandard_Sampling_CV.ipynb', 'r') as f:
        original_notebook = json.load(f)

    # Create necessary directories
    directories = [
        'notebooks/data_preparation',
        'notebooks/utils',
        'notebooks/model_training',
        'notebooks/evaluation'
    ]
    for directory in directories:
        ensure_dir(directory)

    # Split cells into different notebooks
    data_loading_cells = []
    feature_engineering_cells = []
    utils_cells = []
    data_splitting_cells = []
    model_training_cells = []
    evaluation_cells = []

    current_section = None
    for cell in original_notebook['cells']:
        # Determine which section this cell belongs to based on content
        if cell['cell_type'] == 'markdown':
            content = cell['source'][0].lower() if cell['source'] else ''
            if 'setup environment' in content or 'import' in content or 'load dataset' in content:
                current_section = 'data_loading'
            elif 'feature' in content or 'preprocessing' in content:
                current_section = 'feature_engineering'
            elif 'support classes' in content or 'custom' in content:
                current_section = 'utils'
            elif 'split' in content or 'oot' in content:
                current_section = 'data_splitting'
            elif 'model' in content and 'train' in content:
                current_section = 'model_training'
            elif 'evaluation' in content or 'metrics' in content:
                current_section = 'evaluation'

        # Add cell to appropriate section
        if current_section == 'data_loading':
            data_loading_cells.append(cell)
        elif current_section == 'feature_engineering':
            feature_engineering_cells.append(cell)
        elif current_section == 'utils':
            utils_cells.append(cell)
        elif current_section == 'data_splitting':
            data_splitting_cells.append(cell)
        elif current_section == 'model_training':
            model_training_cells.append(cell)
        elif current_section == 'evaluation':
            evaluation_cells.append(cell)

    # Create the new notebooks
    notebooks = [
        ('notebooks/data_preparation/01_data_loading.ipynb', data_loading_cells),
        ('notebooks/data_preparation/02_feature_engineering.ipynb', feature_engineering_cells),
        ('notebooks/utils/custom_classes.ipynb', utils_cells),
        ('notebooks/model_training/01_data_splitting.ipynb', data_splitting_cells),
        ('notebooks/model_training/02_model_training.ipynb', model_training_cells),
        ('notebooks/evaluation/01_model_evaluation.ipynb', evaluation_cells)
    ]

    for path, cells in notebooks:
        if cells:  # Only create notebook if it has cells
            create_notebook(cells, path)
            print(f"Created {path}")

if __name__ == '__main__':
    main() 