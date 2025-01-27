import os
import pandas as pd
import numpy as np
import torch
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from efficient_kan import KAN
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Select the signal combination (chips) to run")
    parser.add_argument(
        '--chips', 
        type=str, 
        nargs='+',  # Support multiple chip selections
        choices=['chip1', 'chip2', 'chip3', 'chip4', 'chip5'],  # Available chips
        default=['chip1', 'chip2', 'chip3', 'chip4', 'chip5'],  # Default to all chips
        help="Specify which signal combinations (chips) to run, e.g., chip1 chip3 chip5"
    )
    return parser.parse_args()

def main():
    # Parse the command line arguments
    args = parse_args()
    selected_chips = args.chips  # User-selected signal combinations (chips)

    # Update data path
    path = './'

    # Load test data and predictors for the selected chips
    test_data = {
        chip: pd.read_csv(path + f'{chip}_test_7-3.csv')
        for chip in selected_chips
    }

    predictors = {
        chip: TabularPredictor.load(path + f"{chip}_predictor")
        for chip in selected_chips
    }

    # Ensure the label column exists
    label_column = 'label'  # Confirm this column name exists in all test datasets

    # Extract model predictions for each chip
    model_predictions = {
        chip: np.column_stack([  # Stack each model's prediction results
            predictors[chip].predict(test_data[chip].drop(columns=[label_column]), model=model_name)
            for model_name in predictors[chip].model_names()
        ]) for chip in predictors
    }

    # Combine model predictions into a DataFrame
    blending_features = pd.concat([  # Merge prediction results from all chips
        pd.DataFrame(model_predictions[chip], columns=[f'{chip}_Model_{i+1}' for i in range(len(predictors[chip].model_names()))])
        for chip in model_predictions
    ], axis=1)

    # Prepare labels
    labels = test_data[selected_chips[0]][label_column].values

    # Convert labels to PyTorch-compatible format
    num_classes = len(np.unique(labels))
    labels_categorical = to_categorical(labels, num_classes=num_classes)

    # KAN Network hyperparameters
    kan_params = {
        'layers_hidden': [blending_features.shape[1], 128, 64],
        'grid_size': 5,
        'spline_order': 3,
        'scale_noise': 0.1,
        'scale_base': 1.0,
        'scale_spline': 1.0,
        'base_activation': torch.nn.SiLU,
        'grid_eps': 0.02,
        'grid_range': [-1, 1],
        'epochs': 10,
        'batch_size': 32
    }

    # Build KAN network
    kan_model = KAN(
        layers_hidden=kan_params['layers_hidden'],
        grid_size=kan_params['grid_size'],
        spline_order=kan_params['spline_order'],
        scale_noise=kan_params['scale_noise'],
        scale_base=kan_params['scale_base'],
        scale_spline=kan_params['scale_spline'],
        base_activation=kan_params['base_activation'],
        grid_eps=kan_params['grid_eps'],
        grid_range=kan_params['grid_range']
    )

    # Compile KAN network
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=0.001)

    # Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    auroc_scores = []
    auprc_scores = []

    # Evaluate KAN network
    for fold, (train_index, test_index) in enumerate(kf.split(blending_features, labels)):
        print(f'Fold {fold + 1}:')

        X_train, X_test = blending_features.values[train_index], blending_features.values[test_index]
        y_train, y_test = labels_categorical[train_index], labels_categorical[test_index]

        # Convert data to PyTorch tensors
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        y_train_torch = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        test_dataset = TensorDataset(X_test_torch, y_test_torch)
        train_loader = DataLoader(train_dataset, batch_size=kan_params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=kan_params['batch_size'])

        # Train KAN model
        kan_model.train()
        for epoch in range(kan_params['epochs']):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = kan_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate KAN model
        kan_model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = kan_model(inputs)
                all_preds.append(outputs)
                all_targets.append(targets)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        final_predictions_classes = np.argmax(all_preds, axis=1)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), final_predictions_classes)
        accuracy_scores.append(accuracy)
        print(f"Accuracy for fold {fold + 1}: {accuracy:.4f}")

        if num_classes == 2:
            final_predictions_prob = all_preds[:, 1]
            auroc = roc_auc_score(np.argmax(y_test, axis=1), final_predictions_prob)
            auprc = average_precision_score(np.argmax(y_test, axis=1), final_predictions_prob)
            auroc_scores.append(auroc)
            auprc_scores.append(auprc)
            print(f"AUROC for fold {fold + 1}: {auroc:.4f}")
            print(f"AUPRC for fold {fold + 1}: {auprc:.4f}")

        print(classification_report(np.argmax(y_test, axis=1), final_predictions_classes))
        print(confusion_matrix(np.argmax(y_test, axis=1), final_predictions_classes))

    # Print overall cross-validation accuracy
    mean_accuracy = np.mean(accuracy_scores)
    print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")

    if num_classes == 2:
        mean_auroc = np.mean(auroc_scores)
        mean_auprc = np.mean(auprc_scores)
        print(f"Mean Cross-Validation AUROC: {mean_auroc:.4f}")
        print(f"Mean Cross-Validation AUPRC: {mean_auprc:.4f}")

    # Save the final KAN model
    torch.save(kan_model.state_dict(), 'kan_model.pth')
    print("KAN model saved.")

if __name__ == "__main__":
    main()
