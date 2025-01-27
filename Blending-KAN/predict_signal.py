import argparse
import torch
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
from efficient_kan import KAN

# Set up command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Choose chip combinations for prediction.")
    parser.add_argument(
        '--chips', 
        type=str, 
        help="Comma-separated list of chips to use (e.g., 'chip1,chip3,chip5')", 
        default='chip1,chip2,chip3,chip4,chip5'
    )
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Use the current directory as the path
path = './'

# Select the chips to use based on the user input
selected_chips = args.chips.split(',')

# Load the test data and predictors for the selected chips
test_data = {
    chip: pd.read_csv(path + f'{chip}_predict.csv')
    for chip in selected_chips
}

predictors = {
    chip: TabularPredictor.load(path + f"{chip}_predictor")
    for chip in selected_chips
}

# Ensure there are no missing values in the test data (simple imputation)
for chip in test_data:
    test_data[chip] = test_data[chip].fillna(0)

# Extract model predictions for each chip
model_predictions = {
    chip: np.column_stack([  # Stack predictions from each model for each chip
        predictors[chip].predict(test_data[chip], model=model_name)
        for model_name in predictors[chip].model_names()
    ]) for chip in predictors
}

# Combine all chip predictions into a single DataFrame
blending_features = pd.concat([  # Merge predictions from all chips
    pd.DataFrame(model_predictions[chip], columns=[f'{chip}_Model_{i+1}' for i in range(len(predictors[chip].model_names()))])
    for chip in model_predictions
], axis=1)

# KAN network hyperparameters
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

# Construct the KAN network
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

# Load the best trained model weights
best_model_path = f"{path}kan_model.pth"
kan_model.load_state_dict(torch.load(best_model_path))

# Evaluate the KAN model on the test data
kan_model.eval()
with torch.no_grad():
    X_test = blending_features.values
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    # Get model output (predicted probabilities)
    predictions = kan_model(X_test_torch)
    
    # Convert to numpy and get class probabilities
    predictions_numpy = predictions.numpy()
    final_predictions_classes = np.argmax(predictions_numpy, axis=1)
    final_predictions_prob = predictions_numpy[:, 1]  # Assuming binary classification

# Save predictions to CSV
output_df = pd.DataFrame({
    'Sample_ID': test_data[selected_chips[0]].index,  # Use index as Sample_ID
    'Predicted_Class': final_predictions_classes,
    'Predicted_Probability': final_predictions_prob
})

output_df.to_csv(path + 'predictions.csv', index=False)

print("Predictions have been saved to 'predictions.csv'.")
