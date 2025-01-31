import argparse
import torch
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
from efficient_kan import KAN

# Set up the command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Select the chips to be used for prediction.")
    parser.add_argument(
        '--chips', 
        type=str, 
        help="Comma-separated list of chips (e.g., 'chip1,chip3,chip5')", 
        default='chip1,chip2,chip3,chip4,chip5'
    )
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Use the current directory as the path
path = './'

# Select the chips based on user input
selected_chips = args.chips.split(',')

# Load test data and predictors for the selected chips
test_data = {
    chip: pd.read_csv(path + f'{chip}_predict.csv')
    for chip in selected_chips
}

predictors = {
    chip: TabularPredictor.load(path + f"{chip}_predictor")
    for chip in selected_chips
}

# Ensure there are no missing values in the test data (simple fill with zeros)
for chip in test_data:
    test_data[chip] = test_data[chip].fillna(0)

# Extract model predictions for each chip
model_predictions = {
    chip: np.column_stack([  # Stack the predictions from each model for the chip
        predictors[chip].predict(test_data[chip], model=model_name)
        for model_name in predictors[chip].model_names()
    ]) for chip in predictors
}

# Combine the predictions from all chips into a single DataFrame
blending_features = pd.concat([  # Concatenate the predictions from all chips
    pd.DataFrame(model_predictions[chip], columns=[f'{chip}_Model_{i+1}' for i in range(len(predictors[chip].model_names()))])
    for chip in model_predictions
], axis=1)

# Hyperparameters for the KAN network
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

# Build the KAN network
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

# Load the trained KAN model weights
best_model_path = f"{path}kan_model.pth"
kan_model.load_state_dict(torch.load(best_model_path))

# Evaluate the KAN model on the test data
kan_model.eval()
with torch.no_grad():
    X_test = blending_features.values
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    # Get the model output (logits)
    logits = kan_model(X_test_torch)
    
    # Apply the Softmax function to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Convert to numpy and obtain the class predictions
    predictions_numpy = probabilities.numpy()
    final_predictions_classes = np.argmax(predictions_numpy, axis=1)

# Save the predictions to a CSV file (including the top two probabilities)
output_df = pd.DataFrame({
    'Sample_ID': test_data[selected_chips[0]].index,  # Use the index of the first chip as Sample_ID
    'Predicted_Class': final_predictions_classes  # Predicted classes
})

# Add the top two probabilities to the DataFrame
output_df['Probability_Class_0'] = predictions_numpy[:, 0]
output_df['Probability_Class_1'] = predictions_numpy[:, 1]

# Save the prediction results to a CSV file
output_df.to_csv(path + 'predictions_with_probabilities.csv', index=False)

print("Prediction results (including the top two probabilities) have been saved to 'predictions_with_probabilities.csv'.")