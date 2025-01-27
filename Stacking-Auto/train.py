import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor, TabularDataset

# 1. Load the original training data
train_data = pd.read_csv('train.csv')

# 2. Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. Initialize a DataFrame to store all model predictions
all_model_predictions = pd.DataFrame()

# 4. Define the models you want to use (model names supported by AutoGluon)
specified_models = ['GBM']  # Only use the GBM model

# Output folder (needs to be defined)
output_folder = './output_folder'
os.makedirs(output_folder, exist_ok=True)

# 6. Perform 10-fold cross-validation
fold_num = 1
for train_index, val_index in kf.split(train_data):
    # Get the training and validation sets for this fold
    train_fold = train_data.iloc[train_index]
    val_fold = train_data.iloc[val_index]

    # Convert the data to TabularDataset format
    train = TabularDataset(train_fold)
    val = TabularDataset(val_fold)

    # Specify a different model path for each fold
    model_path = os.path.join(output_folder, f'fold_{fold_num}_predictor')
    os.makedirs(model_path, exist_ok=True)  # Create a folder for this fold

    # Create and train the predictor, specifying the model to use
    predictor = TabularPredictor(
        label='label',
        problem_type='binary',
        eval_metric='accuracy',
        path=model_path  # Specify the model save path
    ).fit(
        train_fold,  # Directly pass the train_fold
        presets='best_quality',
        hyperparameters={model: {} for model in specified_models}  # Only use the specified models
    )

    # Make predictions on the validation set without specifying a model, using the default model (only one GBM model here)
    val_predictions = pd.DataFrame({'index': val_index})

    val_pred = predictor.predict(val).values  # No model specified, auto-use the trained model
    val_predictions['GBM_label'] = val_pred

    # Collect all fold predictions
    all_model_predictions = pd.concat([all_model_predictions, val_predictions], axis=0, ignore_index=True)

    fold_num += 1

# Save all model predictions to the specified folder
all_model_predictions_file = os.path.join(output_folder, 'all_model_predictions.csv')
all_model_predictions.to_csv(all_model_predictions_file, index=False)
print(f'All model predictions saved to {all_model_predictions_file}')

# 7. Merge the prediction results with the original data
train_data.reset_index(drop=True, inplace=True)
all_model_predictions.reset_index(drop=True, inplace=True)

# Merge the original data and predictions
merged_data = pd.concat([train_data, all_model_predictions['GBM_label']], axis=1)

# Rename columns, setting the prediction label to 'model_pred'
merged_data.columns = list(train_data.columns) + ['model_pred']

# 8. Save the merged training data to the specified folder
merged_train_data_file = os.path.join(output_folder, 'merged_train_data.csv')
merged_data.to_csv(merged_train_data_file, index=False)
print(f'Merged training data saved to {merged_train_data_file}')

# 9. Use the merged data to train a new model
new_train_data = TabularDataset(merged_data)
# Specify the model save path
model_path = os.path.join(output_folder, 'layer2_predictor')

# Train the new model, assuming the target variable is 'label' (the original target column)
predictor = TabularPredictor(
    label='label',  # Original target column
    problem_type='binary',
    eval_metric='accuracy',
    path=model_path  # Ensure the model save path is correct
).fit(
    new_train_data,
    presets='best_quality'
)

# Output completion of model training
print("Model training completed.")
