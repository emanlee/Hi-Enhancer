import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef

# 1. Load the test data
test_data = pd.read_csv('features.csv')

# 2. Load the pre-trained Random Forest model
rf_predictor = TabularPredictor.load('./output_folder/fold_1_predictor/')  # Load RF model

# 3. Make predictions on the test data using the RF model
test_data = TabularDataset(test_data)
rf_test_predictions = rf_predictor.predict(test_data, model='LightGBM_BAG_L1')

# Add the RF predictions as a new feature to the test dataset
test_data['model_pred'] = rf_test_predictions

# 4. Save the test data with the new feature
test_data.to_csv('test_with_new_features.csv', index=False)
print('Test data with new features saved as test_with_new_features.csv')

# 5. Load the final model that was trained
final_predictor = TabularPredictor.load('./output_folder/layer2_predictor/')  # Load the final model

# 6. Specify the best model for final predictions
best_model_name = 'NeuralNetTorch_r185_BAG_L2'
print(f"Best model for final predictions: {best_model_name}")

# 7. Make final predictions on the test dataset with the new features
final_test_predictions = final_predictor.predict(test_data, model=best_model_name)

# Get the predicted probabilities for each class
pred_prob = final_predictor.predict_proba(test_data)

# Combine the predictions and probabilities with the original data
result_df = test_data.copy()  # Keep the original data
result_df['predictions'] = final_test_predictions  # Add prediction results
result_df = pd.concat([result_df, pred_prob], axis=1)  # Combine prediction probabilities

# Print the first few rows of the merged data
print(result_df.head())

# Save the final results to a CSV file
result_df.to_csv('predictions_with_proba.csv', index=False)

print("Prediction completed, results saved to 'predictions_with_proba.csv'")
