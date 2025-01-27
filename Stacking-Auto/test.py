# Ensure pandas is imported
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef

# 1. Load the test data
test_data = pd.read_csv('test.csv')  # Assuming the test dataset file is named chip6_test.csv

# 2. Load the RF model
rf_predictor = TabularPredictor.load('./output_folder/fold_1_predictor/')  # Load RF model

# 3. Make predictions on the test data using the RF model
test_data = TabularDataset(test_data)
rf_test_predictions = rf_predictor.predict(test_data, model='LightGBM_BAG_L1')

# Add RF predictions as a new feature to the test data
test_data['model_pred'] = rf_test_predictions

# 4. Save the dataset with the new feature
test_data.to_csv('test_with_new_features.csv', index=False)
print('Test data with new features saved as test_with_new_features.csv')

# 5. Load the final trained model
final_predictor = TabularPredictor.load('./output_folder/layer2_predictor/')  # Load the final model

# 6. Get the best-performing model for final predictions
leaderboard = final_predictor.leaderboard(test_data, silent=False)
best_model_name = leaderboard.iloc[0]['model']  # Get the name of the best model
print(f"Best model for final predictions: {best_model_name}")

# 7. Make final predictions on the test data with new features
final_test_predictions = final_predictor.predict(test_data, model=best_model_name)

# Add final predictions to the test data
test_data['final_pred'] = final_test_predictions

# 8. Save the final predictions dataset (including new features and predictions)
test_data.to_csv('test_with_final_predictions.csv', index=False)
print('Test data with final predictions saved as test_with_final_predictions.csv')

# Get the actual target column (assuming the target column is named 'label')
y_true = test_data['label']

# 9. Calculate evaluation metrics
y_pred = test_data['final_pred']

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate sensitivity (Recall)
sensitivity = recall_score(y_true, y_pred)

# Calculate specificity
tn = sum((y_true == 0) & (y_pred == 0))
fp = sum((y_true == 0) & (y_pred == 1))
specificity = tn / (tn + fp)

# Calculate Matthews correlation coefficient (MCC)
mcc = matthews_corrcoef(y_true, y_pred)

# 10. Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Sn): {sensitivity:.4f}")
print(f"Specificity (Sp): {specificity:.4f}")
print(f"MCC: {mcc:.4f}")

# 11. Display the leaderboard with all models' performances
print(leaderboard)
