# 确保导入pandas库
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef

# 1. 加载测试数据
test_data = pd.read_csv('test.csv')  # 假设测试集文件名为 chip6_test.csv

# 2. 加载RF模型
rf_predictor = TabularPredictor.load('./AutogluonModels/ag-20241210_031852/')  # 加载RF模型

# 3. 对测试集进行RF预测
test_data = TabularDataset(test_data)
rf_test_predictions = rf_predictor.predict(test_data, model='LightGBM_BAG_L1')

# 将RF预测结果添加到测试集作为新特征
test_data['model_pred'] = rf_test_predictions

# 4. 保存包含新特征的数据集
test_data.to_csv('test_with_new_features.csv', index=False)
print('Test data with new features saved as test_with_new_features.csv')

# 5. 加载最终训练的模型
final_predictor = TabularPredictor.load('./AutogluonModels/ag-20241210_032651/')  # 加载最终模型

# 6. 获取最终模型的最佳表现模型
leaderboard = final_predictor.leaderboard(test_data, silent=False)
best_model_name = leaderboard.iloc[0]['model']  # 获取最优模型的名称
print(f"Best model for final predictions: {best_model_name}")

# 7. 对包含新特征的测试集进行最终预测
final_test_predictions = final_predictor.predict(test_data, model=best_model_name)

# 将最终预测结果添加到测试集
test_data['final_pred'] = final_test_predictions

# 8. 保存最终预测结果（包含新特征和预测）
test_data.to_csv('test_with_final_predictions.csv', index=False)
print('Test data with final predictions saved as test_with_final_predictions.csv')

# 获取实际目标列（假设目标列为'target'）
y_true = test_data['label']

# 9. 计算评估指标
y_pred = test_data['final_pred']

# 计算准确率（Accuracy）
accuracy = accuracy_score(y_true, y_pred)

# 计算灵敏度（Sensitivity / Recall）
sensitivity = recall_score(y_true, y_pred)

# 计算特异性（Specificity）
tn = sum((y_true == 0) & (y_pred == 0))
fp = sum((y_true == 0) & (y_pred == 1))
specificity = tn / (tn + fp)

# 计算Matthews相关系数（MCC）
mcc = matthews_corrcoef(y_true, y_pred)

# 10. 打印输出
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Sn): {sensitivity:.4f}")
print(f"Specificity (Sp): {specificity:.4f}")
print(f"MCC: {mcc:.4f}")

# 11. 查看所有模型的表现（leaderboard）
print(leaderboard)
