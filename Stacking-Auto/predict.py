import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef

# 加载测试数据
test_data = pd.read_csv('features.csv')

# 加载已训练好的模型
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

# 6. 指定最终模型的最佳表现模型
best_model_name = 'NeuralNetTorch_r185_BAG_L2'
print(f"Best model for final predictions: {best_model_name}")

# 7. 对包含新特征的测试集进行最终预测
final_test_predictions = final_predictor.predict(test_data, model=best_model_name)
# 获取每个类别的预测概率
pred_prob =  final_predictor.predict_proba(test_data)

# 将预测结果和预测概率合并到源数据
result_df = test_data.copy()  # 保留原始数据
result_df['predictions'] = final_test_predictions  # 添加预测结果
result_df = pd.concat([result_df, pred_prob], axis=1)  # 合并预测概率

# 打印合并后的数据前几行
print(result_df.head())

# 将结果保存到 CSV 文件
result_df.to_csv('predictions_with_proba.csv', index=False)

print("预测已完成，结果已保存到 'predictions_with_proba.csv'")

