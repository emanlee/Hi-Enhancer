import torch
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
from efficient_kan import KAN

# 使用当前文件夹作为路径
path = './'

# 选择要运行的 chips
selected_chips = ['chip1', 'chip2', 'chip3', 'chip4', 'chip5']

# 加载所选的 chip 的测试数据和预测器
test_data = {
    chip: pd.read_csv(path + f'{chip}_test_7-3.csv')
    for chip in selected_chips
}

predictors = {
    chip: TabularPredictor.load(path + f"{chip}_predictor")
    for chip in selected_chips
}

# 提取每个 chip 的模型预测结果
model_predictions = {
    chip: np.column_stack([
        predictors[chip].predict(test_data[chip], model=model_name)
        for model_name in predictors[chip].model_names()
    ]) for chip in predictors
}

# 将模型预测结果组合成 DataFrame
blending_features = pd.concat([
    pd.DataFrame(model_predictions[chip], columns=[f'{chip}_Model_{i+1}' for i in range(len(predictors[chip].model_names()))])
    for chip in model_predictions
], axis=1)

# KAN 网络超参数
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

# 构建 KAN 网络
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

# 加载最佳训练好的模型权重
best_model_path = f"{path}kan_model.pth"  # 请替换为实际保存的最佳模型路径
kan_model.load_state_dict(torch.load(best_model_path))

# 评估KAN模型在测试数据上的表现
kan_model.eval()
with torch.no_grad():
    X_test = blending_features.values
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    # 获取模型输出（预测概率分数）
    predictions = kan_model(X_test_torch)
    
    # 转换为 numpy 并获取类别概率
    predictions_numpy = predictions.numpy()
    final_predictions_classes = np.argmax(predictions_numpy, axis=1)
    final_predictions_prob = predictions_numpy[:, 1]  # 针对类别1的概率

# 将预测结果保存到 CSV 文件
output_df = pd.DataFrame({
    'Sample_ID': test_data[selected_chips[0]].index,  # 假设测试数据有索引或 ID 列
    'Predicted_Class': final_predictions_classes,
    'Predicted_Probability': final_predictions_prob
})

output_df.to_csv(path + 'predictions.csv', index=False)

print("预测结果已保存到 predictions.csv 文件中。")