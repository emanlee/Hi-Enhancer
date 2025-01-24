import torch
import torch.nn.functional as F
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from efficient_kan import KAN
from torch.utils.data import DataLoader, TensorDataset

# 更新数据路径
path = './'

test_data = {
    'chip1': pd.read_csv(path + 'chip1_test_7-3.csv'),
    'chip2': pd.read_csv(path + 'chip2_test_7-3.csv'),
    'chip3': pd.read_csv(path + 'chip3_test_7-3.csv'),
    'chip4': pd.read_csv(path + 'chip4_test_7-3.csv'),
    'chip5': pd.read_csv(path + 'chip5_test_7-3.csv')
}

# 确保标签列存在，替换为正确的列名
label_column = 'label'  # 请确认此列名在所有测试数据集中存在

# 加载每个 chip 的预测器
predictors = {
    'chip1': TabularPredictor.load(path + "/chip1_predictor"),
    'chip2': TabularPredictor.load(path + "/chip2_predictor"),
    'chip3': TabularPredictor.load(path + "/chip3_predictor"),
    'chip4': TabularPredictor.load(path + "/chip4_predictor"),
    'chip5': TabularPredictor.load(path + "/chip5_predictor")
}

# 提取每个 chip 的模型预测结果
model_predictions = {
    chip: np.column_stack([
        predictors[chip].predict(test_data[chip].drop(columns=[label_column]), model=model_name)
        for model_name in predictors[chip].model_names()
    ]) for chip in predictors
}

# 将模型预测结果组合成 DataFrame
blending_features = pd.concat([
    pd.DataFrame(model_predictions[chip], columns=[f'{chip}_Model_{i+1}' for i in range(len(predictors[chip].model_names()))])
    for chip in model_predictions
], axis=1)

# 准备标签
labels = test_data['chip1'][label_column].values

# 将标签转换为PyTorch所需的格式
num_classes = len(np.unique(labels))
labels_categorical = to_categorical(labels, num_classes=num_classes)

# KAN 网络超参数
kan_params = {
    'layers_hidden': [blending_features.shape[1], 128, 64],  # 输入维度及隐藏层大小
    'grid_size': 5,
    'spline_order': 3,
    'scale_noise': 0.1,
    'scale_base': 1.0,
    'scale_spline': 1.0,
    'base_activation': torch.nn.SiLU,
    'grid_eps': 0.02,
    'grid_range': [-1, 1],
    'epochs': 10,  # 训练的轮数
    'batch_size': 32  # 批量大小
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

# 编译KAN网络 - 对于PyTorch，设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(kan_model.parameters(), lr=0.001)

# 交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
auroc_scores = []
auprc_scores = []

# 评估KAN网络
for fold, (train_index, test_index) in enumerate(kf.split(blending_features, labels)):
    print(f'Fold {fold + 1}:')
    
    X_train, X_test = blending_features.values[train_index], blending_features.values[test_index]
    y_train, y_test = labels_categorical[train_index], labels_categorical[test_index]

    # 将数据转换为PyTorch张量
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)
    train_loader = DataLoader(train_dataset, batch_size=kan_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=kan_params['batch_size'])

    # 训练KAN模型
    kan_model.train()
    for epoch in range(kan_params['epochs']):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = kan_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 评估KAN模型
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
    
    # 计算 AUROC 和 AUPRC
    if num_classes == 2:
        final_predictions_prob = all_preds[:, 1]  # 获取正类的预测概率
        auroc = roc_auc_score(np.argmax(y_test, axis=1), final_predictions_prob)
        auprc = average_precision_score(np.argmax(y_test, axis=1), final_predictions_prob)
        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        print(f"AUROC for fold {fold + 1}: {auroc:.4f}")
        print(f"AUPRC for fold {fold + 1}: {auprc:.4f}")
    
    # 打印分类报告和混淆矩阵
    print(classification_report(np.argmax(y_test, axis=1), final_predictions_classes))
    print(confusion_matrix(np.argmax(y_test, axis=1), final_predictions_classes))

# 打印整体交叉验证准确率
mean_accuracy = np.mean(accuracy_scores)
print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")

if num_classes == 2:
    mean_auroc = np.mean(auroc_scores)
    mean_auprc = np.mean(auprc_scores)
    print(f"Mean Cross-Validation AUROC: {mean_auroc:.4f}")
    print(f"Mean Cross-Validation AUPRC: {mean_auprc:.4f}")

# 保存最终KAN模型
# 这里使用torch.save保存PyTorch模型
torch.save(kan_model.state_dict(), '/home/zhouhaotian/zhengqiangzi/kan_model.pth')
print("KAN model saved.")
