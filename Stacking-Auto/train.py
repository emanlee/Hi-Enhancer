import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor, TabularDataset

# 1. 加载原始训练数据
train_data = pd.read_csv('chip6_train.csv')

# 2. 设置十倍交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. 初始化存储所有模型预测结果的 DataFrame
all_model_predictions = pd.DataFrame()

# 4. 定义您想要使用的模型（AutoGluon 支持的模型名称）
specified_models = ['GBM']  # 只使用随机森林模型（RF）

# 5. 进行十倍交叉验证
fold_num = 1
for train_index, val_index in kf.split(train_data):
    # 获取训练集和验证集
    train_fold = train_data.iloc[train_index]
    val_fold = train_data.iloc[val_index]

    # 将数据转换为 TabularDataset 格式
    train = TabularDataset(train_fold)
    val = TabularDataset(val_fold)

    # 创建并训练预测器，指定使用的模型
    predictor = TabularPredictor(
        label='label',
        problem_type='binary',
        eval_metric='accuracy'
    ).fit(
        train_data=train,
        presets='best_quality',
        hyperparameters={model: {} for model in specified_models}  # 只使用指定的模型
    )

    # 对验证集进行预测，并收集每个基模型的预测结果
    val_predictions = pd.DataFrame({'index': val_index})

    for model in predictor.model_names():  # 获取训练的所有模型名称
        # 对每个基模型进行类别预测（0 或 1）
        val_pred = predictor.predict(val, model=model).values
        val_predictions[f'{model}_label'] = val_pred

    # 保存每一折的预测结果
    all_model_predictions = pd.concat([all_model_predictions, val_predictions], axis=0)

    # 保存每一折的预测结果为 CSV 文件
    val_predictions.to_csv(f'fold_{fold_num}_model_predictions.csv', index=False)
    print(f'Fold {fold_num} model predictions saved.')
    fold_num += 1

# 保存所有基模型的预测结果
all_model_predictions.to_csv('all_model_predictions.csv', index=False)
print('All model predictions saved.')

# 6. 将预测结果与原始数据合并
# 在合并之前重置原始数据和预测结果的索引
train_data.reset_index(drop=True, inplace=True)
all_model_predictions.reset_index(drop=True, inplace=True)

# 合并原始数据和预测结果
merged_data = pd.concat([train_data, all_model_predictions.iloc[:, 1]], axis=1)

# 重命名列，将预测标签命名为 'model_pred'
merged_data.columns = list(train_data.columns) + ['model_pred']

# 7. 保存合并后的训练数据
merged_data.to_csv('merged_train_data.csv', index=False)
print('Merged training data saved as merged_train_data.csv')

# 8. 使用合并后的数据进行新的训练
# 创建 TabularDataset 格式的数据
new_train_data = TabularDataset(merged_data)

# 训练新模型，假设新的目标变量是 'label'（原始的目标列）
predictor = TabularPredictor(
    label='label',  # 原始的目标列
    problem_type='binary',
    eval_metric='accuracy'
).fit(
    new_train_data,
    presets='best_quality'
)

# 输出训练完成的模型
print("Model training completed.")
