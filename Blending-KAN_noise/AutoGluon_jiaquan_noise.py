
import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

# 设置路径
path = ''  # 更新为你的CSV文件所在目录
output_path = ''  # 更新为保存噪声数据的目录

# 设置均值和标准差
mu = 0        # 噪声均值
sigma = 0.2   # 噪声标准差

# 定义添加高斯噪声的函数
def add_gaussian_noise(column, mu=0, sigma=0.1):
    noise = np.random.normal(mu, sigma, column.shape)
    return column + noise

# 数据集名称列表
chips = ['chip1', 'chip2', 'chip3', 'chip4', 'chip5']

for chip in chips:
    # 加载数据
    train_data = pd.read_csv(f'{path}{chip}_train_7-3.csv')
    test_data = pd.read_csv(f'{path}{chip}_test_7-3.csv')

    # 为数值特征添加高斯噪声，跳过第一列（标签列）
    for column in train_data.columns[1:]:  # 从第二列开始
        if pd.api.types.is_numeric_dtype(train_data[column]):
            train_data[column] = add_gaussian_noise(train_data[column], mu, sigma)

    for column in test_data.columns[1:]:
        if pd.api.types.is_numeric_dtype(test_data[column]):
            test_data[column] = add_gaussian_noise(test_data[column], mu, sigma)

    # 保存带噪声的数据
    #train_data.to_csv(f'{output_path}{chip}_train_7-3_noisy.csv', index=False)
    #test_data.to_csv(f'{output_path}{chip}_test_7-3_noisy.csv', index=False)

    # 计算权重
    train_data['weight'] = train_data['label'].apply(lambda x: 5 if x == 1 else 1)

    # 根据权重重复样本
    train_data_weighted = train_data.loc[train_data.index.repeat(train_data['weight'])].reset_index(drop=True)

    # 移除权重列
    train_data_weighted = train_data_weighted.drop(columns=['weight'])

    # 转换为 TabularDataset
    train = TabularDataset(train_data_weighted)
    test = TabularDataset(test_data)

    # 为每个chip指定不同的模型路径
    model_path = f'{chip}_predictor'

    # 创建并训练预测器
    predictor = TabularPredictor(
        label='label',
        problem_type='binary',
        eval_metric='accuracy',
        path=model_path
    ).fit(
        train_data=train,
        presets='best_quality',
        num_bag_folds=5
    )

    # 评估模型
    leaderboard = predictor.leaderboard(test, silent=True)
    print(f'Leaderboard for {chip}:')
    print(leaderboard)

    # 进行预测
    predictions = predictor.predict(test)
    print(f'Predictions for {chip}:')
    print(predictions.head())  # 打印前几条预测结果

    print(f'{chip} training and evaluation completed.\n')
