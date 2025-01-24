import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# 确保路径正确
path = ''  # 更新为你的CSV文件所在目录

# 用户可以选择要运行的chips，例如：['chip1', 'chip3', 'chip5']
chips = ['chip1', 'chip2', 'chip3', 'chip4', 'chip5']

for chip in chips:
    # 加载数据
    train_data = pd.read_csv(f'{path}{chip}_train_7-3.csv')
    test_data = pd.read_csv(f'{path}{chip}_test_7-3.csv')

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

