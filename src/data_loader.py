#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : data_loader.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 加载数据集并统一进行特征工程
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path="config/config.yaml"):
    """读取 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_engineering(df):
    """
    统一的特征工程函数
    在这里添加你的业务逻辑，保证训练集和测试集经过完全相同的处理
    """
    df_processed = df.copy()

    # 示例衍生特征：计算三种主要授粉蜜蜂的总密度
    if all(col in df_processed.columns for col in ['bumbles', 'andrena', 'osmia']):
        df_processed['total_bees'] = df_processed['bumbles'] + df_processed['andrena'] + df_processed['osmia'] + df_processed['honeybee']
        df_processed = df_processed.drop(columns=['honeybee', 'bumbles', 'andrena', 'osmia'])
        print(df_processed)
    return df_processed


def get_train_val_data(config_path="config/config.yaml"):
    """
    读取训练数据，进行特征工程，并划分为训练集和验证集
    """
    config = load_config(config_path)

    print("正在加载训练数据...")
    train_df = pd.read_csv(config['data']['train_path'])

    # 设置 ID 为索引（如果不设为索引，也可以直接 drop 掉）
    if config['data']['id_col'] in train_df.columns:
        train_df.set_index(config['data']['id_col'], inplace=True)

    # 执行特征工程
    train_df = feature_engineering(train_df)

    # 分离特征 (X) 和标签 (y)
    X = train_df.drop(columns=[config['data']['target_col']])
    y = train_df[config['data']['target_col']]

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config['project']['random_seed']
    )

    print(f"数据加载完毕! 训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    return X_train, X_val, y_train, y_val, X, y


def get_test_data(config_path="config/config.yaml"):
    """
    读取并处理最终的测试集数据
    """
    config = load_config(config_path)

    test_df = pd.read_csv(config['data']['test_path'])
    test_ids = test_df[config['data']['id_col']]  # 保存 ID 用于提交

    if config['data']['id_col'] in test_df.columns:
        test_df.set_index(config['data']['id_col'], inplace=True)

    test_df = feature_engineering(test_df)

    return test_df, test_ids
