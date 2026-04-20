#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : predict.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 真实预测
"""
import joblib
import pandas as pd
import os
from src.data_loader import feature_engineering, load_config


def predict_from_saved_model(data_path, model_type="rf"):
    """
    加载本地保存的模型并对新数据进行预测
    :param data_path: 待预测的数据文件路径 (csv)
    :param model_type: 使用哪种模型 ('lgbm' 或 'rf')
    """
    # 1. 加载配置
    config = load_config()
    model_dir = "saved_models"

    # 2. 确定模型文件路径
    model_name = f"best_model.pkl"
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}，请确认已运行 main.py 生成模型。")
        return

    # 3. 加载模型
    print(f"正在加载模型: {model_name}...")
    model = joblib.load(model_path)

    # 4. 读取待预测数据
    print(f"正在读取待预测数据: {data_path}...")
    new_df = pd.read_csv(data_path)

    # 保留 ID 列用于最后结果展示
    ids = new_df[config['data']['id_col']]

    # 5. 执行特征工程 (必须与训练时的处理逻辑完全一致)
    # 我们直接调用 src/data_loader 里封装好的函数
    X_new = feature_engineering(new_df)

    # 剔除掉不参与预测的列 (如 id 和 yield，如果新数据里带了 yield 的话)
    cols_to_drop = [config['data']['id_col'], config['data']['target_col']]
    X_new = X_new.drop(columns=[col for col in cols_to_drop if col in X_new.columns])

    # 6. 生成预测
    print("正在生成预测结果...")
    print(f"预测数据特征数: {X_new.shape[1]}, 特征列表: {X_new.columns.tolist()}")
    predictions = model.predict(X_new)

    # 7. 合并结果并输出
    result_df = pd.DataFrame({
        config['data']['id_col']: ids,
        "predicted_yield": predictions
    })

    return result_df


if __name__ == "__main__":
    # 示例调用：使用测试集数据进行一次快速预测
    # 在实际业务中，你可以把这里换成任何包含相同特征的新数据
    input_csv = "datasets/raw/test.csv"

    # 你可以选择使用 'lgbm' 或 'rf'
    results = predict_from_saved_model(input_csv, model_type="rf")

    if results is not None:
        print("\n预测完成！前 5 条结果如下：")
        print(results.head())

        # 也可以保存到本地
        results.to_csv("new_predictions.csv", index=False)
        print("\n结果已保存至 `new_predictions.csv`")
