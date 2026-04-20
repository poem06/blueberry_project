#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : evaluator.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 用于评估算法
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    计算基础的回归评估指标
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 返回字典格式，方便后续汇总成表格
    return {
        "Model": model_name,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2 Score": round(r2, 4)
    }


def print_comparison_report(results_list):
    """
    接收评估结果列表，打印结构化的对比报告
    """
    if not results_list:
        return

    # 将结果列表转换为 DataFrame
    df_results = pd.DataFrame(results_list)
    df_results.set_index("Model", inplace=True)

    print("\n" + "=" * 75)
    print(" 蓝莓产量预测模型性能对比报告 (Model Comparison) ".center(75, "="))
    print("=" * 75)
    # 使用 to_markdown 打印出漂亮的表格
    print(df_results.to_markdown())
    print("=" * 75 + "\n")
