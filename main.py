#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : main.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 用于蓝莓产量预测模型训练代码
"""

import os
import joblib
from src.data_loader import get_train_val_data, load_config
from src.models.rf_model import train_rf
from src.models.xgb_model import train_xgb
from src.models.lgbm_model import train_lgbm
from src.evaluator import evaluate_model, print_comparison_report
from src.visualizer import plot_multi_model_comparison, plot_residuals, plot_feature_importance


def main():
    # 1. 加载配置
    cfg = load_config()
    model_names = cfg.get('compare_models', ['rf'])
    X_train, X_val, y_train, y_val, X_full, y_full = get_train_val_data()

    results_list = []
    all_predictions = {}
    model_factory = {"rf": train_rf, "xgb": train_xgb, "lgbm": train_lgbm}

    # 初始化“擂台”变量
    best_r2 = -float('inf')  # 初始最高分为负无穷
    best_model_obj = None
    best_model_name = ""

    # 2. 开启循环大 PK
    for m_type in model_names:
        print(f"\n>>> 正在训练并评估模型: {m_type.upper()} ...")
        m_cfg = cfg['models'][m_type]

        # 训练
        model = model_factory[m_type](
            X_train, y_train,
            param_grid=m_cfg['params'],
            **m_cfg['search_params']
        )

        # 预测与评估
        preds = model.predict(X_val)
        metrics = evaluate_model(y_val, preds, model_name=m_type.upper())

        results_list.append(metrics)
        all_predictions[m_type] = preds

        # 以 R2 Score 为第一基准 (越接近1越好)
        current_r2 = metrics['R2 Score']
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_model_name = m_type.upper()
            best_model_obj = model

        # 绘图逻辑
        plot_dir = cfg['project']['plot_dir']
        plot_residuals(y_val, preds, m_type.upper(), plot_dir)
        plot_feature_importance(model, X_train.columns, m_type.upper(), plot_dir)

    # 3. 打印对比报告
    print_comparison_report(results_list)

    # 4. 终极对比曲线
    plot_multi_model_comparison(y_val, all_predictions, cfg['project']['plot_dir'])

    print("\n" + "=" * 50)
    print(f"最佳模型: {best_model_name}")
    print(f"验证集最高 R2 Score: {best_r2:.4f}")
    print("=" * 50)

    # 保存模型
    save_dir = cfg['project']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pkl")

    joblib.dump(best_model_obj, save_path)
    print(f"\n最佳模型已保存至: {save_path}")

    print(f"\n>>> 所有的图表已保存在 `{cfg['project']['plot_dir']}/` 文件夹下 <<<")


if __name__ == "__main__":
    main()
