#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : xgb_model.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : xgboost训练
"""

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

def train_xgb(X_train, y_train, param_grid, cv=3, n_iter=10, random_state=42):
    """
    训练 XGBoost 模型并进行超参数调优 (参数由外部 YAML 注入)
    """
    print("\n[XGBoost] 初始化 XGBoost 模型...")
    xgbr = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)

    print(f"[XGBoost] 开始进行随机网格搜索调参 (组合数: {n_iter}, 折数: {cv})...")
    search = RandomizedSearchCV(
        estimator=xgbr,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"[XGBoost] 最佳参数: {search.best_params_}")
    return search.best_estimator_

def predict_xgb(model, X):
    return model.predict(X)
