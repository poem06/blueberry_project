#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : lgbm_model.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 用于蓝莓产量预测模型训练代码
"""
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

def train_lgbm(X_train, y_train, param_grid, cv=3, n_iter=10, random_state=42):
    """
    训练 LightGBM 模型并进行超参数调优
    """
    print("\n[LGBM] 初始化 LightGBM 模型...")
    lgbm = lgb.LGBMRegressor(random_state=random_state, verbose=-1)

    print(f"[LGBM] 开始进行随机网格搜索调参 (组合数: {n_iter}, 折数: {cv})...")
    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"[LGBM] 最佳参数: {search.best_params_}")
    return search.best_estimator_

def predict_lgbm(model, X):
    return model.predict(X)
