#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : rf_model.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 随机森林训练
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# 注意看参数列表：现在它主动接收 param_grid, cv, n_iter
def train_rf(X_train, y_train, param_grid, cv=3, n_iter=10, random_state=42):
    """
    训练随机森林模型并进行超参数调优 (参数由外部 YAML 注入)
    """
    print("\n[RF] 初始化随机森林模型...")
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    print(f"[RF] 开始进行随机网格搜索调参 (组合数: {n_iter}, 折数: {cv})...")
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"[RF] 最佳参数: {search.best_params_}")
    return search.best_estimator_

def predict_rf(model, X):
    return model.predict(X)
