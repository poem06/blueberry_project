# 日期：2026/4/20
# 作者：刘玉婷
# 功能：野生蓝莓产量预测 - 完整多分类与可视化分析管道 (含高级调优与ROC/AUC)
# ******************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
from itertools import cycle


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与基础清洗
try:
    df = pd.read_csv('wild_blueberry_yield.csv.csv')
    print("数据加载成功！")
    print(f"数据集形状: {df.shape}")
except FileNotFoundError:
    print("错误：找不到文件。请确保文件存在并检查路径。")
    exit()

print("开始进行探索性数据分析 (EDA) 与特征工程")
# 2.1 创造新特征：总传粉昆虫密度
if all(col in df.columns for col in ['honeybee', 'bumbles', 'andrena', 'osmia']):
    df['Total_Pollinator_Density'] = df['honeybee'] + df['bumbles'] + df['andrena'] + df['osmia']
    print("已创建新特征: 'Total_Pollinator_Density'")

#  可视化
if 'yield' in df.columns:
    target_col = 'yield'
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(['id', 'Row#'], errors='ignore')

    # 1. 目标变量直方图 (粉色)
    plt.figure(figsize=(8, 5))
    df[target_col].plot(kind="hist", bins=40, color="pink")
    plt.title(f"{target_col} (产量分布直方图)", fontsize=14)
    plt.xlabel(target_col)
    plt.ylabel("频数 (count)")
    plt.show()

    # 2. 目标变量箱线图 (黄色边框)
    plt.figure(figsize=(6, 6))
    df[[target_col]].plot(kind="box", color="yellow")
    plt.title("产量分布箱线图 (boxplot)", fontsize=14)
    plt.show()

    # 3. 相关性热力图
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    top_corr_features = corr[target_col].abs().sort_values(ascending=False).head(15).index
    corr_top = df[top_corr_features].corr()
    plt.imshow(corr_top, aspect="auto", cmap='viridis')
    plt.xticks(range(len(top_corr_features)), top_corr_features, rotation=90)
    plt.yticks(range(len(top_corr_features)), top_corr_features)
    plt.colorbar()
    plt.title("核心特征相关性热力图", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 4. 最强特征 vs 目标变量 散点图
    top_feat = corr[target_col].abs().sort_values(ascending=False).index[1]
    plt.figure(figsize=(8, 6))
    plt.scatter(df[top_feat], df[target_col], alpha=0.4)
    plt.xlabel(top_feat)
    plt.ylabel(target_col)
    plt.title(f"最强特征 ({top_feat}) vs 产量", fontsize=14)
    plt.show()

    # 5. 特征偏度分析 (紫色)
    plt.figure(figsize=(10, 5))
    df[num_cols].skew().sort_values(ascending=False).head(15).plot(kind="bar", color="purple")
    plt.title("特征偏度分析 (top skewed features)", fontsize=14)
    plt.ylabel("偏度 (skewness)")
    plt.xticks(rotation=90)
    plt.show()

    # 6. 特征方差分析 (棕色)
    plt.figure(figsize=(10, 5))
    df[num_cols].var().sort_values(ascending=False).head(15).plot(kind="bar", color="brown")
    plt.title("特征方差分析 (top variance features)", fontsize=14)
    plt.ylabel("方差 (variance)")
    plt.xticks(rotation=90)
    plt.show()
# ---------------------------------------------------------

# 2.2 核心步骤：目标变量转换 (连续转分类)
if 'yield' in df.columns:
    df['yield_class'] = pd.qcut(df['yield'], q=3, labels=[0, 1, 2])
    print("\n已将连续 'yield' 转换为离散类别 'yield_class'。")
else:
    print("错误：未找到 'yield' 列，无法进行分类转换。")
    exit()

# 2.3 剔除无用的列和原始连续标签，防止数据泄露
drop_cols = ['id', 'Row#', 'yield']
cols_to_drop = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=cols_to_drop + ['yield_class'])
y = df['yield_class']

print("\n目标变量分布情况：")
print(y.value_counts().sort_index())

# 3. 数据集划分与标准化
print("数据集划分与标准化")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集特征维度: {X_train_scaled.shape}")
print(f"测试集特征维度: {X_test_scaled.shape}")

# 4. 模型训练与混淆矩阵
print("开始训练多分类模型")

# 训练基础模型
lr_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# 绘制混淆矩阵对比图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['低产(0)', '中产(1)', '高产(2)'], yticklabels=['低产(0)', '中产(1)', '高产(2)'])
axes[0].set_title('逻辑回归 - 混淆矩阵', fontsize=14)
axes[0].set_xlabel('预测类别', fontsize=12)
axes[0].set_ylabel('真实类别', fontsize=12)

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['低产(0)', '中产(1)', '高产(2)'], yticklabels=['低产(0)', '中产(1)', '高产(2)'])
axes[1].set_title('随机森林 (默认) - 混淆矩阵', fontsize=14)
axes[1].set_xlabel('预测类别', fontsize=12)
axes[1].set_ylabel('真实类别', fontsize=12)

plt.tight_layout()
plt.show()

# 网格搜索与交叉验证
print("进行高级优化：随机森林超参数网格搜索")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_
best_rf_pred = best_rf_model.predict(X_test_scaled)

# 6. 特征重要性分析
print("绘制特征重要性图表")
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_rf_model.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('优化后随机森林模型：影响野生蓝莓产量的前10大核心因素', fontsize=16)
plt.xlabel('特征重要性得分', fontsize=14)
plt.ylabel('特征名称', fontsize=14)
plt.show()

# 7. 最终算法性能对比图
print("生成最终分类算法性能对比图")

# 增加一个强大的梯度提升树模型作对比
hgb_model = HistGradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train)
hgb_pred = hgb_model.predict(X_test_scaled)

models = ['逻辑回归', '随机森林(优化后)', '梯度提升树 (HGB)']
preds = [lr_pred, best_rf_pred, hgb_pred]
accuracy_scores = [accuracy_score(y_test, p) for p in preds]
f1_scores = [f1_score(y_test, p, average='macro') for p in preds]

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, accuracy_scores, width, label='准确率 (Accuracy)', color='skyblue')
rects2 = ax.bar(x + width / 2, f1_scores, width, label='F1分数 (Macro-F1)', color='salmon')

ax.set_title('不同分类算法在野生蓝莓数据集上的最终性能对比', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(loc='lower right', fontsize=12)

# 在柱子上标出具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# 8. 绘制多分类 ROC 曲线与 AUC
print("绘制多分类 ROC 曲线并计算 AUC")

# 1. 标签二值化 (One-vs-Rest 策略要求)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# 2. 获取最优模型的预测概率
y_score = best_rf_model.predict_proba(X_test_scaled)

# 3. 计算每个类别的 FPR, TPR 和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 4. 绘制 ROC 曲线
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
class_names = ['低产等级(0)', '中产等级(1)', '高产等级(2)']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} ROC 曲线 (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制随机猜测的对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)', fontsize=14)
plt.ylabel('真正率 (True Positive Rate)', fontsize=14)
plt.title('优化后随机森林模型的多分类 ROC 曲线 (One-vs-Rest)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

import joblib
print("保存最终模型与标准化器到本地")

# 1. 保存调优后的冠军模型 (随机森林)
joblib.dump(best_rf_model, 'blueberry_best_rf_model.pkl')

# 2. 保存基线模型 (逻辑回归) -
joblib.dump(lr_model, 'blueberry_lr_model.pkl')

# 3. 保存预处理用的标准化器 (极其重要，所有模型共用这一个)
joblib.dump(scaler, 'blueberry_scaler.pkl')

print("🎉 大功告成！所有模型和标准化器已安全保存到当前文件夹。")