#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    : visualizer.py
@Author  : 王诗哲
@Date    : 2026/4/20
@Desc    : 多模型可视化对比
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Rectangle  # 引入矩形对象用于提取直方图数据

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_residuals(y_true, y_pred, model_name, save_dir):
    """
    绘制残差分布图
    """
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))

    # === [核心修改 1]：明确捕获直方图生成的 patch 对象 ===
    ax = sns.histplot(residuals, kde=True, color='seagreen', bins=50,
                      label='误差频数分布')

    # === [核心修改 2]：自动计算并标注最高柱对应的数值 ===
    # 提取所有柱子对象
    patches = ax.patches
    if patches:
        # 获取所有柱子的高度（即频数）和 Y 轴起始位置
        heights = [p.get_height() for p in patches]

        # 寻找最高柱索引
        max_idx = np.argmax(heights)
        max_height = heights[max_idx]

        # 获取最高柱的矩形对象
        max_patch = patches[max_idx]

        # 计算最高柱的中心 X 轴坐标
        x_start = max_patch.get_x()
        width = max_patch.get_width()
        x_center = x_start + width / 2

        # 绘制文本标签
        # np.mean(max_height) 防止 heights 列表出现 nan
        plt.text(
            x=x_center,  # 中心点坐标
            y=max_height + 10,  # 在柱子上方 10 个单位处绘制
            s=f'{int(max_height)}',  # 显示频数数值
            ha='center',  # 水平居中
            va='bottom',  # 垂直靠下
            fontsize=12,
            fontweight='bold',
            color='darkgreen'  # 使用特定的深绿色
        )

    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='完美预测线 (Error=0)')

    plt.xlabel('预测误差 (残差)', fontsize=12)
    plt.ylabel('样本频数', fontsize=12)
    plt.title(f'{model_name} - 残差分布诊断图', fontsize=14)
    plt.legend()

    save_path = os.path.join(save_dir, f"{model_name}_residuals.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plot] 残差图已保存至: {save_path}")

def plot_feature_importance(model, feature_names, model_name, save_dir, top_n=10):
    """
    绘制特征重要性热力图
    """
    os.makedirs(save_dir, exist_ok=True)

    # 提取重要性并排序
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # 兼容没有直接 feature_importances_ 的模型（如线性回归系数）
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[indices]
    top_values = importances[indices].reshape(-1, 1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        top_values,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        yticklabels=top_features,
        xticklabels=["贡献权重"]
    )
    plt.title(f'{model_name} - 前 {top_n} 核心特征权重', fontsize=14)

    save_path = os.path.join(save_dir, f"{model_name}_feature_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plot] 特征热力图已保存至: {save_path}")


def plot_multi_model_comparison(y_true, predictions_dict, save_dir, sample_size=60):
    """
    多模型同框对比曲线图
    :param predictions_dict: 格式为 {"rf": preds, "xgb": preds, ...}
    """
    os.makedirs(save_dir, exist_ok=True)
    y_true_sub = np.array(y_true)[:sample_size]
    x = np.arange(len(y_true_sub))

    plt.figure(figsize=(16, 7))
    # 画真实值（黑色基准）
    plt.plot(x, y_true_sub, 'k-o', label='真实产量', lw=3, zorder=3)

    # 循环画出参与对比的所有模型
    styles = ['--', '-.', ':', '-']
    colors = ['crimson', 'royalblue', 'orange', 'seagreen']

    for i, (name, preds) in enumerate(predictions_dict.items()):
        p_sub = np.array(preds)[:sample_size]
        plt.plot(x, p_sub, label=f'{name.upper()} 预测',
                 linestyle=styles[i % len(styles)],
                 color=colors[i % len(colors)], lw=2)

    plt.title(f"多算法拟合能力横向评测 (前 {sample_size} 样本)", fontsize=16)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3)

    plt.savefig(f"{save_dir}/multi_model_comparison.png", dpi=300)
    plt.close()


def plot_actual_vs_predicted_scatter(y_true, y_pred, model_name, save_dir):
    """
    绘制 真实值 vs 预测值 的散点对角线图 (包含拟合趋势线)
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))

    # 1. 绘制基础散点
    # 使用较小的透明度 (alpha)，以便看清点的密集程度
    plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue', edgecolors='none', label='预测样本')

    # 2. 计算并绘制 完美对角线 (Y = X)
    # 找到真实值和预测值中的最大最小值，作为对角线的两端
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    # 为了图表美观，给两端留出 5% 的余量
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin],
             [min_val - margin, max_val + margin],
             color='red', linestyle='--', linewidth=2, label='完美预测线 (Y = X)')

    # 3. 计算并绘制 实际拟合趋势线
    # 用多项式拟合一条直线，看看模型整体的预测趋势是不是偏离了对角线
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)

    # 生成一条平滑的线
    x_trend = np.linspace(min_val, max_val, 100)
    plt.plot(x_trend, p(x_trend), color='darkorange', linewidth=2, label=f'实际拟合趋势线')

    # 4. 图表修饰
    plt.xlabel('真实蓝莓产量 (Actual Yield)', fontsize=12)
    plt.ylabel('模型预测产量 (Predicted Yield)', fontsize=12)
    plt.title(f'{model_name} - 真实值 vs 预测值 散点诊断图', fontsize=14, pad=15)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)

    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.5)

    # 保存图片
    save_path = os.path.join(save_dir, f"{model_name}_actual_vs_predicted.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plot] 散点诊断图已保存至: {save_path}")


def plot_error_cdf_comparison(y_true, predictions_dict, save_dir):
    """
    绘制多模型绝对误差累积分布函数图 (Error CDF)
    用于直观对比多个模型的整体预测精度和稳定性。
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))

    y_true = np.array(y_true)

    # 定义统一的配色方案
    palette = {'LGBM': 'seagreen', 'RF': 'orange', 'XGB': 'crimson'}
    line_styles = {'LGBM': '-', 'RF': '--', 'XGB': '-.'}

    for m_name, m_preds in predictions_dict.items():
        # 1. 计算绝对误差
        abs_errors = np.abs(y_true - np.array(m_preds))

        # 2. 对绝对误差进行排序
        sorted_errors = np.sort(abs_errors)

        # 3. 计算累积概率 (CDF)
        # 例如，第 i 个排序后的误差，它覆盖了 i / N 的样本比例
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        m_name_upper = m_name.upper()
        plt.plot(sorted_errors, cdf,
                 label=f'{m_name_upper}',
                 color=palette.get(m_name_upper, 'blue'),
                 linestyle=line_styles.get(m_name_upper, '-'),
                 linewidth=2.5)

    # 4. 图表修饰
    plt.xlabel('绝对预测误差 (Absolute Error)', fontsize=12)
    plt.ylabel('累积样本比例 (Cumulative Proportion)', fontsize=12)
    plt.title('多模型绝对误差累积分布对比 (Error CDF)', fontsize=14, pad=15)

    # 设置网格，方便读取数值
    plt.grid(True, linestyle=':', alpha=0.7)

    # 限制 X 轴的显示范围，去掉极端的长尾数据，只看核心对比区
    # 假设你的数据集中，95%的误差都在 1500 以内，我们可以将 X 轴限制在 2000
    # 这里通过所有模型误差的 95 分位数来动态设置 X 轴上限
    all_errors = np.concatenate([np.abs(y_true - np.array(preds)) for preds in predictions_dict.values()])
    x_max = np.percentile(all_errors, 95)
    plt.xlim(0, x_max * 1.2)  # 留一点余量
    plt.ylim(0, 1.05)

    # 添加图例和说明
    plt.legend(loc='lower right', fontsize=11, title='曲线越靠左上方，模型越优秀')

    # 保存图片
    save_path = os.path.join(save_dir, "multi_model_error_cdf.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plot] 误差CDF对比图已保存至: {save_path}")
