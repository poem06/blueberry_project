
# 蓝莓产量预测系统 (Blueberry Yield Prediction)

这是一个基于机器学习的结构化数据回归预测项目，旨在通过环境参数、授粉蜜蜂密度等特征，精准预测蓝莓的最终产量。

本项目采用**高度工程化、配置驱动**的设计理念，集成了 Random Forest、XGBoost 和 LightGBM 三大主流树模型算法。实现了从数据清洗、特征工程、多模型超参数自动化搜索（打擂台）、到多维可视化诊断与最优模型自动固化的全流程端到端管线。

---

## 核心特性 (Key Features)

- **纯配置驱动 (Config-Driven)**：将超参数搜索空间、数据路径、对比模型列表统一收敛至 `config.yaml`。无需修改任何代码，即可实现模型的一键切换与参数调节。
- *自动化模型对比 (Auto-Model Selection)**：通过 `$R^2$` Score 等综合指标，自动在多个算法间进行横向评测，并将得分最高的“最优模型”自动保存为 `best_model.pkl`。
- **多维深度可视化 (Advanced Visualization)**：超越单一的评价指标，系统会自动生成丰富的对比图表（存放在 `plots/` 目录下）：
  - **残差频数对比直方图** & **核密度曲线图**（诊断局部精度）
  - **真实值 vs 预测值 散点对角线图**（诊断系统性偏差与异方差性）
  - **误差累积分布图 (Error CDF)**（直观对比模型整体稳定性）
  - **特征重要性热力图**（增强模型可解释性）
- **统一进行特征工程**：确保训练集与测试集经过 100% 严格对齐的特征转换（如衍生 `total_bees` 特征），杜绝数据穿越和特征错位。

---

## 目录结构 (Project Structure)

```text
.
├── config/
│   └── config.yaml          # 全局核心配置文件（调参、路径、随机种子等）
├── datasets/
│   ├── raw/                 # 原始数据集 (train.csv, test.csv)
│   └── processed/           # 预处理后的中间数据集
├── plots/                   # 保存模型可视化诊断图表
├── saved_models/
│   └── best_model.pkl       # 保存的最优模型
├── src/                     # 核心源代码目录
│   ├── models/              # 各类算法的具体实现与调参封装
│   │   ├── rf_model.py      # 随机森林
│   │   ├── xgb_model.py     # XGBoost
│   │   └── lgbm_model.py    # LightGBM
│   ├── data_loader.py       # 数据读取与统一特征工程
│   ├── evaluator.py         # 核心评估指标计算 (MSE, MAE, R2等)
│   └── visualizer.py        # 基于 seaborn/matplotlib 的多维可视化引擎
├── main.py                  # 主程序：执行训练、评估、PK 与绘图全流程
├── predict.py               # 预测脚本：加载最佳模型对新数据生成预测
├── app.py                   # 回归预测终端交互式应用入口
└── requirements.txt         # 运行环境依赖包列表
```

------

## 快速开始 (Quick Start)

### 1. 环境准备

确保你的环境中已安装 Python 3.8+。建议使用 Anaconda 创建独立的虚拟环境。

Bash

```
# 安装所需依赖
pip install -r requirements.txt
```

### 2. 准备数据

请将训练集和测试集文件放入 `datasets/raw/` 目录下，并确保文件名为 `train.csv` 和 `test.csv`。

### 3. 一键训练与模型对比

运行主程序。系统将读取 `config.yaml`，依次训练配置列表中的所有模型，进行交叉验证与超参数寻优，最终在终端打印出对比报告，并保存最优模型。

Bash

```
python main.py
```

> **提示**：运行结束后，请前往 `plots/` 目录查看保存的模型对比图表

### 4. 预测新数据

使用刚才训练出的“最优模型”，对未知的测试集进行预测，并生成最终的提交文件：

Bash

```
python predict.py
```

执行完毕后，项目根目录将生成包含预测结果的 `final_submission.csv` 文件。

------

## 配置说明 (Configuration Guide)

项目的所有核心行为均由 `config/config.yaml` 控制。你可以自由调整以下内容以探索更高的模型精度：

- **`compare_models`**: 增删你想对比的模型（如 `['rf', 'xgb', 'lgbm']`）。
- **`params`**: 为不同模型量身定制非对称的超参数搜索空间。
- **`search_params`**: 调整 `n_iter`（随机搜索迭代次数）和 `cv`（交叉验证折数）来平衡寻找最优解的概率与计算时间。

------
