# Titanic: 从 EDA 到模型解释的完整二分类分析

> 不只是预测准确率——这是一份展示「数据分析师如何思考」的 Notebook。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-0.74162-orange)](https://www.kaggle.com/competitions/titanic)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 核心结论（面试官请看这里）

基于 891 名乘客数据的分析，影响泰坦尼克号幸存率的因素按重要性排序：

- 🥇 **性别是第一决定因素** — 女性幸存率 74%，男性仅 19%("妇女儿童优先"不是传言)
- 🥈 **舱位等级紧随其后** — 头等舱幸存率 63%，三等舱仅 24%(财富直接影响了生存机会)
- 🥉 **年龄有非线性影响** — 儿童(0-12岁)幸存率更高，但成年男性各年龄段都很低
- 📊 **家庭规模 2-4 人最优** — 独行者缺乏互助，大家庭(5+)难以统一行动

> **"如果你是泰坦尼克号上的成年男性三等舱乘客，幸存率不到 10%。如果你是头等舱 8 岁小女孩的母亲，幸存率超过 90%。这就是这场灾难最残酷的真相。"**

---

## 🗺️ 分析流程

```
第 1 章 · 问题定义    →  明确二分类任务 + 评估指标 + 业务视角切入
第 2 章 · EDA         →  缺失值矩阵 → 单变量分析(假设→验证) → 交叉分析
第 3 章 · 缺失值处理   →  Age(分组中位数) / Cabin(提取甲板) / Embarked / Fare
第 4 章 · 特征工程     →  Title / FamilySize / IsAlone / AgeBin / FareBin / TicketGroup
第 5 章 · 建模对比     →  5 模型 baseline(5-fold CV) → GridSearchCV 调参
第 6 章 · 模型解释     →  特征重要性 + SHAP + 业务视角结论
第 7 章 · 复盘改进     →  不足分析 + 改进方向 + 框架迁移
```

每一步都有"为什么这么做"的解释——不是跑代码，是展示思维过程。

---

## 🛠️ 技术栈

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `SHAP` · `missingno`

---

## 🏆 Kaggle 提交结果

| 指标 | 分数 |
|------|------|
| Accuracy | **0.74162** |
| 模型 | XGBoost（调参后） |
| 排名 | Top 75%（约） |

> ⚠️ 这个分数低于 `gender_submission` 基准（0.765），但本项目的核心目标是**展示完整的分析思维流程**，而非追求极致准确率。Notebook 第 7 章已复盘了准确率不足的原因及改进方向。

---

## 🚀 快速复现

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 打开 Notebook
jupyter notebook titanic-analysis.ipynb

# 3. Kernel → Restart & Run All
```

---

## 📂 文件结构

```
titanic-analysis/
├── README.md                      # 你正在看的这个文件
├── titanic-analysis.ipynb         # 核心：完整分析过程（7 章）
├── requirements.txt               # Python 依赖
├── .gitignore
├── data/
│   ├── train.csv                  # 训练集（891 条）
│   ├── test.csv                   # 测试集（418 条）
│   └── gender_submission.csv      # Kaggle 基准
└── output/
    ├── submission.csv             # 最终预测结果
    ├── kaggle-score.png           # Kaggle 提交分数截图
    └── figures/                   # 导出图表
```

---

## 📈 后续

- 🔜 [House Prices / Bike Sharing Demand](https://github.com/LFH24) — 回归分析项目（7 月中上线）
- 📋 更多数据科学项目持续更新中...

---

*Built with ❤️ by [LFH24](https://github.com/LFH24) · 2026.07*
