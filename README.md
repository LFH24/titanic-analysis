# Titanic：从 EDA 到模型解释的二分类分析

基于 Kaggle Titanic 数据集，走了一遍完整的数据分析流程，包括数据清洗、探索性分析、特征工程、建模和模型解释。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-0.76555-orange)](https://www.kaggle.com/competitions/titanic)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 主要发现

891 名乘客的数据里，影响幸存率最大的几个因素：

- **性别** — 女性幸存率 74%，男性 19%。"妇女儿童优先"在数据上有实打实的体现
- **舱位等级** — 头等舱 63%，三等舱 24%。舱位等级和生存机会之间的关系很直接
- **年龄** — 12 岁以下的儿童幸存率明显高于成年人，但成年男性各年龄段差异不大
- **家庭规模** — 同行 2 到 4 人的乘客存活率最高。独自旅行的人缺少互助，而大家庭（5 人以上）在混乱中很难统一行动

一个具体的数字：成年男性、三等舱乘客，幸存率不到 10%。

---

## 分析流程

```
第 1 章 · 问题定义    →  明确二分类任务、评估指标、从业务角度切入
第 2 章 · EDA         →  缺失值矩阵 → 单变量分析（假设→验证） → 交叉分析
第 3 章 · 缺失值处理   →  Age（分组中位数）/ Cabin（提取甲板） / Embarked / Fare
第 4 章 · 特征工程     →  Title / FamilySize / IsAlone / AgeBin / FareBin / TicketGroup
第 5 章 · 建模对比     →  5 模型 baseline → GridSearchCV → VotingClassifier 集成
第 6 章 · 模型解释     →  特征重要性 + SHAP + 业务视角结论
第 7 章 · 分析总结     →  关键结论回顾 + 改进方向
```

每一步都解释了"为什么这么做"。比起准确率数字，我更在意分析逻辑是否经得起推敲。

---

## 技术栈

Python、pandas、numpy、scikit-learn、XGBoost、VotingClassifier、matplotlib、seaborn、SHAP、missingno

---

## Kaggle 提交结果

| 指标 | 结果 |
|------|------|
| Accuracy | **0.76555** |
| 模型 | XGBoost（StratifiedKFold + 正则化） |
| 特征数 | 12 + Sex×Pclass 交互项 |
| CV Accuracy | 0.8507 |

---

## 复现

```bash
# 安装依赖
pip install -r requirements.txt

# 打开 Notebook
jupyter notebook titanic-analysis.ipynb

# Kernel → Restart & Run All
```

---

## 文件结构

```
titanic-analysis/
├── README.md
├── titanic-analysis.ipynb         # 完整分析过程（7 章）
├── requirements.txt
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

LFH24 · 2026.07
