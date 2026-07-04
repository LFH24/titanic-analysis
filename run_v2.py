"""Titanic v2 prediction — 独立运行脚本，生成 submission.csv"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ==================== 加载数据 ====================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train_len = len(train)

# ==================== Age 缺失标记 ====================
train['AgeMissing'] = train['Age'].isnull().astype(int)
test['AgeMissing'] = test['Age'].isnull().astype(int)

# ==================== Title 提取 ====================
def extract_title(name):
    match = re.search(r',\s*([^\.]+)\.', name)
    return match.group(1).strip() if match else 'Other'

all_data = pd.concat([train, test], axis=0, sort=False)
all_data['Title'] = all_data['Name'].apply(extract_title)

title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Royal', 'Sir': 'Royal', 'the Countess': 'Royal',
    'Jonkheer': 'Royal', 'Don': 'Royal', 'Dona': 'Royal',
    'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Dr': 'Officer', 'Rev': 'Officer'
}
all_data['Title'] = all_data['Title'].replace(title_map)

# Age 填充（Title 分组中位数）
for title in all_data['Title'].unique():
    median_age = all_data.loc[all_data['Title'] == title, 'Age'].median()
    mask = (all_data['Title'] == title) & (all_data['Age'].isnull())
    all_data.loc[mask, 'Age'] = median_age

# ==================== Embarked & Fare 填充 ====================
all_data['Embarked'] = all_data['Embarked'].fillna('S')
fare_median = all_data.loc[all_data['Pclass'] == 3, 'Fare'].median()
all_data.loc[all_data['Fare'].isnull(), 'Fare'] = fare_median

# ==================== Cabin 特征 ====================
all_data['HasCabin'] = all_data['Cabin'].notnull().astype(int)

# ==================== FamilySize & IsAlone ====================
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# ==================== TicketGroup ====================
ticket_counts = all_data['Ticket'].value_counts()
all_data['TicketGroup'] = all_data['Ticket'].map(ticket_counts)

# ==================== 数据准备 ====================
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'HasCabin', 'TicketGroup']

le_cols = ['Sex', 'Embarked', 'Title']
for col in le_cols:
    all_data[col] = LabelEncoder().fit_transform(all_data[col].astype(str))

train_processed = all_data[:train_len].copy()
test_processed = all_data[train_len:].copy()

# Sex × Pclass 交互特征
train_processed['Sex_Pclass'] = train_processed['Sex'] * train_processed['Pclass']
test_processed['Sex_Pclass'] = test_processed['Sex'] * test_processed['Pclass']
features.append('Sex_Pclass')

X = train_processed[features]
y = train_processed['Survived'].astype(int)
X_test_final = test_processed[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f'特征: {X.shape[1]} 维, 样本: {X.shape[0]} 条')

# ==================== GridSearchCV 调参 ====================
print('\n=== Random Forest 调参 ===')
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, None],
     'min_samples_split': [5, 10, 15], 'min_samples_leaf': [2, 4, 6],
     'max_features': ['sqrt', 'log2']},
    cv=skf, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_scaled, y)
print(f'RF 最优 CV: {rf_grid.best_score_:.4f} | 参数: {rf_grid.best_params_}')

print('\n=== XGBoost 调参 ===')
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5],
     'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 1.0],
     'reg_alpha': [0, 0.1, 1], 'reg_lambda': [1, 2, 5],
     'min_child_weight': [1, 3, 5]},
    cv=skf, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_scaled, y)
print(f'XGB 最优 CV: {xgb_grid.best_score_:.4f} | 参数: {xgb_grid.best_params_}')

print('\n=== SVM 调参 ===')
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1], 'kernel': ['rbf']},
    cv=skf, scoring='accuracy', n_jobs=-1
)
svm_grid.fit(X_scaled, y)
print(f'SVM 最优 CV: {svm_grid.best_score_:.4f} | 参数: {svm_grid.best_params_}')

print('\n=== Logistic Regression 调参 ===')
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2']},
    cv=skf, scoring='accuracy'
)
lr_grid.fit(X_scaled, y)
print(f'LR 最优 CV: {lr_grid.best_score_:.4f} | 参数: {lr_grid.best_params_}')

# ==================== VotingClassifier ====================
print('\n=== VotingClassifier 集成 ===')
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_grid.best_estimator_),
        ('rf', rf_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_),
        ('lr', lr_grid.best_estimator_),
    ],
    voting='soft'
)

ensemble_scores = cross_val_score(voting_clf, X_scaled, y, cv=skf, scoring='accuracy')
print(f'VotingClassifier CV: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})')

best_single = max(rf_grid.best_score_, xgb_grid.best_score_, svm_grid.best_score_, lr_grid.best_score_)
print(f'最佳单一模型 CV: {best_single:.4f}')
print(f'集成提升: {(ensemble_scores.mean() - best_single)*100:.1f} 个百分点')

# ==================== 生成提交文件 ====================
voting_clf.fit(X_scaled, y)
predictions = voting_clf.predict(X_test_scaled)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('output/submission.csv', index=False)

print(f'\n=== 提交文件已保存 ===')
print(f'预测幸存: {submission["Survived"].sum()} 人 ({submission["Survived"].mean()*100:.1f}%)')
print(f'预测遇难: {len(submission) - submission["Survived"].sum()} 人')
print(f'训练集幸存率: {train["Survived"].mean()*100:.1f}%')
print(f'文件: output/submission.csv ({len(submission)} 条)')
