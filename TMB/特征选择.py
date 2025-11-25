import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 加载数据
coad = pd.read_csv('data/COAD_TMB.csv')
read = pd.read_csv('data/READ_TMB.csv')

# 合并数据集
data = pd.concat([coad, read], axis=0)

# 创建二分类标签
data['High_TMB'] = np.where(data.iloc[:, 0] > 10, 1, 0)
X = data.iloc[:, 1:-1]  # 基因表达量特征
y = data['High_TMB']

# 处理缺失值（示例：删除缺失样本）
#X = X.dropna()
#y = y.loc[X.index]

# 标准化特征（用于需要标准化的模型）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def select_features(model, X, y, top_n=50):
    model.fit(X, y)
    if isinstance(model, LogisticRegression):
        importances = np.abs(model.coef_[0])
    else:
        importances = model.feature_importances_
    features = pd.Series(importances, index=X.columns)
    return features.nlargest(top_n).index.tolist()

# 随机森林（无需标准化）
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_features = select_features(rf, X, y)

# Lasso回归（需要标准化）
lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', class_weight='balanced')
lasso_features = select_features(lasso, X_scaled, y)

# XGBoost（无需标准化）
xgb = XGBClassifier(n_estimators=100, scale_pos_weight=np.sum(y==0)/np.sum(y==1))
xgb_features = select_features(xgb, X, y)

# 计算交集
common_features = list(
    set(rf_features) & 
    set(lasso_features) & 
    set(xgb_features)
)
print("随机森林特征基因:", rf_features)
print("lasso特征基因:", lasso_features)
print("XGB特征基因:", xgb_features)
print("交集特征基因:", common_features)