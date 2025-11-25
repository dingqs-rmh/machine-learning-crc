import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# 读取数据
coad_data = pd.read_csv('/home/xkj/project/KRAS/data/COAD_KRAS.csv')
read_data = pd.read_csv('/home/xkj/project/KRAS/data/READ_KRAS.csv')

# 合并数据集
data = pd.concat([coad_data, read_data], axis=0)

# 提取特征名
with open('/home/xkj/project/KRAS/model/selected_features.txt', 'r') as f:
    selected_features = f.read().splitlines()

# 保留目标列和选定的特征
data_filtered = data[['Target'] + selected_features]

# 更新X和y
y = data_filtered.iloc[:, 0].astype(int)
X = data_filtered.iloc[:, 1:]

# 对数变换
X = np.log2(X + 1)

# 创建目录保存模型和文件
model_dir = '/home/xkj/project/KRAS/model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 保存标准化器为pkl格式
scaler_save_path = f'{model_dir}scaler.pkl'
joblib.dump(scaler, scaler_save_path)
print(f'标准化模型已保存为: {scaler_save_path}')

# SMOTE进行数据平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 定义基本分类器和元分类器
base_estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),
    ('lgb', LGBMClassifier(random_state=42))
]

# 使用堆叠法创建分类器
stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())

# 训练堆叠模型
stacking_model.fit(X_resampled, y_resampled)

# 保存堆叠模型为pkl格式
model_save_path = f'{model_dir}stacking_model.pkl'
joblib.dump(stacking_model, model_save_path)
print(f'堆叠模型已保存为: {model_save_path}')
