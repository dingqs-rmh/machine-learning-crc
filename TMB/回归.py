import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取数据
train_data = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB_2.csv')
test_data = pd.read_csv('/home/xkj/project/TMB/data/READ_TMB_2.csv')

# 对数变换（log2(TPM + 1)）
train_data.iloc[:, 1:] = np.log2(train_data.iloc[:, 1:] + 1)
test_data.iloc[:, 1:] = np.log2(test_data.iloc[:, 1:] + 1)

# 分离特征和目标变量
X_train = train_data.iloc[:, 1:]  # 所有基因表达数据
y_train = train_data.iloc[:, 0]   # TMB值

X_test = test_data.iloc[:, 1:]    # 所有基因表达数据
y_test = test_data.iloc[:, 0]     # TMB值

# 数据标准化（z-score）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 特征选择 - 使用LASSO回归选择特征
lasso = LassoCV(cv=5)  # 使用5折交叉验证选择最佳正则化参数
lasso.fit(X_train_scaled, y_train)

# 选择非零系数的特征
selected_features = X_train.columns[(lasso.coef_ != 0)]
X_train_selected = X_train_scaled[:, lasso.coef_ != 0]
X_test_selected = X_test_scaled[:, lasso.coef_ != 0]

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train_selected, y_train)

# 进行预测
y_pred = model.predict(X_test_selected)

# 评估模型表现
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'选择的特征数: {len(selected_features)}')
print(f'均方误差 (MSE): {mse:.2f}')
print(f'决定系数 (R²): {r2:.2f}')

