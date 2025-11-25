import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import shap

# 读取数据
train_data = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB.csv')
test_data = pd.read_csv('/home/xkj/project/TMB/data/READ_TMB.csv')

# 提取特征名
feature_names = train_data.columns[1:]

# 对数变换（log2(TPM + 1)）
train_data.iloc[:, 1:] = np.log2(train_data.iloc[:, 1:] + 1)
test_data.iloc[:, 1:] = np.log2(test_data.iloc[:, 1:] + 1)

# 设置阈值，将TMB分类为高（1）或低（0）
threshold = 10
y_train = (train_data.iloc[:, 0] > threshold).astype(int)  # 高TMB为1，低TMB为0
y_test = (test_data.iloc[:, 0] > threshold).astype(int)

# 分离特征
X_train = train_data.iloc[:, 1:]  # 所有基因表达数据
X_test = test_data.iloc[:, 1:]    # 所有基因表达数据

# 数据标准化（z-score）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用LASSO进行特征选择
lasso = LassoCV(cv=5, random_state=42)  # 使用5折交叉验证选择最佳正则化参数
lasso.fit(X_train_scaled, y_train)

# 选择非零系数的特征
selected_features = np.where(lasso.coef_ != 0)[0]
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# 获取选择后的特征名称
selected_feature_names = feature_names[selected_features]

# 创建SMOTE实例
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# 调整XGBoost分类器的超参数
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [1, 2, 5]  # 对不平衡数据进行权重调整
}

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_

# 进行预测
y_pred = best_model.predict(X_test_selected)
y_pred_prob = best_model.predict_proba(X_test_selected)[:, 1]

# 评估模型表现
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f'准确率: {accuracy:.2f}')
print(f'AUC-ROC: {roc_auc:.2f}')
print('分类报告:\n', classification_report(y_test, y_pred))

# 绘制改进的ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='teal', lw=2)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.title('ROC curve of XGBoost')
plt.text(0.6, 0.2, f'AUC in test set: {roc_auc:.3f}', fontsize=12)
plt.savefig('/home/xkj/project/TMB/plots/roc_curve.pdf')
plt.close()

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('/home/xkj/project/TMB/plots/confusion_matrix.pdf')
plt.close()

# 使用SHAP解释模型
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_selected)

# 绘制SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, show=False)
plt.savefig('/home/xkj/project/TMB/plots/shap_summary_plot.pdf')
plt.close()
