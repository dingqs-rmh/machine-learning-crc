import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from scipy.stats import ttest_ind

# 设置全局字体为“Times New Roman”，并调整默认字体大小和加粗选项
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 全局字体大小
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗

# 读取数据
coad_data = pd.read_csv('/home/xkj/project/PIK3CA/data/COAD_PIK3CA.csv')
read_data = pd.read_csv('/home/xkj/project/PIK3CA/data/READ_PIK3CA.csv')

# 合并数据集
data = pd.concat([coad_data, read_data], axis=0)

# 提取特征名
feature_names = data.columns[1:]

# 对数变换
data.iloc[:, 1:] = np.log2(data.iloc[:, 1:] + 1)

# 定义组别
group_1 = data[data.iloc[:, 0] == 1]
group_0 = data[data.iloc[:, 0] == 0]

# 提取基因表达数据
X_group_1 = group_1.iloc[:, 1:]
X_group_0 = group_0.iloc[:, 1:]

# 执行t检验，筛选出p值小于阈值（如0.05）的基因
p_values = []
for col in feature_names:
    stat, p_value = ttest_ind(X_group_1[col], X_group_0[col], equal_var=False)
    p_values.append(p_value)

# 转换为DataFrame，添加基因名
p_values_df = pd.DataFrame({
    'Gene': feature_names,
    'P_Value': p_values
})

# 选择p值小于0.05的基因
selected_genes = p_values_df[p_values_df['P_Value'] < 0.05]['Gene'].values
print(f'筛选出的差异基因数: {len(selected_genes)}')

# 只保留筛选出的差异基因数据
data_filtered = data[['Target'] + list(selected_genes)]

# 更新X和y
y = data_filtered.iloc[:, 0].astype(int)
X = data_filtered.iloc[:, 1:]

# LASSO特征选择
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# 选择非零系数的特征，并从原始X中获取特征名称
selected_features = np.where(lasso.coef_ != 0)[0]
selected_feature_names = X.columns[selected_features]  # 使用原始特征名
X_selected = X[selected_feature_names]  # 用原始特征数据

# 使用RFE选择最重要的特征
rfe_model = RandomForestClassifier(random_state=2024)
rfe = RFE(estimator=rfe_model, n_features_to_select=50)
X_rfe_selected = rfe.fit_transform(X_selected, y)

# 保留RFE选择的特征名
rfe_selected_feature_names = selected_feature_names[rfe.support_]  # 使用 RFE 选择的特征名
X_selected_final = X_selected[rfe_selected_feature_names]  # 从原始数据中选择这些特征

# 标准化选择的特征
scaler_selected = StandardScaler()
X_selected_scaled = scaler_selected.fit_transform(X_selected_final)

# 创建SMOTE实例
smote = SMOTE(random_state=42)
# SMOTE
X_resampled, y_resampled = smote.fit_resample(X_selected_scaled, y)

# 训练随机森林
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_resampled, y_resampled)



# 保存模型和选择的特征
model_output_path = '/home/xkj/project/PIK3CA/model/best_random_forest_model.pkl'
features_output_path = '/home/xkj/project/PIK3CA/model/selected_features.csv'
scaler_output_path = '/home/xkj/project/PIK3CA/model/scaler_selected.pkl'
shap_pdf_path = '/home/xkj/project/PIK3CA/model/shap_summary_plot.pdf'

joblib.dump(best_model, model_output_path)
selected_features_df = pd.DataFrame(rfe_selected_feature_names, columns=['Selected Features'])
selected_features_df.to_csv(features_output_path, index=False)

# 保存标准化模型
joblib.dump(scaler_selected, scaler_output_path)

# 计算SHAP值并保存
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_selected_scaled)

# 生成SHAP summary plot并保存为PDF
with PdfPages(shap_pdf_path) as pdf:
    plt.figure()
    shap.summary_plot(shap_values[1], X_selected_scaled, feature_names=rfe_selected_feature_names, show=False)
    plt.title("SHAP Summary Plot")
    pdf.savefig()  # 保存当前图
    plt.close()

print(f"模型已保存到：{model_output_path}")
print(f"选择的特征已保存到：{features_output_path}")
print(f"标准化模型已保存到：{scaler_output_path}")
print(f"SHAP summary plot已保存到：{shap_pdf_path}")
