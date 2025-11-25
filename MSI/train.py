import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import seaborn as sns
import os
import joblib
import shutil
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import (SelectKBest, RFE, SelectFromModel, SequentialFeatureSelector,
                                       VarianceThreshold, SelectPercentile, mutual_info_classif, GenericUnivariateSelect,
                                       f_classif)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import *
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, accuracy_score, recall_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from fancyimpute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ttest_ind
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay
import scipy.stats as stats
from collections import Counter
import glob
import statsmodels.api as sm
import statsmodels.stats.multitest as multitest
from scipy.stats import pearsonr, mannwhitneyu, ranksums, ttest_ind
from sklearn import svm
import multiprocessing as mlp
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import VotingClassifier
from preprocess import preprocess_tpm_data
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression

# Set font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def feature_selection_lasso(X, y, alpha=0.01):
    """
    使用Lasso回归进行特征选择。

    参数:
    X (pd.DataFrame): 特征DataFrame。
    y (pd.Series): 标签Series。
    alpha (float): Lasso正则化参数。

    返回:
    tuple: 选择后的特征DataFrame和非零系数的特征名称。
    """

    # 使用Lasso训练模型
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)

    # 获取非零系数的特征
    selected_features = X.columns[lasso.coef_ != 0]
    importance = np.abs(lasso.coef_[lasso.coef_ != 0])
    
    X_selected = X[selected_features]

    return X_selected, selected_features, importance

def plot_lasso_feature_importance(feature_names, importances, output_directory):
    """
    绘制特征重要性图并保存为PDF，显示前30个特征，并使用指定颜色。
    
    参数:
    feature_names (list): 特征名称列表。
    importances (list): 特征重要性列表。
    output_directory (str): 保存PDF的目录。
    """

    # 创建一个DataFrame来存储特征重要性
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # 根据重要性排序，显示前30个特征
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(30)

    # 动态调整图形尺寸，设置每个特征行的高度
    num_features = len(feature_importance_df)
    plt.figure(figsize=(10, num_features * 0.25))

    plt.title("Top 30 Lasso Gene Importances")
    plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], align="center", color='#78c2ad')
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
    plt.gca().invert_yaxis()  # Invert the y-axis to have the highest importance at the top
    plt.xlabel('Importance')
    plt.ylabel('Genes')
    plt.tight_layout()

    # 保存图像为PDF
    pdf_path = os.path.join(output_directory, "lasso_top30_gene_importances.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig()
        plt.close()

    print(f"Feature importance plot saved as {pdf_path}")
# Set working directory
os.chdir('/home/xkj/project/msi/')
output_directory = "/home/xkj/project/msi/train_output/"
os.makedirs(output_directory, exist_ok=True)

model_directory = "/home/xkj/project/msi/model/"
os.makedirs(model_directory, exist_ok=True)

csv_files = glob.glob('data/TCGA_COAD_clinical_msi.csv')

# 初始化一个空的DataFrame
data = pd.DataFrame()

# 定义缺失值的表示形式
na_values = ['', ' ', 'NA']

# 遍历所有CSV文件并合并
for csv_file in csv_files:
    temp_data = pd.read_csv(csv_file, index_col=0, na_values=na_values)
    data = pd.concat([data, temp_data], ignore_index=True)

# Split labels and features
X = data.iloc[:, 1:]  # Features
y = data.iloc[:, 0]  # Labels

# 填充缺失值
missing_value_strategy='mean'
imputer = SimpleImputer(strategy=missing_value_strategy)
X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X = X_filled

# 过滤低表达基因
min_tpm_threshold=1
filtered_X = X.loc[:, (X > min_tpm_threshold).sum() > (X.shape[0] * 0.2)]
X = filtered_X

# 将样本分成两组
msi_h_group = X[y == 1]
mss_group = X[y == 0]

print(f"MSI-H group size: {msi_h_group.shape[0]}")
print(f"MSS group size: {mss_group.shape[0]}")

p_values = []
gene_names = X.columns

# 对每个基因进行t检验
for gene in gene_names:
    t_stat, p_val = stats.ttest_ind(msi_h_group[gene], mss_group[gene])
    p_values.append(p_val)

# 打印前10个p值以进行检查
print("First 10 p-values:", p_values[:10])

# 检查p值中是否有nan或inf
p_values = [p for p in p_values if not (pd.isna(p) or p == float('inf') or p == float('-inf'))]

# 打印清理后的前10个p值
print("First 10 p-values after removing invalid values:", p_values[:10])

# 调整p值
if p_values:
    adjusted_p_values = multitest.multipletests(p_values, method='fdr_bh')[1]
else:
    adjusted_p_values = []

# 打印前10个调整后的p值以进行检查
print("First 10 adjusted p-values:", adjusted_p_values[:10])

# 创建结果DataFrame
results = pd.DataFrame({
    'gene': gene_names[:len(p_values)],  # 确保基因名称和p值长度匹配
    'p_value': p_values,
    'adjusted_p_value': adjusted_p_values
})

# 过滤显著差异表达的基因
significant_genes = results[results['adjusted_p_value'] < 0.01]['gene']

# 如果没有显著差异表达的基因，降低阈值或按百分比筛选
if significant_genes.empty:
    print("No significant genes found with adjusted p-value < 0.01. Trying a higher threshold...")
    significant_genes = results[results['adjusted_p_value'] < 0.05]['gene']
    if significant_genes.empty:
        print("No significant genes found with adjusted p-value < 0.05. Selecting top 5% of genes based on p-value...")
        top_percent = 0.05
        top_n = int(len(results) * top_percent)
        significant_genes = results.nsmallest(top_n, 'p_value')['gene']

# 筛选后的特征
X_selected = X[significant_genes]

# 预处理训练集特征数据，并保存预处理器
preprocessed_X_train = preprocess_tpm_data(X_selected)

# 特征选择
selected_X_train, selected_features, importances = feature_selection_lasso(preprocessed_X_train, y, alpha=0.01)

# 绘制特征重要性图并保存为PDF
plot_lasso_feature_importance(selected_features, importances, output_directory=output_directory)

# 保存选择的特征名称
with open(os.path.join(model_directory, 'selected_features.txt'), 'w') as f:
    for feature in selected_features:
        f.write("%s\n" % feature)


# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(selected_X_train, y)

# 定义基础模型和融合模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(random_state=42)
lgb_model = lgb.LGBMClassifier(random_state=42)
stacking_model = LogisticRegression()

from sklearn.model_selection import KFold, cross_val_predict
# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 保存每一折的性能指标
metrics = []

# 保存每一折的预测结果和实际标签
val_preds = []
val_labels = []
val_probas = []

for train_index, val_index in kf.split(X_resampled):
    X_train, X_val = X_resampled.iloc[train_index], X_resampled.iloc[val_index]
    y_train, y_val = y_resampled.iloc[train_index], y_resampled.iloc[val_index]
    
    # 训练基础模型
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    
    # 基础模型在验证集上的预测
    rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
    xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    
    # 将基础模型的预测结果组合为新的特征
    stacked_val_pred = pd.DataFrame({
        'rf': rf_val_pred,
        'xgb': xgb_val_pred,
        'lgb': lgb_val_pred
    })
    
    # 训练第二层模型
    stacking_model.fit(stacked_val_pred, y_val)
    
    # 第二层模型在验证集上的预测
    final_val_pred = stacking_model.predict(stacked_val_pred)
    final_val_proba = stacking_model.predict_proba(stacked_val_pred)[:, 1]
    
    val_probas.extend(final_val_proba)
    val_labels.extend(y_val)

    # 计算性能指标
    auc_score = roc_auc_score(y_val, final_val_proba)
    acc_score = accuracy_score(y_val, final_val_pred)
    recall = recall_score(y_val, final_val_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, final_val_pred).ravel()
    specificity = tn / (tn + fp)
    
    metrics.append((auc_score, acc_score, recall, specificity))

output_file_path = os.path.join(output_directory, 'performance_metrics.txt')

# 将性能指标写入txt文件
with open(output_file_path, 'w') as f:
    f.write('AUC\tACC\tSensitivity\tSpecificity\n')
    for metric in metrics:
        f.write(f"{metric[0]:.4f}\t{metric[1]:.4f}\t{metric[2]:.4f}\t{metric[3]:.4f}\n")

print(f"Performance metrics saved to {output_file_path}")
  
# 计算并绘制ROC曲线
fpr, tpr, _ = roc_curve(val_labels, val_probas)
roc_auc = auc(fpr, tpr)

pdf_path = os.path.join(output_directory, "roc_curve.pdf")

with PdfPages(pdf_path) as pdf:
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    pdf.savefig()
    plt.close()

print(f"ROC curve saved as {pdf_path}")



# 重新定义基础模型和融合模型
rf_model_final = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model_final = XGBClassifier(random_state=42)
lgb_model_final = lgb.LGBMClassifier(random_state=42)
# stacking_model_final = LogisticRegression(max_iter=1000)

# 训练基础模型
rf_model_final.fit(X_resampled, y_resampled)
xgb_model_final.fit(X_resampled, y_resampled)
lgb_model_final.fit(X_resampled, y_resampled)

# 基础模型在整个数据集上的预测
rf_full_pred = rf_model_final.predict_proba(X_resampled)[:, 1]
xgb_full_pred = xgb_model_final.predict_proba(X_resampled)[:, 1]
lgb_full_pred = lgb_model_final.predict_proba(X_resampled)[:, 1]

# 将基础模型的预测结果组合为新的特征
stacked_full_pred = pd.DataFrame({
    'rf': rf_full_pred,
    'xgb': xgb_full_pred,
    'lgb': lgb_full_pred
})

# 定义参数网格
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'max_iter': [100, 200, 500],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # 仅在penalty='elasticnet'时有用
}

# 过滤无效的参数组合
valid_param_grid = []
for params in ParameterGrid(param_grid):
    if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
        continue
    if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
        continue
    if params['penalty'] == 'l2' and params['solver'] not in ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag']:
        continue
    if params['penalty'] == 'none' and params['solver'] not in ['lbfgs', 'newton-cg', 'sag']:
        continue
    valid_param_grid.append(params)

# 确保每个参数的值都是列表
valid_param_grid = [
    {key: [value] if not isinstance(value, list) else value for key, value in params.items()}
    for params in valid_param_grid
]

# 创建 LogisticRegression 模型
logreg = LogisticRegression()

# 使用过滤后的参数网格进行网格搜索
grid_search = GridSearchCV(logreg, valid_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(stacked_full_pred, y_resampled)

# 打印最佳参数和最佳得分
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation AUC score: ", grid_search.best_score_)

# 使用最佳参数创建最终模型
best_logreg = grid_search.best_estimator_


# 保存最终模型
joblib.dump(rf_model_final, os.path.join(model_directory, 'rf_model.pkl'))
joblib.dump(xgb_model_final, os.path.join(model_directory, 'xgb_model.pkl'))
joblib.dump(lgb_model_final, os.path.join(model_directory, 'lgb_model.pkl'))
joblib.dump(best_logreg, os.path.join(model_directory, 'stacking_model.pkl'))

print("最终模型已保存。")

import shap
# 加载训练好的基础模型和元学习器
rf_model_final = joblib.load(os.path.join(model_directory, 'rf_model.pkl'))
xgb_model_final = joblib.load(os.path.join(model_directory, 'xgb_model.pkl'))
lgb_model_final = joblib.load(os.path.join(model_directory, 'lgb_model.pkl'))
best_stack = joblib.load(os.path.join(model_directory, 'stacking_model.pkl'))

# 基础模型在全数据集上的预测
rf_val_pred_full = rf_model_final.predict_proba(X_resampled)[:, 1]
xgb_val_pred_full = xgb_model_final.predict_proba(X_resampled)[:, 1]
lgb_val_pred_full = lgb_model_final.predict_proba(X_resampled)[:, 1]

# 将基础模型的预测结果组合为新的特征
stacked_val_pred_full = pd.DataFrame({
    'rf': rf_val_pred_full,
    'xgb': xgb_val_pred_full,
    'lgb': lgb_val_pred_full
})
# stacked_val_pred_full = cp.array(stacked_val_pred_full)
# 创建解释器
explainer_rf = shap.KernelExplainer(rf_model_final.predict_proba, shap.kmeans(X_resampled, 10))
explainer_xgb = shap.KernelExplainer(xgb_model_final.predict_proba, shap.kmeans(X_resampled, 10))
explainer_lgb = shap.KernelExplainer(lgb_model_final.predict_proba, shap.kmeans(X_resampled, 10))
explainer_stacking = shap.KernelExplainer(best_stack.predict_proba, shap.kmeans(stacked_val_pred_full, 10))

# 计算SHAP值
shap_values_rf = explainer_rf.shap_values(X_resampled)
shap_values_xgb = explainer_xgb.shap_values(X_resampled)
shap_values_lgb = explainer_lgb.shap_values(X_resampled)
shap_values_stacking = explainer_stacking.shap_values(stacked_val_pred_full)

# 综合特征重要性计算
# 计算每个特征的平均SHAP值，保持三维
mean_shap_values = np.mean([shap_values_rf, shap_values_xgb, shap_values_lgb], axis=0)

# 保存SHAP值图到PDF
shap_result_directory = "/home/xkj/project/msi/SHAP_result"
os.makedirs(shap_result_directory, exist_ok=True)
pdf_path = os.path.join(shap_result_directory, "shap_summary_plots.pdf")

with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(16, 18))
    shap.summary_plot(shap_values_rf, X_resampled, plot_type="bar", show=False)
    plt.title("Random Forest Feature Importance")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(16, 18))
    shap.summary_plot(shap_values_xgb, X_resampled, plot_type="bar", show=False)
    plt.title("XGBoost Feature Importance")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(16, 18))
    shap.summary_plot(shap_values_lgb, X_resampled, plot_type="bar", show=False)
    plt.title("LGBM Feature Importance")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 16))
    shap.summary_plot(shap_values_stacking, stacked_val_pred_full, plot_type="bar", show=False)
    plt.title("Stacking Model Feature Importance")
    pdf.savefig()
    plt.close()

    # Combined Feature Importance for each class
    for i in range(mean_shap_values.shape[0]):
        plt.figure(figsize=(16, 18))
        shap.summary_plot(mean_shap_values[i], X_resampled, plot_type="bar", show=False)
        plt.title(f"Combined Feature Importance - Class {i + 1}")
        pdf.savefig()
        plt.close()

print(f"SHAP summary plots saved to {pdf_path}")