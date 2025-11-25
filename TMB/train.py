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
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
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
from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import *
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, matthews_corrcoef, accuracy_score, recall_score, balanced_accuracy_score
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
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import brier_score_loss
import shap
import cupy as cp
import cudf
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR
from cuml.svm import SVC as cuSVC
import gc
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

def release_cuml_resources(obj):
    if hasattr(obj, '_reset_forest_data'):
        try:
            obj._reset_forest_data()
        except AttributeError:
            pass
    if hasattr(obj, '_dealloc'):
        try:
            obj._dealloc()
        except AttributeError:
            pass


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set working directory
os.chdir('/home/xkj/project/Ischemic_cardiomyopathy/')
output_directory = "/home/xkj/project/Ischemic_cardiomyopathy/train_output/"
os.makedirs(output_directory, exist_ok=True)

model_directory = "/home/xkj/project/Ischemic_cardiomyopathy/model/"
os.makedirs(model_directory, exist_ok=True)

# 定义缺失值的表示形式
na_values = ['', ' ', 'NA']

df = pd.read_csv('/home/xkj/project/Ischemic_cardiomyopathy/data/train_data/train.csv', na_values=na_values)
# 分离标签和特征
# 将Target列作为标签
y = df['Target']
# 除Target列，剩下的作为特征
X = df.drop(columns=['Target'])

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y) # type: ignore

from sklearn.model_selection import KFold, cross_val_predict
# 10折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 保存每一折的性能指标
metrics = []

# 保存每一折的预测结果和实际标签
val_preds = []
val_labels = []
val_probas = []

# 定义XGBoost的超参数网格
valid_param_grid_stacking = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'tree_method': ['hist'],
    'device': ['cuda']
}


# 记录每一折的最佳参数
best_params_stacking_list = []

for train_index, val_index in kf.split(X_resampled):
    X_train, X_val = X_resampled.iloc[train_index], X_resampled.iloc[val_index]
    y_train, y_val = y_resampled.iloc[train_index], y_resampled.iloc[val_index]

    X_train_cudf = cudf.DataFrame.from_pandas(X_train).astype('float32')
    X_val_cudf = cudf.DataFrame.from_pandas(X_val).astype('float32')
    y_train_cudf = cudf.Series(y_train).astype('float32')
    y_val_cudf = cudf.Series(y_val).astype('float32')
    
    # 对每个基础模型进行超参数调优
    best_rf = cuRF(n_estimators=100, random_state=42, n_streams=1)
    best_lgb = lgb.LGBMClassifier(n_estimators=100, random_state=42, device='gpu')
    # best_lr = cuLR()
    best_svc = cuSVC(C=1.0, kernel='rbf', probability=True, tol=1e-3, max_iter=5000)

    # 训练模型
    best_rf.fit(X_train_cudf, y_train_cudf)
    best_lgb.fit(X_train, y_train)
    # best_lr.fit(X_train_cudf, y_train_cudf)
    best_svc.fit(X_train_cudf, y_train_cudf)
    
    # 在验证集上进行预测
    rf_val_pred = best_rf.predict_proba(X_val_cudf).iloc[:, 1].to_array()
    lgb_val_pred = best_lgb.predict_proba(X_val)[:, 1] # type: ignore
    # lr_val_pred = best_lr.predict_proba(X_val_cudf).iloc[:, 1].to_array()
    svc_val_pred = best_svc.predict_proba(X_val_cudf).iloc[:, 1].to_array()

    release_cuml_resources(best_rf)
    # release_cuml_resources(best_lr)
    release_cuml_resources(best_svc)
    gc.collect()

    # 将基础模型的预测结果组合为新的特征
    stacked_val_pred = pd.DataFrame({
        'rf': rf_val_pred,
        'lgb': lgb_val_pred,
        # 'lr': lr_val_pred,
        'svc': svc_val_pred
    })

    stacked_val_pred = cp.array(stacked_val_pred)

    # 对元学习器进行超参数调优
    grid_search_stacking = GridSearchCV(XGBClassifier(), valid_param_grid_stacking, cv=5, scoring='roc_auc', n_jobs=8)
    grid_search_stacking.fit(stacked_val_pred, y_val)
    best_stacking_model = grid_search_stacking.best_estimator_

    best_params_stacking_list.append(best_stacking_model.get_params())

    # 元学习器在验证集上的预测
    final_val_pred = best_stacking_model.predict(stacked_val_pred)
    final_val_proba = best_stacking_model.predict_proba(stacked_val_pred)[:, 1]

    val_probas.extend(final_val_proba)
    val_labels.extend(y_val)

    # 计算性能指标
    auc_score = roc_auc_score(y_val, final_val_proba)
    acc_score = accuracy_score(y_val, final_val_pred)
    recall = recall_score(y_val, final_val_pred)
    f1 = f1_score(y_val, final_val_pred)
    precision = precision_score(y_val, final_val_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, final_val_pred).ravel()
    specificity = tn / (tn + fp)
    brier_score = brier_score_loss(y_val, final_val_proba)
    
    metrics.append((auc_score, acc_score, recall, f1, precision, specificity, brier_score))

def bootstrap_ci(data, n_bootstrap=1000, alpha=0.05):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        samples = resample(data, replace=True, n_samples=len(data))
        bootstrapped_means.append(np.mean(samples))
    lower = np.percentile(bootstrapped_means, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
    return np.mean(data), (lower, upper)

# 计算各个性能指标的均值和置信区间
auc_scores, acc_scores, recall_scores, f1_scores, precision_scores, specificity_scores, brier_scores = zip(*metrics)

auc_mean, auc_ci = bootstrap_ci(auc_scores)
acc_mean, acc_ci = bootstrap_ci(acc_scores)
recall_mean, recall_ci = bootstrap_ci(recall_scores)
f1_mean, f1_ci = bootstrap_ci(f1_scores)
precision_mean, precision_ci = bootstrap_ci(precision_scores)
specificity_mean, specificity_ci = bootstrap_ci(specificity_scores)
brier_mean, brier_ci = bootstrap_ci(brier_scores)


output_file_path = os.path.join(output_directory, 'performance_metrics.txt')

# 将性能指标写入txt文件
with open(output_file_path, 'w') as f:
    f.write('Metric\tMean\t95% CI\n')
    f.write(f"AUC\t{auc_mean:.4f}\t{auc_ci[0]:.4f}-{auc_ci[1]:.4f}\n")
    f.write(f"ACC\t{acc_mean:.4f}\t{acc_ci[0]:.4f}-{acc_ci[1]:.4f}\n")
    f.write(f"Recall\t{recall_mean:.4f}\t{recall_ci[0]:.4f}-{recall_ci[1]:.4f}\n")
    f.write(f"F1\t{f1_mean:.4f}\t{f1_ci[0]:.4f}-{f1_ci[1]:.4f}\n")
    f.write(f"Precision\t{precision_mean:.4f}\t{precision_ci[0]:.4f}-{precision_ci[1]:.4f}\n")
    f.write(f"Specificity\t{specificity_mean:.4f}\t{specificity_ci[0]:.4f}-{specificity_ci[1]:.4f}\n")
    f.write(f"Brier Score\t{brier_mean:.4f}\t{brier_ci[0]:.4f}-{brier_ci[1]:.4f}\n")

print(f"Performance metrics saved to {output_file_path}")

# 计算并绘制ROC曲线
fpr, tpr, _ = roc_curve(val_labels, val_probas)
roc_auc = auc(fpr, tpr)

# # 计算AUC的95%置信区间
# def auc_ci_cal(val_labels, val_probas, alpha=0.95, n_bootstraps=1000):
#     rng = np.random.RandomState(42)
#     bootstrapped_aucs = []
#     for _ in range(n_bootstraps):
#         indices = rng.randint(0, len(val_probas), len(val_probas))
#         if len(np.unique(val_labels[indices])) < 2:
#             continue
#         score = roc_auc_score(val_labels[indices], val_probas[indices])
#         bootstrapped_aucs.append(score)
#     sorted_scores = np.array(bootstrapped_aucs)
#     sorted_scores.sort()
#     lower = sorted_scores[int((1.0 - alpha) / 2.0 * len(sorted_scores))]
#     upper = sorted_scores[int((1.0 + alpha) / 2.0 * len(sorted_scores))]
#     return lower, upper

# ci_lower, ci_upper = auc_ci_cal(np.array(val_labels), np.array(val_probas))

pdf_path = os.path.join(output_directory, "roc_curve.pdf")

with PdfPages(pdf_path) as pdf:
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_mean:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    pdf.savefig()
    plt.close()

print(f"ROC curve saved as {pdf_path}")

# 计算并绘制校准曲线
prob_true, prob_pred = calibration_curve(val_labels, val_probas, n_bins=10)

calibration_curve_path = os.path.join(output_directory, "calibration_curve.pdf")

with PdfPages(calibration_curve_path) as pdf:
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Stacked Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    pdf.savefig()
    plt.close()

print(f"Calibration curve saved as {calibration_curve_path}")

# 选择最优参数
def get_most_common_params(param_list):
    counter = Counter(tuple(sorted(params.items())) for params in param_list)
    most_common_params = dict(counter.most_common(1)[0][0])
    return most_common_params

best_params_stacking = get_most_common_params(best_params_stacking_list)

X_resampled_cudf = cudf.DataFrame.from_pandas(X_resampled).astype('float32')
y_resampled_cudf = cudf.Series(y_resampled).astype('float32')

# 使用最优参数对整个数据集进行训练
rf_model_final = cuRF(n_estimators=100, random_state=42, n_streams=1)
lgb_model_final = lgb.LGBMClassifier(n_estimators=100, random_state=42, device='gpu')
# lr_model_final = cuLR()
svc_model_final = cuSVC(C=1.0, kernel='rbf', probability=True, tol=1e-3, max_iter=5000)

rf_model_final.fit(X_resampled_cudf, y_resampled_cudf)
lgb_model_final.fit(X_resampled, y_resampled)
# lr_model_final.fit(X_resampled_cudf, y_resampled_cudf)
svc_model_final.fit(X_resampled_cudf, y_resampled_cudf)

joblib.dump(rf_model_final, os.path.join(model_directory, 'rf_model.pkl'))
joblib.dump(lgb_model_final, os.path.join(model_directory, 'lgb_model.pkl'))
# joblib.dump(lr_model_final, os.path.join(model_directory, 'lr_model.pkl'))
joblib.dump(svc_model_final, os.path.join(model_directory, 'svc_model.pkl'))

rf_val_pred_full = rf_model_final.predict_proba(X_resampled_cudf).iloc[:, 1].to_array()
lgb_val_pred_full = lgb_model_final.predict_proba(X_resampled)[:, 1]
# lr_val_pred_full = lr_model_final.predict_proba(X_resampled_cudf).iloc[:, 1].to_array()
svc_val_pred_full = svc_model_final.predict_proba(X_resampled_cudf).iloc[:, 1].to_array()

release_cuml_resources(rf_model_final)
# release_cuml_resources(lr_model_final)
release_cuml_resources(svc_model_final)
gc.collect()


stacked_val_pred_full = pd.DataFrame({
    'rf': rf_val_pred_full,
    'lgb': lgb_val_pred_full,
    # 'lr': lr_val_pred_full,
    'svc': svc_val_pred_full
})
stacked_val_pred_full = cp.array(stacked_val_pred_full)

best_xgb = XGBClassifier(**best_params_stacking)
best_xgb.fit(stacked_val_pred_full, y_resampled)

joblib.dump(best_xgb, os.path.join(model_directory, 'stacking_model.pkl'))

print("Models have been saved successfully.")



# 加载训练好的基础模型和元学习器
rf_model_final = joblib.load(os.path.join(model_directory, 'rf_model.pkl'))
lgb_model_final = joblib.load(os.path.join(model_directory, 'lgb_model.pkl'))
# lr_model_final = joblib.load(os.path.join(model_directory, 'lr_model.pkl'))
svc_model_final = joblib.load(os.path.join(model_directory, 'svc_model.pkl'))
best_xgb = joblib.load(os.path.join(model_directory, 'stacking_model.pkl'))

# 基础模型在全数据集上的预测
rf_val_pred_full = rf_model_final.predict_proba(X_resampled_cudf).iloc[:, 1].to_array()
lgb_val_pred_full = lgb_model_final.predict_proba(X_resampled)[:, 1]
# lr_val_pred_full = lr_model_final.predict_proba(X_resampled_cudf).iloc[:, 1].to_array()
svc_val_pred_full = svc_model_final.predict_proba(X_resampled_cudf).iloc[:, 1].to_array()

# 将基础模型的预测结果组合为新的特征
stacked_val_pred_full = pd.DataFrame({
    'rf': rf_val_pred_full,
    'lgb': lgb_val_pred_full,
    # 'lr': lr_val_pred_full,
    'svc': svc_val_pred_full
})
# stacked_val_pred_full = cp.array(stacked_val_pred_full)
# 创建解释器
explainer_rf = shap.KernelExplainer(rf_model_final.predict_proba, shap.kmeans(X_resampled_cudf.to_pandas(), 10))
explainer_lgb = shap.KernelExplainer(lgb_model_final.predict_proba, shap.kmeans(X_resampled, 10))
# explainer_lr = shap.KernelExplainer(lr_model_final.predict_proba, shap.kmeans(X_resampled_cudf.to_pandas(), 10))
explainer_svc = shap.KernelExplainer(svc_model_final.predict_proba, shap.kmeans(X_resampled_cudf.to_pandas(), 10))
explainer_stacking = shap.KernelExplainer(best_xgb.predict_proba, shap.kmeans(stacked_val_pred_full, 10))

# 计算SHAP值
shap_values_rf = explainer_rf.shap_values(X_resampled_cudf.to_pandas())
shap_values_lgb = explainer_lgb.shap_values(X_resampled)
shap_values_svc = explainer_svc.shap_values(X_resampled_cudf.to_pandas())
shap_values_stacking = explainer_stacking.shap_values(stacked_val_pred_full)

# 综合特征重要性计算
# 计算每个特征的平均SHAP值，保持三维
mean_shap_values = np.mean([shap_values_rf, shap_values_lgb, shap_values_svc], axis=0)

# 保存SHAP值图到PDF
shap_result_directory = "/home/xkj/project/Ischemic_cardiomyopathy/SHAP_result"
os.makedirs(shap_result_directory, exist_ok=True)
pdf_path = os.path.join(shap_result_directory, "shap_summary_plots.pdf")

with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(16, 18))
    shap.summary_plot(shap_values_rf, X_resampled, plot_type="bar", show=False)
    plt.title("Random Forest Feature Importance")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(16, 18))
    shap.summary_plot(shap_values_lgb, X_resampled, plot_type="bar", show=False)
    plt.title("Gradient Boosting Feature Importance")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(16, 18))
    shap.summary_plot(shap_values_svc, X_resampled, plot_type="bar", show=False)
    plt.title("SVC Feature Importance")
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