import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# 设置全局字体为“Times New Roman”，并调整默认字体大小和加粗选项
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 全局字体大小
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗

# 读取数据
coad_data = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB.csv')
read_data = pd.read_csv('/home/xkj/project/TMB/data/READ_TMB.csv')

# 合并数据集
data = pd.concat([coad_data, read_data], axis=0)

# 提取特征名
feature_names = data.columns[1:]

# 对数变换（log2(TPM + 1)）
data.iloc[:, 1:] = np.log2(data.iloc[:, 1:] + 1)

# 设置阈值，将TMB分类为高（1）或低（0）
threshold = 10
y = (data.iloc[:, 0] > threshold).astype(int)  # 高TMB为1，低TMB为0

# 分离特征
X = data.iloc[:, 1:]  # 所有基因表达数据

# 创建SMOTE实例
smote = SMOTE(random_state=42)

# 使用5折交叉验证的结果来绘制ROC曲线和生成混淆矩阵
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4048)

# 自定义决策阈值
custom_threshold = 0.4

# 初始化一个列表来存储每折的性能指标
performance_records = []

# 创建两个PDF文件，分别保存ROC/混淆矩阵和SHAP图
with PdfPages('/home/xkj/project/TMB/plots/combined_roc_cm.pdf') as pdf_roc_cm, \
     PdfPages('/home/xkj/project/TMB/plots/combined_shap.pdf') as pdf_shap:

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 数据标准化：仅在训练集上拟合，然后在测试集上应用
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 使用LASSO进行特征选择：仅在训练集上进行拟合
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train_scaled, y_train)

        # 选择非零系数的特征
        selected_features = np.where(lasso.coef_ != 0)[0]
        X_train_selected = X_train_scaled[:, selected_features]
        X_test_selected = X_test_scaled[:, selected_features]

        # 获取选择后的特征名称
        selected_feature_names = feature_names[selected_features]

        # SMOTE：在特征选择之后的训练集上进行过采样
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train) # type: ignore

        # 使用随机森林进行训练
        best_model = RandomForestClassifier(random_state=42)
        best_model.fit(X_train_resampled, y_train_resampled)

        # 预测
        y_pred_prob = best_model.predict_proba(X_test_selected)[:, 1]
        y_pred = (y_pred_prob >= custom_threshold).astype(int)

        # 创建一个新页面保存ROC曲线和混淆矩阵
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 每一折两列，分别放ROC和混淆矩阵

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线到第一列
        axes[0].plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey', lw=2)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('1 - Specificity', fontsize=12, fontweight='bold')  # 加粗字体
        axes[0].set_ylabel('Sensitivity', fontsize=12, fontweight='bold')  # 加粗字体
        axes[0].set_title(f'ROC Curve Fold {i+1}', fontsize=14, fontweight='bold')  # 加粗字体
        axes[0].legend(loc="lower right")

        # 混淆矩阵到第二列
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[1], values_format='d')
        axes[1].set_title(f'Confusion Matrix Fold {i+1}', fontsize=14, fontweight='bold')  # 加粗字体
        axes[1].set_xlabel('Predicted label', fontsize=12, fontweight='bold')  # 加粗字体
        axes[1].set_ylabel('True label', fontsize=12, fontweight='bold')  # 加粗字体

        plt.tight_layout()
        pdf_roc_cm.savefig(fig)  # 保存当前页面的ROC和混淆矩阵
        plt.close()

        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 保存每折的性能记录
        performance_records.append({
            'Fold': i+1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': roc_auc
        })

        # 使用SHAP解释模型并生成SHAP summary plot
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_selected)

        # 创建一个新图来生成SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values[1], X_test_selected, feature_names=selected_feature_names, show=False)
        plt.title(f'SHAP Summary Plot Fold {i+1}', fontsize=14, fontweight='bold')  # 加粗字体
        pdf_shap.savefig()  # 保存SHAP summary plot到PDF
        plt.close()

# 检查并创建目录
output_dir = '/home/xkj/project/TMB/metrics/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将性能记录保存为表格文件
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv('/home/xkj/project/TMB/metrics/performance_metrics.csv', index=False)

print("交叉验证的性能记录已保存为CSV文件，SHAP图已保存到PDF文件。")
