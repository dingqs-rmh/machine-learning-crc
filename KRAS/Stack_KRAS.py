import pandas as pd 
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from scipy.stats import ttest_ind
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.colors as mcolors

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 读取数据
coad_data = pd.read_csv('/home/xkj/project/KRAS/data/COAD_KRAS.csv')
read_data = pd.read_csv('/home/xkj/project/KRAS/data/READ_KRAS.csv')

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

# 创建SMOTE实例
smote = SMOTE(random_state=42)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4048)

# 自定义决策阈值
custom_threshold = 0.5

# 初始化一个列表来存储每折的性能指标
performance_records = []

# 创建PDF文件保存ROC/混淆矩阵和SHAP图
with PdfPages('/home/xkj/project/KRAS/plots/combined_roc_cm.pdf') as pdf_roc_cm, \
     PdfPages('/home/xkj/project/KRAS/plots/combined_feature_importance.pdf') as pdf_shap:

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 使用LASSO进行特征选择
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train_scaled, y_train)

        # 选择非零系数的特征
        selected_features = np.where(lasso.coef_ != 0)[0]
        X_train_selected = X_train_scaled[:, selected_features]
        X_test_selected = X_test_scaled[:, selected_features]

        # 获取选择后的特征名称
        selected_feature_names = feature_names[selected_features]

        # SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

        # 定义基本分类器和元分类器
        base_estimators = [
            ('rf', RandomForestClassifier(random_state=42)),
            ('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),
            ('lgb', LGBMClassifier(random_state=42))
        ]

        # 使用堆叠法创建分类器
        stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())

        # 训练堆叠模型
        stacking_model.fit(X_train_resampled, y_train_resampled)

        # 预测
        y_pred_prob = stacking_model.predict_proba(X_test_selected)[:, 1]
        y_pred = (y_pred_prob >= custom_threshold).astype(int)

        # 创建一个新页面保存ROC曲线和混淆矩阵
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        axes[0].plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey', lw=2)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('1 - Specificity', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Sensitivity', fontsize=12, fontweight='bold')
        axes[0].set_title(f'ROC Curve Fold {i+1}', fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[1], values_format='d')
        axes[1].set_title(f'Confusion Matrix Fold {i+1}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted label', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True label', fontsize=12, fontweight='bold')

        plt.tight_layout()
        pdf_roc_cm.savefig(fig)
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

        # 计算并可视化基于模型的特征重要性
        importances = np.zeros(X_train_selected.shape[1])
        
        # 对每个基分类器计算特征重要性
        for name, model in stacking_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_

        # 取平均值
        importances /= len(stacking_model.named_estimators_)

        # 将重要性转换为DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': selected_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 仅选择前20个重要性最高的特征
        top_n = 20
        top_features = feature_importance_df.head(top_n)

        # 自定义颜色，您可以根据需要调整颜色
        color = '#78c2ad'

        # 绘制水平条形图
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color=color)
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Genes', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Gene Importance Fold {i+1}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # 反转y轴，使得重要性最高的特征在最上面

        # 保存到PDF
        pdf_shap.savefig()
        plt.close()

# 检查并创建目录
output_dir = '/home/xkj/project/KRAS/metrics/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存性能记录为表格文件
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv('/home/xkj/project/KRAS/metrics/performance_metrics.csv', index=False)

print("交叉验证的性能记录已保存为CSV文件，SHAP图已保存到PDF文件。")
