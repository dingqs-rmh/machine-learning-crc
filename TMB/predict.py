import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib
from matplotlib.backends.backend_pdf import PdfPages

# 设置全局字体为 Times New Roman，并加粗轴标签和标题
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 文件路径
test_data_paths = {
    'LUAD': '/home/xkj/project/TMB/data/LUAD_TMB_MSI.csv',
    'LUSC': '/home/xkj/project/TMB/data/LUSC_TMB_MSI.csv',
    'BRCA': '/home/xkj/project/TMB/data/BRCA_TMB_MSI.csv'
}
model_path = '/home/xkj/project/TMB/models/best_random_forest_model.pkl'
scaler_path = '/home/xkj/project/TMB/models/scaler_selected.pkl'
selected_features_path = '/home/xkj/project/TMB/models/selected_features.csv'
performance_output_path = '/home/xkj/project/TMB/result/performance_metrics.csv'

# 加载模型和标准化器
best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 读取保存的特征名
selected_features_df = pd.read_csv(selected_features_path)
selected_feature_names = selected_features_df['Selected Features'].values

# 准备存储性能指标的列表
performance_metrics = []

# 打开PDF以保存ROC曲线和混淆矩阵
with PdfPages('/home/xkj/project/TMB/result/roc_curves.pdf') as roc_pdf, \
     PdfPages('/home/xkj/project/TMB/result/confusion_matrices.pdf') as cm_pdf:
    
    # 创建ROC曲线的图形
    plt.figure(figsize=(10, 8))

    # 遍历测试集
    for dataset_name, test_data_path in test_data_paths.items():
        # 读取测试数据并忽略第一列作为索引列
        test_data = pd.read_csv(test_data_path, index_col=0, low_memory=False)

        # 跳过第三、第四列（即列索引为 1 和 2 的列）
        test_data = test_data.drop(test_data.columns[[1, 2]], axis=1)


        # 生成标签：TMB > 10 属于 1，否则为 0
        y_test = (test_data['TMB'] > 30).astype(int)

        # 提取特征：第五列及之后的所有列作为特征
        X_test = test_data.iloc[:, 1:]

        # 检查和填充缺失特征
        missing_features = [f for f in selected_feature_names if f not in test_data.columns]
        for feature in missing_features:
            test_data[feature] = np.nan

        # 使用IterativeImputer插值部分缺失值
        imputer = IterativeImputer(max_iter=10, random_state=0)
        test_data_imputed = imputer.fit_transform(test_data[selected_feature_names])

        # 对数变换（log2(TPM + 1)）
        X_test = pd.DataFrame(test_data_imputed, columns=selected_feature_names)
        X_test = np.log2(X_test + 1)

        # 标准化
        X_test_scaled = scaler.transform(X_test)

        # 预测结果和概率
        y_test_pred = best_model.predict(X_test_scaled)
        y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        # 计算性能指标
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        # 存储性能指标
        performance_metrics.append({
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {roc_auc:.2f})')

        # 绘制混淆矩阵
        fig, ax = plt.subplots()
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
        for j in range(conf_matrix.shape[0]):
            for k in range(conf_matrix.shape[1]):
                ax.text(x=k, y=j, s=conf_matrix[j, k], va='center', ha='center')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{dataset_name} Confusion Matrix', fontsize=14)
        cm_pdf.savefig(fig)  # 保存当前混淆矩阵到PDF
        plt.close(fig)

    # 完成ROC曲线的绘制
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlabel('1 - Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.title('ROC Curves for LUAD, LUSC, and BRCA', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    roc_pdf.savefig()  # 保存ROC曲线到PDF
    plt.close()

# 保存性能指标为CSV文件
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(performance_output_path, index=False)

print(f"性能指标已保存到：{performance_output_path}")
print("ROC曲线和混淆矩阵已分别保存为PDF。")
