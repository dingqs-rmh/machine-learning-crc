import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
# 读取数据
coad_data = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB_MSI.csv', index_col=0, low_memory=False)
read_data = pd.read_csv('/home/xkj/project/TMB/data/READ_TMB_MSI.csv', index_col=0, low_memory=False)

# 合并数据集
data = pd.concat([coad_data, read_data], axis=0)

# 提取特征名
feature_names = data.columns[1:]

# 对数变换（log2(TPM + 1)）
data.iloc[:, 3:] = np.log2(data.iloc[:, 3:] + 1)

# 设置阈值，将TMB分类为高（1）或低（0）
threshold = 10
y = (data.iloc[:, 0] > threshold).astype(int)  # 高TMB为1，低TMB为0

# 分离特征
X = data.iloc[:, 3:]  # 所有基因表达数据

# 创建SMOTE实例
smote = SMOTE(random_state=42)

# 使用5折交叉验证的结果来生成混淆矩阵
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4048)

# 自定义决策阈值
custom_threshold = 0.4

# 初始化一个列表来存储每折的性能指标
performance_records = []

# 创建一个PDF文件保存混淆矩阵
with PdfPages('/home/xkj/project/TMB/plots/图3.pdf') as pdf_cm:

    # 创建一个新的图形来容纳所有折叠的混淆矩阵
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 创建2行3列的子图

    # 遍历每个折叠
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

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 在对应位置绘制混淆矩阵
        if i < 5:  # 只绘制前5个折叠的混淆矩阵
            ax = axes[i // 3, i % 3]  # 根据折叠索引决定位置
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues')  # 不显示颜色条

            ax.set_title(f'Fold {i+1}')

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
        })

    # 删除最后一个子图
    fig.delaxes(axes[1, 2])  # 删除第6个空白子图

    # 调整布局，使其不留下空白
    
    plt.tight_layout()
    pdf_cm.savefig()  # 保存混淆矩阵到PDF文件
    plt.close()

# 检查并创建目录
output_dir = '/home/xkj/project/TMB/metrics/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将性能记录保存为表格文件
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv('/home/xkj/project/TMB/metrics/performance_metrics.csv', index=False)

print("交叉验证的性能记录已保存为CSV文件，混淆矩阵已保存到PDF文件。")
