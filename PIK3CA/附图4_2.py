import pandas as pd 
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.colors as mcolors

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'

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

# 创建SMOTE实例
smote = SMOTE(random_state=42)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4048)

# 自定义决策阈值
custom_threshold = 0.35

# 初始化一个列表来存储每折的性能指标
performance_records = []

# 创建PDF文件保存混淆矩阵
with PdfPages('/home/xkj/project/PIK3CA/plots/附图4_2.pdf') as pdf_cm:

    # 创建一行5列的布局来绘制混淆矩阵
    fig, axes = plt.subplots(1, 5, figsize=(16.67, 3))

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 使用LASSO进行特征选择
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)

        # 选择非零系数的特征
        selected_features = np.where(lasso.coef_ != 0)[0]
        X_train_selected = X_train_scaled[:, selected_features]
        X_test_selected = X_test_scaled[:, selected_features]

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

        # 获取预测的概率
        y_pred_proba = stacking_model.predict_proba(X_test_selected)[:, 1]  # 获取类别1的概率

        # 使用自定义阈值进行分类
        y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred_custom)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # 绘制混淆矩阵
        disp.plot(cmap='Blues', ax=axes[i], values_format='d')
        axes[i].set_title(f'PIK3CA Fold {i+1}')
        axes[i].set_xlabel('Predicted label')
        axes[i].set_ylabel('True label')

       
    # 调整布局
    plt.tight_layout()

    # 保存混淆矩阵图
    pdf_cm.savefig(fig)
    plt.close()

# 检查并创建目录
output_dir = '/home/xkj/project/PIK3CA/metrics/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存性能记录为表格文件
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv('/home/xkj/project/PIK3CA/metrics/performance_metrics.csv', index=False)

print("交叉验证的性能记录已保存为CSV文件，混淆矩阵已保存到PDF文件。")
