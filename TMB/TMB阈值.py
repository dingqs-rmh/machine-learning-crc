import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 读取数据
data = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB_clinical.csv')

# 绘制TMB值的分布
plt.figure(figsize=(8, 6))
plt.hist(data.iloc[:, 0], bins=30, color='skyblue', edgecolor='black')
plt.title('TMB Distribution')
plt.xlabel('TMB')
plt.ylabel('Frequency')

# 保存TMB分布图
plt.savefig('/home/xkj/project/TMB/plots/TMB_Distribution.png')  # 保存为指定目录和文件名
plt.show()

# 基于ROC曲线分析选择阈值
y_true = data['clinical_outcome']  # 假设你有一个临床结局列，如是否对免疫治疗有反应
tmb_values = data.iloc[:, 0]  # TMB值

fpr, tpr, thresholds = roc_curve(y_true, tmb_values)
roc_auc = roc_auc_score(y_true, tmb_values)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for TMB Threshold Selection')
plt.legend(loc='best')

# 保存ROC曲线图
plt.savefig('/home/xkj/project/TMB/plots/ROC_Curve.png')  # 保存为指定目录和文件名
plt.show()

# 找到最大化Youden's Index的阈值
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f'最佳TMB阈值: {optimal_threshold:.2f}')

