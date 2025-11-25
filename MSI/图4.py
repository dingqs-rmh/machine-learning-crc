import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import os
from preprocess import preprocess_tpm_data
from preprocess import preprocess_tpm_data2
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

# 设置新罗马字体为默认字体
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

# =========================== MSI 二分类模型 高权重基因 (图A) ===========================

def plot_msi_high_weight_genes(ax, model_dir="/home/xkj/project/MSI/model/"):
    """
    展示MSI二分类模型的高权重基因，并绘制条形图。
    
    参数:
    ax (matplotlib.axes.Axes): 用于绘制高权重基因的条形图。
    model_dir (str): MSI模型保存的目录。
    """
    # 加载MSI模型（假设它是一个RF模型）
    rf_model = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
    lgb_model = joblib.load(os.path.join(model_dir, 'lgb_model.pkl'))

    # 提取特征重要性
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_
    lgb_importances = lgb_model.feature_importances_

    # 合并所有模型的特征重要性
    all_importances = (rf_importances + xgb_importances + lgb_importances) / 3

    # 获取选定特征
    with open(os.path.join(model_dir, 'selected_features.txt'), 'r') as f:
        selected_features = f.read().splitlines()

    # 创建一个DataFrame来存储特征和它们的平均重要性
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': all_importances
    })

    # 排序特征按重要性（从高到低）
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # 选择前20个重要特征
    top_genes = importance_df.head(20)
    top_genes = top_genes[::-1]

    # 绘制高权重基因的条形图
    ax.barh(top_genes['Feature'], top_genes['Importance'], color='skyblue')
    ax.set_xlabel('Feature Importance', fontsize=14)
    ax.set_title('Top 20 High Weight Genes for MSI Model', fontsize=16)

    # 美化图形
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('Genes', fontsize=14)

# =========================== TMB 二分类模型 高权重基因 (图B) ===========================

def plot_tmb_high_weight_genes(ax, data_path="/home/xkj/project/TMB/data/", smote=None):
    """
    展示TMB二分类模型的高权重基因，并绘制条形图。
    
    参数:
    ax (matplotlib.axes.Axes): 用于绘制高权重基因的条形图。
    data_path (str): TMB数据路径
    smote (object): SMOTE对象用于数据过采样
    """
    # 读取数据
    coad_data = pd.read_csv(os.path.join(data_path, 'COAD_TMB_MSI.csv'), index_col=0, low_memory=False)
    read_data = pd.read_csv(os.path.join(data_path, 'READ_TMB_MSI.csv'), index_col=0, low_memory=False)

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

    # 使用LASSO进行特征选择
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)

    # 获取Lasso选择的特征和系数
    selected_features = np.where(lasso.coef_ != 0)[0]
    selected_feature_names = feature_names[selected_features]

    # 选择前20个Lasso选择的特征
    top_lasso_genes = selected_feature_names[:20]

    # 绘制Lasso选择的前20个特征
    ax.barh(top_lasso_genes, lasso.coef_[selected_features][:20], color='lightcoral')
    ax.set_xlabel('Lasso Coefficients', fontsize=14)
    ax.set_title('Top 20 Features for TMB Model', fontsize=16)

    # 美化图形
    ax.tick_params(axis='both', labelsize=12)
    # ax.set_ylabel('Genes', fontsize=14)

# =========================== 合并图表到PDF ===========================

def plot_high_weight_genes_combined():
    """
    绘制MSI和TMB二分类模型的高权重基因条形图，并保存到同一PDF。
    """
    # 创建PDF文件保存图表
    output_pdf = '/home/xkj/project/MSI/图4.high_weight_genes_combined.pdf'
    
    with PdfPages(output_pdf) as pdf:
        # 创建并排显示的两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))  # 1行2列
        plot_msi_high_weight_genes(ax1, model_dir='/home/xkj/project/MSI/model/')
        plot_tmb_high_weight_genes(ax2, data_path='/home/xkj/project/TMB/data/')

        # 在整个图形的左上角添加字母 "A" 和 "B"
        fig.text(0.08, 0.95, 'A', fontsize=18, fontweight='bold', va='top', ha='left')
        fig.text(0.5, 0.95, 'B', fontsize=18, fontweight='bold', va='top', ha='left')
        
        # 保存到PDF
        pdf.savefig(fig)
        plt.close(fig)

    print(f"图表已保存为PDF文件：{output_pdf}")

# 调用函数生成并保存PDF文件
plot_high_weight_genes_combined()
