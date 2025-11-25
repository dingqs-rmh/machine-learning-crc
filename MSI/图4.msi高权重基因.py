import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import os
from preprocess import preprocess_tpm_data
from preprocess import preprocess_tpm_data2
from sklearn.impute import SimpleImputer

# 设置新罗马字体为默认字体
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

def preprocess_and_select_features(X, save_dir='model', preprocessing_function=1):
    """
    预处理并选择特征。 (此部分不做修改)
    """
    # 预处理特征数据
    missing_value_strategy = 'mean'
    imputer = SimpleImputer(strategy=missing_value_strategy)
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X = X_filled

    if preprocessing_function == 1:
        preprocessed_X_test = preprocess_tpm_data2(X)
    else:
        preprocessed_X_test = preprocess_tpm_data(X)

    # 读取选择的特征
    with open(os.path.join(save_dir, 'selected_features.txt'), 'r') as f:
        selected_features = f.read().splitlines()
        
    # 应用特征选择
    selected_X_test = preprocessed_X_test[selected_features]
    
    return selected_X_test

def plot_model_high_weight_genes(ax, model_dir="/home/xkj/project/msi/model/"):
    """
    展示模型的高权重基因，并绘制条形图。
    
    参数:
    ax (matplotlib.axes.Axes): 用于绘制高权重基因的条形图。
    model_dir (str): 模型保存的目录。
    """
    # 加载模型
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
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 20 High Weight Genes Across Models', fontsize=14)
    
    # 美化图形
    ax.tick_params(axis='both', labelsize=10)
    ax.set_ylabel('Genes', fontsize=12)

def plot_model_high_weight_genes_all(model_dir="/home/xkj/project/msi/model/", output_path="high_weight_genes.pdf"):
    """
    绘制并保存模型的高权重基因条形图。
    
    参数:
    model_dir (str): 模型保存的目录。
    output_path (str): 输出PDF的文件路径。
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制高权重基因
    plot_model_high_weight_genes(ax, model_dir=model_dir)

    # 保存到PDF文件
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)  # 将图形保存到PDF文件中
        plt.close(fig)

# 示例用法：绘制并保存模型的高权重基因图表
output_pdf = "/home/xkj/project/msi/test_result/图4.高权重基因图表.pdf"
plot_model_high_weight_genes_all(model_dir="/home/xkj/project/msi/model/", output_path=output_pdf)
