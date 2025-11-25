import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
from fancyimpute import IterativeImputer
from preprocess import preprocess_tpm_data
from preprocess import preprocess_tpm_data2
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from matplotlib import rcParams

# 设置新罗马字体为默认字体
rcParams['font.family'] = 'Times New Roman'

def preprocess_and_select_features(X, save_dir='model', preprocessing_function=1):
    """
    预处理并选择特征。 (此部分不做修改)
    """
    # 预处理测试集特征数据
    missing_value_strategy='mean'
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

def test_stacking_model(test_file, ax, model_dir="/home/xkj/project/msi/model/", test_preprocessing_function=1):
    """
    测试堆叠模型并绘制混淆矩阵。
    
    参数:
    test_file (str): 测试数据文件路径。
    ax (matplotlib.axes.Axes): 子图的Axes对象，用于绘制混淆矩阵。
    model_dir (str): 模型保存的目录。
    """
    # 读取测试集数据
    test_data = pd.read_csv(test_file, index_col=0)

    # 分离标签和特征
    X_test = test_data.iloc[:, 1:]  # 特征
    y_test = test_data.iloc[:, 0]   # 标签

    # 预处理并选择特征
    if test_preprocessing_function == 1:
        test_preprocessing_number = 1
    else:
        test_preprocessing_number = 0
    selected_X_test = preprocess_and_select_features(X_test, save_dir=model_dir, preprocessing_function=test_preprocessing_number)

    # 加载模型
    rf_model = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
    lgb_model = joblib.load(os.path.join(model_dir, 'lgb_model.pkl'))
    stacking_model = joblib.load(os.path.join(model_dir, 'stacking_model.pkl'))

    # 在测试集上预测
    rf_test_pred = rf_model.predict_proba(selected_X_test)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(selected_X_test)[:, 1]
    lgb_test_pred = lgb_model.predict_proba(selected_X_test)[:, 1]

    # 将基础模型的预测结果组合为新的特征
    stacked_test_pred = pd.DataFrame({
        'rf': rf_test_pred,
        'xgb': xgb_test_pred,
        'lgb': lgb_test_pred
    })

    # 使用第二层模型进行最终预测
    final_test_pred_prob = stacking_model.predict_proba(stacked_test_pred)[:, 1]
    final_test_pred = (final_test_pred_prob >= 0.5).astype(int)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, final_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # 绘制混淆矩阵到指定的子图
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    # 提取文件名（去掉扩展名）
    file_name = os.path.splitext(os.path.basename(test_file))[0]
    
    # 设置混淆矩阵的标题为文件名
    ax.set_title(f'{file_name}')

def plot_all_confusion_matrices(test_files, model_dir="/home/xkj/project/msi/model/", output_path="confusion_matrices.pdf"):
    """
    绘制所有验证队列的混淆矩阵，并将它们保存为PDF。
    
    参数:
    test_files (list): 测试文件的路径列表。
    model_dir (str): 模型保存的目录。
    output_path (str): 输出PDF的文件路径。
    """
    # 创建一个4x3的子图布局
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    axes = axes.flatten()  # 将二维数组展平成一维

    with PdfPages(output_path) as pdf:
        for i, test_file in enumerate(test_files):
            # 判断当前文件是最后两个文件
            if i >= len(test_files) - 2:
                plot_preprocessing_number = 1
            else:
                plot_preprocessing_number = 0
            test_stacking_model(test_file, axes[i], model_dir=model_dir, test_preprocessing_function=plot_preprocessing_number)
        
        # 调整布局，使子图不重叠
        plt.tight_layout()
        pdf.savefig()  # 将所有子图保存到一个PDF文件中
        plt.close()

# 示例用法：对指定目录下的所有GSE开头的CSV文件进行测试
data_directory = '/home/xkj/project/msi/data/图2_data'
# 给定的文件顺序列表
file_order = [
    "GSE39084.csv", "GSE41258.csv", "GSE75316.csv", "GSE35896.csv",
    "GSE92921.csv", "GSE24550.csv", "GSE27544.csv", "GSE18088.csv",
    "GSE26682.csv", "GSE39582.csv", "GSE13294.csv", "GSE13067.csv"
]
test_files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.startswith('GSE') and file.endswith('.csv')]
test_files_sorted = sorted(test_files, key=lambda x: file_order.index(os.path.basename(x)))

# 输出PDF路径
output_pdf = "/home/xkj/project/msi/test_result/图2.混淆矩阵.pdf"

# 绘制并保存所有混淆矩阵
plot_all_confusion_matrices(test_files_sorted, model_dir="/home/xkj/project/msi/model/", output_path=output_pdf)