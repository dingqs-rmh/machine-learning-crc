import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
from preprocess import preprocess_tpm_data
from preprocess import preprocess_tpm_data2
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# 设置新罗马字体为默认字体
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

def preprocess_and_select_features(X, save_dir='model', preprocessing_function=1):
    """
    预处理并选择特征。
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

def calculate_metrics(y_true, y_pred):
    """
    计算并返回ACC, SPEC, SEN, PPV, NPV。
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 计算各个评估指标
    acc = accuracy_score(y_true, y_pred)
    spe = tn / (tn + fp) if tn + fp != 0 else 0
    sen = tp / (tp + fn) if tp + fn != 0 else 0
    ppv = tp / (tp + fp) if tp + fp != 0 else 0
    npv = tn / (tn + fn) if tn + fn != 0 else 0
    
    return acc, spe, sen, ppv, npv

def test_stacking_model(test_file, model_dir="/home/xkj/project/MSI/model/", test_preprocessing_function=1):
    """
    测试堆叠模型并计算AUC等评估指标。
    
    参数:
    test_file (str): 测试数据文件路径。
    model_dir (str): 模型保存的目录。
    """
    # 读取测试集数据
    test_data = pd.read_csv(test_file, index_col=0)

    # 分离标签和特征
    X_test = test_data.iloc[:, 1:]  # 特征
    y_test = test_data.iloc[:, 0]   # 标签

    # 预处理并选择特征
    selected_X_test = preprocess_and_select_features(X_test, save_dir=model_dir, preprocessing_function=test_preprocessing_function)

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

    # 计算其他指标
    acc, spe, sen, ppv, npv = calculate_metrics(y_test, final_test_pred)

    # 获取数据集名称
    dataset_name = os.path.splitext(os.path.basename(test_file))[0]

    # 计算AUC
    auc_value = roc_auc_score(y_test, final_test_pred_prob)

    return dataset_name, auc_value, acc, spe, sen, ppv, npv

def save_metrics_to_csv(test_files, model_dir="/home/xkj/project/MSI/model/", output_csv="table1.csv"):
    """
    计算所有验证集的评估指标，并将它们保存到CSV文件中。
    
    参数:
    test_files (list): 测试文件的路径列表。
    model_dir (str): 模型保存的目录。
    output_csv (str): 输出CSV的文件路径。
    """
    # 保存所有结果的列表
    results = []

    for i, test_file in enumerate(test_files):
        # 判断当前文件是最后两个文件
        if i >= len(test_files) - 2:
            plot_preprocessing_number = 1
        else:
            plot_preprocessing_number = 0
        
        # 获取并保存当前文件的评估指标
        dataset_name, auc_value, acc, spe, sen, ppv, npv = test_stacking_model(test_file, model_dir=model_dir, test_preprocessing_function=plot_preprocessing_number)
        
        # 将结果添加到列表
        results.append([dataset_name, auc_value, acc, spe, sen, ppv, npv])

    # 将结果保存到CSV文件
    results_df = pd.DataFrame(results, columns=["Dataset", "AUC", "ACC", "SPE", "SEN", "PPV", "NPV"])
    results_df.to_csv(output_csv, index=False)

# 示例用法：对指定目录下的所有GSE开头的CSV文件进行测试
data_directory = '/home/xkj/project/MSI/data/图2_data'
# 给定的文件顺序列表
file_order = [
    "GSE39084.csv", "GSE41258.csv", "GSE75316.csv", "GSE35896.csv",
    "GSE92921.csv", "GSE24550.csv", "GSE27544.csv", "GSE18088.csv",
    "GSE26682.csv", "GSE39582.csv", "GSE13294.csv", "GSE13067.csv"
]
test_files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.startswith('GSE') and file.endswith('.csv')]
test_files_sorted = sorted(test_files, key=lambda x: file_order.index(os.path.basename(x)))

# 输出CSV路径
output_csv = "/home/xkj/project/MSI/test_result/表1.csv"

# 计算并保存所有评估指标到CSV
save_metrics_to_csv(test_files_sorted, model_dir="/home/xkj/project/MSI/model/", output_csv=output_csv)
