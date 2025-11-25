import pandas as pd
import json

# 读取第一个临床数据文件
clinical_data_1 = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB.csv')

# 读取第二个临床数据文件
clinical_data_2 = pd.read_csv('/home/xkj/project/TMB/data/READ_TMB.csv')

# 合并两个数据集
clinical_data = pd.concat([clinical_data_1, clinical_data_2], ignore_index=True)

# 提取MSI列
msi_column = clinical_data['TMB']

# 读取基因名称文件
with open('/home/xkj/project/CRC-APP/TMB/model/selected_features.txt', 'r') as file:
    selected_genes = [line.strip() for line in file]

# 提取所需的基因列
selected_gene_columns = clinical_data[selected_genes]

# 将MSI列和基因列合并为一个新的DataFrame
merged_data = pd.concat([msi_column, selected_gene_columns], axis=1)

# 计算每个基因的平均值
gene_means = selected_gene_columns.mean()

# 将平均值结果转为字典
gene_means_dict = gene_means.to_dict()

# 将结果保存为JSON文件
with open('/home/xkj/project/CRC-APP/TMB/model/gene_means.json', 'w') as json_file:
    json.dump(gene_means_dict, json_file, indent=4)



