import pandas as pd
import json

# 读取临床数据文件
clinical_data = pd.read_csv('/home/xkj/project/msi/data/GSE39084_clinical_msi.csv')

# 提取MSI列
msi_column = clinical_data['MSI']

# 读取基因名称文件
with open('/home/xkj/project/msi/model/selected_features.txt', 'r') as file:
    selected_genes = [line.strip() for line in file]

# 提取所需的基因列
selected_gene_columns = clinical_data[selected_genes]

# 将MSI列和基因列合并为一个新的DataFrame
merged_data = pd.concat([msi_column, selected_gene_columns], axis=1)


# 保存提取的数据到新的CSV文件
merged_data.to_csv('/home/xkj/project/msi/data/extracted_GSE39084.csv', index=False)

print("数据提取完成，结果已保存到 'extracted_GSE39084.csv'")

