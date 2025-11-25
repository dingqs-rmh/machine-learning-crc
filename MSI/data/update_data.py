import os
import pandas as pd
from glob import glob

# 读取selected_genes.txt文件
with open('/home/xkj/project/msi/model/selected_features.txt') as f:
    selected_genes = [line.strip() for line in f]

# 读取TCGA_COAD_clinical_msi.csv文件
tcga_df = pd.read_csv('TCGA_COAD_clinical_msi.csv')

# 获取所有GSE开头的CSV文件
gse_files = glob('gse*.csv')

for file in gse_files:
    # 读取GSE文件
    gse_df = pd.read_csv(file)

    # 对于每个selected_genes中的基因列
    for gene in selected_genes:
        if gene not in gse_df.columns:
            if gene in tcga_df.columns:
                mean_value = tcga_df[gene].mean()
                gse_df[gene] = mean_value
            else:
                gse_df[gene] = 'NA'  # 如果TCGA文件中也没有该列，则填充NA

    # 保存更新后的GSE文件
    gse_df.to_csv(file, index=False)
    print(f"文件 {file} 已更新。")

print("所有文件处理完毕。")
