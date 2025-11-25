import os
import pandas as pd

# 读取genes.txt并创建别名到基因名的映射字典
gene_alias_map = {}
with open('genes.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        gene_name = parts[0]
        aliases = parts[1].split(',')
        for alias in aliases:
            gene_alias_map[alias.strip()] = gene_name

# 获取所有以GSE开头的CSV文件
csv_files = [f for f in os.listdir() if f.startswith('GSE') and f.endswith('.csv')]

# 处理每个CSV文件
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # 替换别名为基因名
    df = df.rename(columns=gene_alias_map)
    df = df.applymap(lambda x: gene_alias_map.get(x, x) if isinstance(x, str) else x)
    
    # 保存修改后的CSV文件
    df.to_csv(csv_file, index=False)

print("All CSV files have been processed.")

