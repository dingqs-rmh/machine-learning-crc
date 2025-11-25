import pandas as pd

# 读取CSV，跳过首行，假设第一列为数据列
df = pd.read_csv('READ_TMB.csv', skiprows=1, header=None)
first_column = df.iloc[:, 0]

count_gt10 = (first_column > 10).sum()
count_lt10 = (first_column <= 10).sum()

print(f"大于10的个数：{count_gt10}")
print(f"小于10的个数：{count_lt10}")
