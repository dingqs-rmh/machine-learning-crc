import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.feature_selection import RFE
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multitest

# 设定输出目录
output_dir = '/home/xkj/project/TMB/preprocess/'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
na_values = ['', ' ', 'NA']
df = pd.read_csv('/home/xkj/project/TMB/data/COAD_TMB.csv', na_values=na_values)
test1 = pd.read_csv('/home/xkj/project/TMB/data/READ_TMB.csv', na_values=na_values)

# 提取每个DataFrame的共同列
common_columns = df.columns.intersection(test1.columns)

# 过滤共同列
df = df[common_columns]
test1 = test1[common_columns]

# 合并数据集
merged_df = pd.concat([df, test1], ignore_index=True)

# 分离标签和特征
threshold = 3.47
y_train = (merged_df.iloc[:, 0] > threshold).astype(int)  # 高TMB为1，低TMB为0
y_merged = merged_df['TMB']  
X_merged = merged_df.drop(columns=['TMB'])  

# 过滤掉包含缺失值的行
X_merged = X_merged.dropna()
y_merged = y_merged.loc[X_merged.index]  # 确保y和X的索引对齐

# # 计算过滤前的基因数量
# initial_gene_count = X_merged.shape[1]

# # 过滤低表达基因
# min_tpm_threshold = 1
# filtered_X = X_merged.loc[:, (X_merged > min_tpm_threshold).sum() > (X_merged.shape[0] * 0.2)]
# X_merged = filtered_X

# # 计算过滤后的基因数量
# filtered_gene_count = X_merged.shape[1]

# # 输出结果
# print(f"过滤低表达前的基因数量: {initial_gene_count}")
# print(f"过滤低表达后的基因数量: {filtered_gene_count}")

# 计算每个基因的平均表达量
mean_expression = X_merged.mean(axis=0)

# 根据平均表达量进行排序
sorted_genes = mean_expression.sort_values(ascending=False)

# 选择前50%的基因
top_50_percent_genes = sorted_genes.index[:int(len(sorted_genes) * 0.5)]

# 保留表达最高的50%基因
X_merged = X_merged[top_50_percent_genes]
top_50_percent_gene_count = X_merged.shape[1]
print(f"保留表达最高的50%基因后的基因数量: {top_50_percent_gene_count}")

# 使用t检验筛选差异基因
uc_group = X_merged[y_merged == 1]
cd_group = X_merged[y_merged == 0]

p_values = []
gene_names = X_merged.columns

for gene in gene_names:
    t_stat, p_val = ttest_ind(uc_group[gene], cd_group[gene])
    p_values.append(p_val)

# 调整p值
adjusted_p_values = multitest.multipletests(p_values, method='fdr_bh')[1]

# 创建结果DataFrame
results = pd.DataFrame({
    'gene': gene_names,
    'p_value': p_values,
    'adjusted_p_value': adjusted_p_values
})

# 选择p值最低的前100个基因
top_n_genes = 100
top_genes = results.nsmallest(top_n_genes, 'p_value')['gene']

# 保留显著基因
X_train = df[top_genes]
X_test1 = test1[top_genes]

# 分离训练集的标签
y_train = df['Target']
# 分离测试集的标签
y_test1 = test1['Target'] 

# 使用SimpleImputer填补缺失值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test1_imputed = imputer.transform(X_test1)

# 对特征进行log变换和标准化
scaler = StandardScaler()

X_train_log = np.log1p(X_train_imputed)
X_test1_log = np.log1p(X_test1_imputed)

X_train_standardized = scaler.fit_transform(X_train_log)
X_test1_standardized = scaler.transform(X_test1_log)

# 使用Lasso, Ridge和Elastic Net进行特征选择
lasso = LassoCV(cv=5, random_state=42, max_iter=50000).fit(X_train_standardized, y_train)
ridge = RidgeCV(cv=5).fit(X_train_standardized, y_train)
elastic_net = ElasticNetCV(cv=5, random_state=42, max_iter=50000).fit(X_train_standardized, y_train)

# 获取选择的特征索引
lasso_selected = set(np.where(lasso.coef_ != 0)[0])
ridge_selected = set(np.where(ridge.coef_ != 0)[0])
elastic_net_selected = set(np.where(elastic_net.coef_ != 0)[0])

# 找到共同选择的特征
common_selected_features = list(lasso_selected & ridge_selected & elastic_net_selected)
common_selected_feature_names = np.array(top_genes)[common_selected_features]

# 输出选择的特征数目
print(f"Lasso选择的特征数目: {len(lasso_selected)}")
print(f"Ridge选择的特征数目: {len(ridge_selected)}")
print(f"Elastic Net选择的特征数目: {len(elastic_net_selected)}")
print(f"共同选择的特征数目: {len(common_selected_features)}")

# 如果需要输出共同选择的特征名称，可以使用以下代码
print(f"共同选择的特征名称: {common_selected_feature_names}")

# 如果共同选择的特征多于30个，使用递归特征消除进一步减少特征数量
if len(common_selected_features) > 30:
    lasso_rfe = RFE(lasso, n_features_to_select=30)
    lasso_rfe.fit(X_train_standardized[:, common_selected_features], y_train)
    common_selected_features = np.where(lasso_rfe.support_)[0]
    common_selected_feature_names = np.array(common_selected_feature_names)[common_selected_features]

     # 输出RFE选择的特征数目和名称
    print(f"RFE选择的特征数目: {len(common_selected_features)}")
    print(f"RFE选择的特征名称: {common_selected_feature_names}")

# 使用共同选择的特征更新数据集
X_train_selected = X_train_standardized[:, common_selected_features]
X_test1_selected = X_test1_standardized[:, common_selected_features]

# 保存预处理后的数据
train_df = pd.DataFrame(X_train_selected, columns=common_selected_feature_names)
train_df['Target'] = y_train.values
train_df.to_csv('/home/xkj/project/Ischemic_cardiomyopathy/data/train_data/train.csv', index=False)

test1_df = pd.DataFrame(X_test1_selected, columns=common_selected_feature_names)
test1_df['Target'] = y_test1.values
test1_df.to_csv('/home/xkj/project/Ischemic_cardiomyopathy/data/test_data/test.csv', index=False)


print('数据预处理完成，训练集和测试集已保存为文件。')
