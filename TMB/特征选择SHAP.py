import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from typing import List

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16
})

# ========== 数据加载与预处理 ==========
#coad = pd.read_csv('data/COAD_TMB.csv')
#read = pd.read_csv('data/READ_TMB.csv')

# 合并数据集
#data = pd.concat([coad, read], axis=0)
data = pd.read_csv('data/COAD_TMB.csv')
# 创建二分类标签
data['High_TMB'] = np.where(data.iloc[:, 0] > 10, 1, 0)
X = data.iloc[:, 1:-1]  # 基因表达量特征
y = data['High_TMB']

# 标准化器
scaler = StandardScaler()

# ========== 通用特征选择函数 ==========
def select_features(model, X, y, top_n=10) -> List[str]:
    model.fit(X, y)
    try:
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("模型不支持特征重要性提取")
        features = pd.Series(importances, index=X.columns)
        return features.nlargest(top_n).index.tolist()
    except Exception as e:
        print(f"特征选择失败: {e}")
        return []

# ========== 模型特征选择封装 ==========
def model_feature_selection(model, X, y, model_name='', scale=False, top_n=10) -> List[str]:
    if scale:
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    selected = select_features(model, X, y, top_n=top_n)
    print(f"{model_name} 选出的特征基因 Top {top_n}:", selected)
    return selected

# ========== SHAP 可视化并输出 PDF ==========
# ========== SHAP 可视化并输出 PDF ==========
def plot_shap_summary(model, X, y, features: List[str], filename: str):
    print(f"绘制 SHAP 图中: {filename} ... 使用特征数: {len(features)}")
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values[:, features], X[features], show=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ SHAP summary 图已保存为: {filename}")

def plot_shap_summary_rf(model, X, y, features: List[str], filename: str):
    print(f"绘制 SHAP 图中: {filename} ... 使用特征数: {len(features)}")
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ SHAP summary 图已保存为: {filename}")


# ========== 模型初始化 ==========
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', class_weight='balanced')
#xgb = XGBClassifier(n_estimators=100, scale_pos_weight=np.sum(y==0)/np.sum(y==1), use_label_encoder=False, eval_metric='logloss')

# ========== 特征选择 ==========
rf_features = model_feature_selection(rf, X, y, model_name='随机森林')
lasso_features = model_feature_selection(lasso, X, y, model_name='Lasso', scale=True)
#xgb_features = model_feature_selection(xgb, X, y, model_name='XGBoost')

# ========== 交集特征 ==========
common_features = list(set(rf_features) & set(lasso_features))
print("交集特征基因:", common_features)

# ========== 绘制所有 SHAP 图 ==========
plot_shap_summary_rf(rf, X, y, rf_features, "tmb_shap_rf.pdf")
plot_shap_summary(lasso, pd.DataFrame(scaler.fit_transform(X), columns=X.columns), y, lasso_features, "tmb_shap_lasso.pdf")
#plot_shap_summary(xgb, X, y, xgb_features, "tmb_shap_xgb.pdf")

from matplotlib_venn import venn2
def plot_feature_venn(rf_feats, lasso_feats, save_path="tmb_venn_features.pdf"):
    set1 = set(rf_feats)
    set2 = set(lasso_feats)
    #set3 = set(xgb_feats)

    plt.figure(figsize=(8, 6))
    venn = venn2(
        [set1, set2],
        set_labels=("Random Forest", "Lasso")
    )
    plt.title("Venn diagram of characteristic gene selection results")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Venn 图已保存为: {save_path}")

def save_venn_sets_to_csv(rf_feats, lasso_feats, path="tmb_venn_feature_sets.csv"):
    # 构建集合
    rf_set = set(rf_feats)
    lasso_set = set(lasso_feats)
    #xgb_set = set(xgb_feats)

    # 各种组合
    # only_rf = rf_set - lasso_set - xgb_set
    # only_lasso = lasso_set - rf_set - xgb_set
    # only_xgb = xgb_set - rf_set - lasso_set
    # rf_lasso = (rf_set & lasso_set) - xgb_set
    # rf_xgb = (rf_set & xgb_set) - lasso_set
    # lasso_xgb = (lasso_set & xgb_set) - rf_set
    # all_three = rf_set & lasso_set & xgb_set
    only_rf = rf_set - lasso_set
    only_lasso = lasso_set - rf_set
    all_two = rf_set & lasso_set

    # 创建 DataFrame
    df = pd.DataFrame({
        "RF only": pd.Series(sorted(only_rf)),
        "Lasso only": pd.Series(sorted(only_lasso)),
        # "XGB only": pd.Series(sorted(only_xgb)),
        # "RF ∩ Lasso only": pd.Series(sorted(rf_lasso)),
        # "RF ∩ XGB only": pd.Series(sorted(rf_xgb)),
        # "Lasso ∩ XGB only": pd.Series(sorted(lasso_xgb)),
        "All two (交集)": pd.Series(sorted(all_two))
    })

    # 保存为 CSV
    df.to_csv(path, index=False)
    print(f"📁 交集与独占特征基因已保存为: {path}")


plot_feature_venn(rf_features, lasso_features)
save_venn_sets_to_csv(rf_features, lasso_features)