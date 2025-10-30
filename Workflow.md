# Midterm Workflow & Code Walkthrough
**Student Notebook:** `midterm_codes_2023200080.ipynb.ipynb`

本说明文档逐段解析你的 Notebook，逐条映射到作业要求（数据→特征工程→建模→调参→评估→提交），并给出改进建议与避免数据泄漏的核查要点。

## 0) 总览 / Checklist 对照
- 读取数据：已发现（read_csv 调用 2 次, read_excel 调用 0 次）
- 去除泄漏特征（如社区价）：疑似已处理
- 划分训练/测试：已使用 train_test_split，random_state==111：是
- 特征工程：log(True) / 多项式(True) / 交互项(True) / 分箱(True) / 哑变量(False)
- 特征选择：相关性(True) / VIF(False) / Lasso(True)
- 异常值处理：IQR(True) / Z-score(False)
- 线性模型：OLS(True) / LASSO(True) / Ridge(True) / ElasticNet(True)
- 调参：GridSearchCV(True)，交叉验证折数：未检测到固定数
- 指标：MAE(True) / RMSE(True) / R²(True)
- 预测导出：有；文件名是否为 prediction.csv：否/未检测到

## 1) 依赖与库导入（Imports）
检测到主要库：
chardet, d, matplotlib.pyplot, numpy, os, pandas, re, s, seaborn, warnings

## 2) 数据读取与基础处理
- `read_csv` 参数示例（共 2 处）：
  1. `pd.read_csv(file_path, encoding=encoding, low_memory=False)`
  2. `pd.read_csv(file_path, encoding=encoding, low_memory=False)`

## 3) 数据泄漏检查与处理
- 检测到以下疑似删除泄漏特征的代码片段（节选）：
```python
_, edges = pd.qcut(df_feat['建筑面积_数值'].dropna(), q=6, labels=False, retbins=True, duplicates='drop')
_, edges = pd.qcut(df_feat['房龄'].dropna(), q=6, labels=False, retbins=True, duplicates='drop')
numeric_cols = numeric_cols.drop(target_col)
train_df = train_df.dropna(subset=[target_col])
train_df = train_df.dropna(subset=[target_col])
train_df = train_df.drop([target_col], axis=1)
```

## 4) 训练/测试集划分
- 发现 `train_test_split(...)` 调用，部分参数片段如下：
  1. `
                X_price_train, y_price_train, test_size=0.2, random_state=111,
                shuffle=True
            `
  2. `
                X_rent_train, y_rent_train, test_size=0.2, random_state=111,
                shuffle=True
            `
- 是否满足作业 **random_state==111**：**是**。

## 5) 特征工程（创建与编码）
- 已检测到：对偏态变量做对数/对数+1 变换；`PolynomialFeatures` 生成多项式与（可选）交互项；显示地构造交互项；分箱（`pd.cut`/`pd.qcut`/`KBinsDiscretizer`）

## 6) 特征选择与多重共线性
- 已检测到：基于相关性筛选/热力图；利用 Lasso 权重稀疏性做特征选择

## 7) 异常值检测与处理
- 已检测到：IQR/分位数法
  注意：仅在训练集上拟合阈值，再应用到验证/测试集，避免泄漏。

## 8) 线性建模与调参
- 已使用模型：LinearRegression (OLS), Lasso, Ridge, Elastic Net
- 是否使用 `GridSearchCV`：是；交叉验证折数：未固定/未检测到
- 模型 `.fit(...)` 调用次数（粗略）：5

## 9) 评估指标与展示
- 已计算的指标：MAE（作业强制）, MSE/RMSE, R²
- 请按要求报告：In-sample / Out-of-sample / **6 折**交叉验证；并统计去除异常值后的样本量。

**Notebook 输出中与指标相关的片段（节选）**：
```text
✅ OLS_Enhanced: 训练MAE = 627,809.12, 验证MAE = 631,630.11
✅ Ridge_Enhanced: 训练MAE = 946,987.50, 验证MAE = 931,591.82
✅ Lasso_Enhanced: 训练MAE = 855,901.17, 验证MAE = 845,393.49
✅ ElasticNet_Enhanced: 训练MAE = 856,609.34, 验证MAE = 846,269.26
📊 验证集MAE: 631,630.11
⭐ Kaggle分数: 0.0
✅ OLS_Enhanced: 训练MAE = 188,083.96, 验证MAE = 192,153.43
✅ Ridge_Enhanced: 训练MAE = 260,237.14, 验证MAE = 263,269.66
✅ Lasso_Enhanced: 训练MAE = 250,001.04, 验证MAE = 252,931.33
✅ ElasticNet_Enhanced: 训练MAE = 249,997.11, 验证MAE = 252,932.67
📊 验证集MAE: 192,153.43
⭐ Kaggle分数: 0.0
   验证集MAE: 631,630.11
   Kaggle分数: 0.0
   验证集MAE: 192,153.43
   Kaggle分数: 0.0
```

## 10) Kaggle 提交流水线
- 已发现 `to_csv(...)` 导出；请确保文件名为 `prediction.csv` 且列名/格式符合比赛要求。

## 11) 数据泄漏风险控制（强制）
- **所有**预处理（标准化、分箱边界、编码器、PCA/多项式、特征选择阈值等）须放入 `Pipeline` / `ColumnTransformer` 并仅基于训练集拟合。
- 你已经显式删除了疑似泄漏列；请继续检查是否存在基于目标的聚合统计或时间穿越。

## 12) 不足与可改进点（清单）
1. 将交叉验证折数统一设为 `cv=6`。
2. 对类别变量进行 one-hot 编码并保持列对齐。
3. 计算 VIF 并剔除高共线特征（阈值 5~10）。
4. 导出 `prediction.csv` 并遵循 Git/Kaggle 提交规范。

## 13) 随机种子与模型一览
- 出现过的 `random_state` 值：[111]
- 模型覆盖：OLS, LASSO, Ridge, ElasticNet

## 附：无泄漏 Pipeline + GridSearchCV 模板（可直接改造）
```python

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

num_cols = [...]   # 数值列
cat_cols = [...]   # 类别列

num_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False))
])

cat_pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=10000),
    "ElasticNet": ElasticNet(max_iter=10000),
}

param_grid = {
    "Ridge": {
        "pre__num__poly__degree": [1, 2],
        "model__alpha": [0.1, 1.0, 10.0]
    },
    "Lasso": {
        "pre__num__poly__degree": [1, 2],
        "model__alpha": [0.0001, 0.001, 0.01, 0.1]
    },
    "ElasticNet": {
        "pre__num__poly__degree": [1, 2],
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
        "model__l1_ratio": [0.2, 0.5, 0.8]
    }
}

# 单模型示例（Ridge）
pipe = Pipeline([("pre", pre), ("model", models["Ridge"])])

gcv = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid["Ridge"],
    scoring="neg_mean_absolute_error",
    cv=6,
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
gcv.fit(X_train, y_train)

y_pred_tr = gcv.predict(X_train)
y_pred_te = gcv.predict(X_test)
print("MAE in-sample:", mean_absolute_error(y_train, y_pred_tr))
print("MAE out-of-sample:", mean_absolute_error(y_test, y_pred_te))
print("Best params:", gcv.best_params_)

# Kaggle 导出（示意）
# sub = pd.DataFrame({'Id': test_id, 'price': gcv.predict(X_test_like_competition)})
# sub.to_csv('prediction.csv', index=False)

```