# 第 7 週作業：決策樹與分類預測模板

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch9 決策樹、Ch10 分類預測模板 |
| 繳交方式 | 在 Fork 的 week07/ 資料夾中建立三個檔案，發 PR 繳交 |
| 繳交期限 | 下週上課前 |
| PR 標題格式 | 學號_姓名_week07 |

---

## 第 1 題：決策樹分類與特徵重要性分析（40 分）

### 任務說明

使用決策樹對 Scikit-learn 內建的 wine 資料集進行分類。觀察不同 max_depth 對過擬合的影響，並分析特徵重要性。

### Python 程式要求

撰寫程式碼完成以下工作：

1. 載入 wine 資料集，切割資料（test_size=0.3, random_state=42）
2. 訓練一棵不限深度的決策樹，記錄訓練集與測試集準確率
3. 分別訓練 max_depth=2, 3, 5, 7, 10, None 的決策樹，記錄訓練/測試準確率
4. 繪製折線圖：X 軸為 max_depth，Y 軸為準確率，兩條線分別代表訓練集和測試集
5. 用最佳 max_depth 訓練模型，印出特徵重要性排名（前 5 名）
6. 使用 export_graphviz 或 plot_tree 視覺化決策樹（max_depth=3 的版本即可）

### 作答內容

請建立 `week07/q1_decision_tree.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 不同 max_depth 的準確率比較 ===
max_depth=2：訓練=???  測試=???
max_depth=3：訓練=???  測試=???
max_depth=5：訓練=???  測試=???
max_depth=7：訓練=???  測試=???
max_depth=10：訓練=???  測試=???
max_depth=None：訓練=???  測試=???

=== 最佳 max_depth ===
最佳 max_depth：???
測試集準確率：???

=== 特徵重要性前 5 名 ===
1. ???
2. ???
3. ???
4. ???
5. ???

=== 過擬合觀察 ===
（從折線圖觀察，說明 max_depth 太大或不限制時，訓練與測試準確率的差距變化）
```

### 提示

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

depths = [2, 3, 5, 7, 10, None]
train_scores = []
test_scores = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

# 特徵重要性
importances = dt.feature_importances_
feat_imp = pd.Series(importances, index=wine.feature_names).sort_values(ascending=False)
print(feat_imp.head())

# 視覺化
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, max_depth=3)
plt.show()
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 多種 max_depth 的準確率計算正確 | 10 分 |
| 繪製訓練/測試準確率折線圖 | 10 分 |
| 特徵重要性分析正確 | 10 分 |
| 決策樹視覺化與過擬合觀察 | 10 分 |

---

## 第 2 題：分類預測模板 — 多模型比較（40 分）

### 任務說明

建立一套可重複使用的分類預測模板，自動處理數值與類別欄位，並一次比較多種分類模型的效能。

### 測試資料

請使用以下程式碼建立測試資料：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 400

# 模擬船舶設備故障預測資料
data = {
    'engine_hours': np.random.randint(100, 20000, n),
    'vibration_level': np.round(np.random.uniform(0.1, 8.0, n), 2),
    'oil_temp': np.round(np.random.uniform(60, 120, n), 1),
    'rpm': np.random.randint(500, 3000, n),
    'engine_type': np.random.choice(['diesel', 'gas_turbine', 'steam'], n, p=[0.5, 0.3, 0.2]),
    'maintenance_grade': np.random.choice(['A', 'B', 'C'], n, p=[0.3, 0.5, 0.2]),
}
df = pd.DataFrame(data)

# 加入遺漏值
df.loc[np.random.choice(n, 20, replace=False), 'oil_temp'] = np.nan
df.loc[np.random.choice(n, 10, replace=False), 'maintenance_grade'] = np.nan

# 故障與否
prob = 1 / (1 + np.exp(-(
    0.0002 * df['engine_hours'] +
    0.3 * df['vibration_level'] +
    0.02 * df['oil_temp'].fillna(90) -
    0.001 * df['rpm'] - 2.5
)))
df['is_fault'] = (np.random.random(n) < prob).astype(int)

df.to_csv('engine_fault.csv', index=False)
print(f"資料形狀：{df.shape}")
print(f"\n故障比例：\n{df['is_fault'].value_counts(normalize=True)}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 讀取資料，自動區分數值型欄位和類別型欄位
2. 建立 ColumnTransformer 處理混合型資料
3. 建立至少四種分類模型的 Pipeline：
   - LogisticRegression
   - KNeighborsClassifier
   - SVC
   - DecisionTreeClassifier
4. 對每個模型訓練並計算測試集準確率
5. 印出所有模型的準確率比較表，由高到低排序
6. 對最佳模型印出 Classification Report

### 作答內容

請建立 `week07/q2_model_comparison.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 欄位自動分類結果 ===
數值型欄位：???
類別型欄位：???

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 模型準確率比較表（由高到低）===
1. ???：???
2. ???：???
3. ???：???
4. ???：???

=== 最佳模型的 Classification Report ===
（貼上最佳模型的分類報告）
```

### 提示

```python
# 自動區分欄位
num_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['is_fault']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# 多模型比較
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'LogisticRegression': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    pipe.fit(X_train, y_train)
    results[name] = pipe.score(X_test, y_test)

# 排序印出
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}：{score:.4f}")
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 自動區分欄位，ColumnTransformer 正確 | 10 分 |
| 四種模型 Pipeline 建立且訓練成功 | 10 分 |
| 準確率比較表完整且排序正確 | 10 分 |
| 最佳模型的 Classification Report 完整 | 10 分 |

---

## 第 3 題：決策樹觀念題（20 分）

### 作答內容

請建立 `week07/q3_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：決策樹為什麼容易過擬合？
    max_depth 如何幫助控制過擬合？
    除了 max_depth，還有什麼參數可以用來防止過擬合？
A1：???

Q2：決策樹的 feature_importances_ 代表什麼意義？
    如果某個特徵的重要性為 0，代表什麼？
    你覺得特徵重要性分析對實務應用有什麼幫助？
A2：???

Q3：這週你比較了多種分類模型。在實際應用中，選擇模型時
    除了準確率之外，還應該考慮哪些因素？請至少提出兩點。
A3：???
```

### 評分標準

| 項目 | 配分 |
|------|------|
| Q1 正確說明過擬合原因與控制方式 | 7 分 |
| Q2 正確解釋特徵重要性意義與實務價值 | 7 分 |
| Q3 提出至少兩個模型選擇的考量因素 | 6 分 |

---

## 繳交 Checklist

- [ ] week07/q1_decision_tree.txt 包含完整程式碼、max_depth 比較與特徵重要性
- [ ] week07/q2_model_comparison.txt 包含完整程式碼與四種模型比較
- [ ] week07/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork
- [ ] 已發 PR，標題格式：學號_姓名_week07
