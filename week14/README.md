# 第 14 週｜作業
- 修改課本第12章程式碼，運用AI製作修改程式記錄的MD檔
  - 修改了什麼
  - 該程式區塊的功能說明
- 放置於README.MD中

# W14-Exercise_C113181120
## Error01
ValueError: could not convert string to float: 'sales' 表示 df.corr() 方法嘗試對非數值型資料（特別是 'sales' 欄位，它包含字串值）計算相關性。corr() 方法只能處理數值型資料。
為了解決這個問題，我修改了程式碼，在 df.corr() 方法中添加了 numeric_only=True。這樣做會指示 corr() 方法只考慮資料框中的數值型欄位來計算相關性，從而避免了嘗試將字串轉換為浮點數的錯誤。
```
#sns.heatmap(df.corr().round(2), annot=True, cmap='coolwarm');
sns.heatmap(df.corr(numeric_only=True).round(2), annot=True, cmap='coolwarm');
```
這行程式碼的目的是繪製一個相關性熱力圖（heatmap），用來視覺化資料框 df 中各數值型欄位之間的相關性。具體來說：
- df.corr(numeric_only=True): 計算資料框 df 中所有數值型欄位之間的皮爾遜相關係數。numeric_only=True 確保只對數值型欄位進行計算，避免非數值型欄位導致的錯誤。
- .round(2): 將相關係數的結果四捨五入到小數點後兩位，使圖表更易讀
- sns.heatmap(...): 使用 Seaborn 函式庫中的 heatmap 函式來繪製熱力圖。
- annot=True: 在熱力圖的每個單元格中顯示相關係數的數值。
- cmap='coolwarm': 設定熱力圖的顏色映射。'coolwarm' 是一種常見的色譜，通常用於表示正相關（暖色）和負相關（冷色），並且中間色表示接近零的相關性。
keep
## Error02
這個錯誤 ValueError: could not convert string to float: 'sales' 表示 corrwith() 方法嘗試將非數值型的 'sales' 欄位（以及其他非數值型欄位，例如 'salary'）轉換為浮點數來計算相關性，但這是不可能的。相關性計算只能針對數值型資料執行。為了修正這個問題，我會修改程式碼，確保在計算相關性之前，只選取資料框中的數值型欄位。
```
# df.drop('left', axis=1).corrwith(df['left']).round(2)
df.drop('left', axis=1).select_dtypes(include=np.number).corrwith(df['left']).round(2)
```
這段程式碼計算並顯示了資料集中所有數值型特徵與目標變數 left（是否離職）之間的相關係數，並且結果四捨五入到小數點後兩位。

這些相關係數的意義是：

- satisfaction_level (滿意度): -0.39 - 呈現負相關，表示員工滿意度越低，離職的可能性越高。
- last_evaluation (上次考評): 0.01 - 相關性非常弱，幾乎沒有線性關係。
- number_project (專案數量): 0.02 - 相關性非常弱，幾乎沒有線性關係。
- average_montly_hours (平均月工時): 0.07 - 相關性非常弱，幾乎沒有線性關係。
- time_spend_company (在公司待的時間): 0.14 - 呈現弱的正相關，表示在公司待的時間越長，離職的可能性稍高（這可能需要進一步分析，例如考慮過長時間工作等因素）。
- Work_accident (工作事故): -0.15 - 呈現弱的負相關，表示發生工作事故的員工離職的可能性略低。
- promotion_last_5years (近5年是否晉升): -0.06 - 呈現弱的負相關，表示近5年有晉升的員工離職的可能性略低。
總體來看，satisfaction_level 與離職率有最強的負相關，意味著員工滿意度是影響離職與否的重要因素。其他數值型特徵與離職率的線性相關性較弱。
keep
## Error03
這個錯誤是因為您嘗試在 accuracy_score 函數返回的浮點數上直接使用 .round() 方法。在 Python 中，通常使用內建的 round() 函數來處理單個浮點數的四捨五入。我會修改程式碼以使用正確的 round() 函數。
‵‵`
##from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
dmy = DummyClassifier(strategy='most_frequent')
dmy.fit(X_train, y_train)
dmy.score(X_train, y_train)
y_pred = dmy.predict(X_test)
print('正確率：', accuracy_score(y_test, y_pred).round(2))
print('混亂矩陣')
print(confusion_matrix(y_test, y_pred))
print('綜合報告')
print(classification_report(y_test, y_pred))

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
dmy = DummyClassifier(strategy='most_frequent')
dmy.fit(X_train, y_train)
dmy.score(X_train, y_train)
y_pred = dmy.predict(X_test)
print('正確率：', round(accuracy_score(y_test, y_pred), 2))
print('混亂矩陣')
print(confusion_matrix(y_test, y_pred))
print('綜合報告')
print(classification_report(y_test, y_pred))
‵‵‵
satisfaction_level (滿意度) : -0.39 - 呈現中等強度的負相關。這意味著員工的滿意度越低，他們離職的可能性越高。
last_evaluation (上次考評) : 0.01 - 幾乎沒有相關性。考評結果與離職的線性關係不明顯。
number_project (專案數量) : 0.02 - 幾乎沒有相關性。專案數量與離職的線性關係不明顯。
average_montly_hours (平均每月工時) : 0.07 - 幾乎沒有相關性。平均每月工時與離職的線性關係不明顯。
time_spend_company (在公司待的時間) : 0.14 - 呈現弱的正相關。這可能意味著在公司待的時間越長，離職的可能性略微增加（這在某些情況下可能代表倦怠或尋求新機會）。
Work_accident (工作事故) : -0.15 - 呈現弱的負相關。這表示發生工作事故的員工，離職的可能性略微降低（這可能與公司在事故後的補償或支持有關）。
promotion_last_5years (過去5年是否有晉升) : -0.06 - 幾乎沒有相關性。過去5年是否晉升與離職的線性關係不明顯。

##Error04

這個錯誤是因為您嘗試在 score 變數（它是一個浮點數）上直接使用 .round() 方法。在 Python 中，通常使用內建的 round() 函數來處理單個浮點數的四捨五入。我會修改程式碼以使用正確的 round() 函數。

```
# 載入所有模型
##from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
# 載作Pipeline，PCA和GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

model_pl = Pipeline([
    ('preprocess', data_pl),
    ('model', LogisticRegression())
])
param_grid = {'model':[LogisticRegression(), SVC(),
              KNeighborsClassifier(), DecisionTreeClassifier(max_depth=10)]}
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預測參數', gs.best_params_)
print('訓練集交叉驗證的最佳結果', gs.best_score_.round(3))
print('測試集的結果', score.round(3))
y_pred = gs.best_estimator_.predict(X_test)
print('混亂矩陣\n',confusion_matrix(y_test, y_pred))


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
# 載作Pipeline，PCA和GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

model_pl = Pipeline([
    ('preprocess', data_pl),
    ('model', LogisticRegression())
])
param_grid = {'model':[LogisticRegression(), SVC(),
              KNeighborsClassifier(), DecisionTreeClassifier(max_depth=10)]}
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預測參數', gs.best_params_)
print('訓練集交叉驗證的最佳結果', round(gs.best_score_, 3))
print('測試集的結果', round(score, 3))
y_pred = gs.best_estimator_.predict(X_test)
print('混亂矩陣\n',confusion_matrix(y_test, y_pred))
```
satisfaction_level (滿意度): -0.39

這表示滿意度與離職之間存在中等強度的負相關。當員工的滿意度越高，他們離職的可能性越低，反之亦然。
last_evaluation (上次考評): 0.01

這表示上次考評與離職之間幾乎沒有線性關係，相關性非常弱。
number_project (專案數量): 0.02

這表示專案數量與離職之間幾乎沒有線性關係，相關性非常弱。
average_montly_hours (平均月工時): 0.07

這表示平均月工時與離職之間存在非常弱的正相關，即月工時略高可能會輕微增加離職的可能性。
time_spend_company (在公司待的時間): 0.14

這表示在公司待的時間與離職之間存在弱的正相關，即在公司待的時間越長，離職的可能性略微增加。這可能有點反直覺，但有時在公司待久了卻沒有升遷或發展，也可能導致離職。
Work_accident (工作事故): -0.15

這表示工作事故與離職之間存在弱的負相關。這可能意味著發生過工作事故的員工離職的可能性略低，這也許是因為他們可能得到了公司的補償或支持，或者這是一個不那麼直接影響離職的因素。
promotion_last_5years (過去5年是否有晉升): -0.06

這表示過去5年是否有晉升與離職之間存在非常弱的負相關，即有晉升的員工離職的可能性略低，但影響很小。

##Error05
這個錯誤是因為您嘗試在浮點數上直接使用 .round() 方法。Python 中的浮點數物件本身沒有這個方法。您應該使用內建的 round() 函數來對浮點數進行四捨五入。我會修改程式碼，將 gs.best_score_.round(3) 和 score.round(3) 改為 round(gs.best_score_, 3) 和 round(score, 3)。
‵‵`
##model_pl = Pipeline([
    ('preprocess', data_pl),
    ('model', LogisticRegression())
])
np.random.seed(42)
param_grid = {'model':[RandomForestClassifier(), AdaBoostClassifier(),
                       BaggingClassifier(), XGBClassifier()]}
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預測參數', gs.best_params_)
print('訓練集交叉驗證的最佳結果', gs.best_score_.round(3))
print('測試集的結果', score.round(3))
y_pred = gs.best_estimator_.predict(X_test)
print('混亂矩陣\n',confusion_matrix(y_test, y_pred))

model_pl = Pipeline([
    ('preprocess', data_pl),
    ('model', LogisticRegression())
])
np.random.seed(42)
param_grid = {'model':[RandomForestClassifier(), AdaBoostClassifier(),
                       BaggingClassifier(), XGBClassifier()]}
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預測參數', gs.best_params_)
print('訓練集交叉驗證的最佳結果', round(gs.best_score_, 3))
print('測試集的結果', round(score, 3))
y_pred = gs.best_estimator_.predict(X_test)
print('混亂矩陣\n',confusion_matrix(y_test, y_pred))
‵‵`
satisfaction_level (滿意度): -0.39

這表示滿意度與離職之間存在中等強度的負相關。當員工的滿意度越高，他們離職的可能性越低，反之亦然。
last_evaluation (上次考評): 0.01

這表示上次考評與離職之間幾乎沒有線性關係，相關性非常弱。
number_project (專案數量): 0.02

這表示專案數量與離職之間幾乎沒有線性關係，相關性非常弱。
average_montly_hours (平均月工時): 0.07

這表示平均月工時與離職之間存在非常弱的正相關，即月工時略高可能會輕微增加離職的可能性。
time_spend_company (在公司待的時間): 0.14

這表示在公司待的時間與離職之間存在弱的正相關，即在公司待的時間越長，離職的可能性略微增加。這可能有點反直覺，但有時在公司待久了卻沒有升遷或發展，也可能導致離職。
Work_accident (工作事故): -0.15

這表示工作事故與離職之間存在弱的負相關。這可能意味著發生過工作事故的員工離職的可能性略低，這也許是因為他們可能得到了公司的補償或支持，或者這是一個不那麼直接影響離職的因素。
promotion_last_5years (過去5年是否有晉升): -0.06

這表示過去5年是否有晉升與離職之間存在非常弱的負相關，即有晉升的員工離職的可能性略低，但影響很小。

