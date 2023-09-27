# Scikit-learn
# Kaggle
很多數據集

很多線上比賽可以打

# 工具使用步驟
用pandas把資料讀進來

用numpy做資料處理

# 模板
### import
```
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, neighbors, 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
```

### 匯入檔案
```
df = pd.read_csv('path')
```
### missing value
```
df = df[index].dropna(axis = 0, how = 'any')
```
### 分割answer 跟data
```
# x = df.drop(index, axis = 1)
# axis = 1 代表直列，column
# axis = 0 代表橫列，row
y = df[13]
x = df.drop(13, axis = 1)
```
### Poly特有，產feature
```
# 產生degree 為 2 的feature
poly = PolynomialFeatures(degree = 2).fit(x)
x = poly.transform(x)
```
###　Split data
```
# test_size通常小於0.5
# random_state = 1，使shuffle機制停止，固定切割資料，debug可以用
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
```
### Normalization
```
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```
### Model Select
```
model = linear_model.LinearRegression()
model = LogisticRegression()
model.fit(x_train, y_train)
```
### Predict
```
# Linear
y_pred = model.predict(x_test)

print('Cofficient : {}'.format(model.coef_))

print('Mean squared error : {}'.format(mean_squared_error(y_test, y_pred)))

print('Variance score : {}'.format(r2_score(y_test, y_pred)))


# Logistic
print(model.coef_)

print(model.intercept_)

y_pred = model.predict(x_test)

print(y_pred)
accuracy = model.score(x_test, y_test)

print(accuracy)
```
