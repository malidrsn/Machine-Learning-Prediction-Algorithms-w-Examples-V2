# 1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Veri Onisleme

# 2.1. Veri Yukleme
veriler = pd.read_csv('tenis.csv')
# pd.read_csv("veriler.csv")


# veri on isleme

# encoder:  Kategorik -> Numeric
# tüm veriler encode edilmiştir
print("***********************************************")
from sklearn.preprocessing import LabelEncoder

verilerApply = veriler.apply(LabelEncoder().fit_transform)
print(verilerApply)

# sadece nümerik veriler label encode işlemine tabi tutulmamalıdır.
# bu yüzden aşağıdaki ohe uygulamaktayız
from sklearn.preprocessing import OneHotEncoder

outlook = verilerApply.iloc[:, :1]
ohe = OneHotEncoder(categories="auto")
outlook = ohe.fit_transform(outlook).toarray()
print("C : ", outlook)

havadurumu = pd.DataFrame(data=outlook, index=range(14), columns=['o', 'r', 's'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis=1)
sonveriler = pd.concat([verilerApply.iloc[:, -2:], sonveriler], axis=1)
print("son veriler : ", sonveriler)

# verilerin egitim ve test icin bolunmesi
# humidity bağımlı diğerleri bağımsız değişken olacak
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1:], test_size=0.33,
                                                    random_state=0)  # split içerisinde ilk parametre bağımsız ikinci parametre bağımlı değişkendir
print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Prediction edilenler\n", y_pred)

# backward elimination yöntemi başlangıç noktası
import statsmodels.api as sm

# X = ax+b de ki b tam sayı kısmıdır 14 olması csv 14 satırdır
X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
X_list = sonveriler.iloc[:, [0, 1, 2, 3, 4, 5]].values
regression_ols = sm.OLS(endog=sonveriler.iloc[:, -1:], exog=X_list)
r = regression_ols.fit()
print("Tüm değerler", r.summary())

sonveriler = sonveriler.iloc[:,
             1:]  # ilk kolonu atar diğerleri kalır ama aşağı kısımda indexler taşınır yine 0dan başlar

# x1'in çıkarıldığı sistem
import statsmodels.api as sm

print("****************************************************")
print("x1'in atıldığı kısımdır")
# X = ax+b de ki b tam sayı kısmıdır
X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
X_l = sonveriler.iloc[:, [0, 1, 2, 3, 4]].values
r_ols = sm.OLS(endog=sonveriler.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

# son tahminleme kısmıdır baştan tahmin edip veri çıkarıldıktan sonra ki halini karşılaştırıyoruz
print("Önceki prediction", y_pred)
# x train kısmından 1.sütun çıkarılınca daha doğru olacağı için tekrar test edilmesini sağlar
x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Son prediction değerleri : ", y_pred)
print(y_test)
