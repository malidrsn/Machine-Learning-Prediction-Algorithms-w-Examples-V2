from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

veriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = veriler.iloc[:, 1:4].values
# print(yas)

imputer = imputer.fit(yas[:, 1:4])
yas[:, 1:4] = imputer.transform(yas[:, 1:4])
# print(yas)

ulke = veriler.iloc[:, 0:1].values
# print(ulke)

from sklearn.preprocessing import LabelEncoder  # kategorik verileri numerik verilere eşitliyor

le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:, 0])
# print(ulke)

from sklearn.preprocessing import OneHotEncoder  # dizi mantığı gibi 1 0 0 ekliyor dönüşüm yapıyor

ohe = OneHotEncoder(categories="auto")
ulke = ohe.fit_transform(ulke).toarray()
# print(ulke)

# verilerin birleştirilmesi
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
# print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(22), columns=['boy', 'kilo', 'yas'])
# print(sonuc2)

cinsiyet = veriler.iloc[:, -1:].values
# print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
# print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
# print(s)

s2 = pd.concat([s, sonuc3], axis=1)
# print(s2)
# önemli test ve eğitim verileri olarlak bölmeye yarar eğitilecek 2/3 test 1/3 olacak şekilde
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# ölcekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# print(X_test)
print(X_test)
print("--------------------------------")
print(X_train)
