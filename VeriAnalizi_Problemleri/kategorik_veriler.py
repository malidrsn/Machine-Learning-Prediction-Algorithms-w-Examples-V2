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
print(ulke)

from sklearn.preprocessing import LabelEncoder  # kategorik verileri numerik verilere eşitliyor

le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:, 0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder  # dizi mantığı gibi 1 0 0 ekliyor dönüşüm yapıyor

ohe = OneHotEncoder(categories="auto")
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
