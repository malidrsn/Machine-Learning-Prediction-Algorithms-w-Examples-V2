# y=ax+b
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Veri Yükleme
veriler = pd.read_csv("satislar.csv")

aylar = veriler[['Aylar']].values.reshape(-1, 1)
print(aylar)

satislar = veriler.Satislar.values.reshape(-1, 1)
print(satislar)

# satislar2 = veriler.iloc[:, 1:].values
# print("Satışlar2 ", satislar2)


# önemli test ve eğitim verileri olarak bölmeye yarar eğitilecek 2/3 test 1/3 olacak şekilde
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)
print("1 : x_train ", x_train)
print("2 : x_test ", x_test)
print("3 : y_train ", y_train)
print("4 : y_test ", y_test)

# verilerin standartscaler küütphanesi ile ölçeklenmesi standart sapma kullandık
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# print(X_test)
print("--------------------------------")
# print("test edilen veriler", X_test)
print("--------------------------------")
# print("Eğitilen veriler", X_train)

# Lineer Regression için model oluşturma kısmı model inşasıdır
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)  # model inşa etme kısmı lineer model eğitme

tahmin = lr.predict(x_test)
print("Tahmin Sonuçları\n", tahmin)
print("Orjinal Değerler\n", y_test)

# ölçekleme yapmak zorunda değiliz direk aylar ve satış fiyatları üzerinden de tahmin yapabilmekteyiz.

# verilerimizi görselleştirelim fakat önce verilerimizi sort edelim
x_train = sorted(x_train, reverse=True)
y_train = sorted(y_train, reverse=True)

# Başlık
plt.title("Aylara Göre Satış")
# x ekseni
plt.xlabel("Aylar")
# y ekseni
plt.ylabel("Satışlar")

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.show()


