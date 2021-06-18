import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Veri Yükleme
veriler = pd.read_csv("maaslar.csv")

# dataframe dilimleme(slice)
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
print(x)  # eğitim seviyesi
print(y)  # maas

# Linear Regression
from sklearn.linear_model import LinearRegression

line_reg = LinearRegression()

# numpy array dönüşümü
X = x.values
Y = y.values

line_reg.fit(X, Y)

plt.scatter(X, Y, color="red")
plt.plot(x, line_reg.predict(X), color="blue")

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)  # degree üstel sayıdır artarsa polinom çizimi daha doğru sonuç vermektedir.

x_poly = poly_reg.fit_transform(X)
# print(x_poly)

line_reg2 = LinearRegression()
line_reg2.fit(x_poly, y)
plt.scatter(X, Y)
plt.plot(X, line_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

# tahminler
# linear
print(line_reg.predict([[11]]))
print(line_reg.predict([[6.6]]))

# polynomial
print(line_reg2.predict(poly_reg.fit_transform([[11]])))
print(line_reg2.predict(poly_reg.fit_transform([[6.6]])))

# support vector regression
# ölçeklendirmek gereklidir.
# verilerin standartscaler küütphanesi ile ölçeklenmesi standart sapma kullandık
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color="red")
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color="blue")
plt.show()
print(svr_reg.predict([[11]]))

from sklearn.tree import DecisionTreeRegressor

dtree_reg = DecisionTreeRegressor(random_state=0)
dtree_reg.fit(X, Y)
plt.scatter(X, Y, color="blue")
plt.plot(X, dtree_reg.predict(X), color="red")
plt.show()
print("Decision Tree 11 tahmini", dtree_reg.predict([[11]]))
print("Decision Tree 6 tahmini", dtree_reg.predict([[6.6]]))
