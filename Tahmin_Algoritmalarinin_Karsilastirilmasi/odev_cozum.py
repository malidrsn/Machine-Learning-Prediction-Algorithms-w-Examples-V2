# maaslar_yeni.csv kullanilarak verilerin ön işlemesi gerekli gereksiz ayrilmasini saglayin
# Daha sonra 5 farkli yontem icin (MLR PR SVR DT RF) yöntemleri karsilastiriniz.
# 10 yil tecrubeli ve 100 puan almis CEO ve Mudur için tahminleri yapiniz ve sonuclari yorumlayiniz
# csv dosyasinden unvan dummy veriable oldugundan ve calisan id kullanmamiz gerektiginden dolayi cikariyoruz.
# kullanmamamizin sebebi overfitting olma ihtimalinden dolayidir. ezber degil ogrenmesi gerekir algoritmanin

# 1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

x = veriler.iloc[:, 2:5]
y = veriler.iloc[:, 5:]
X = x.values
Y = y.values

# korelasyon matrisi
# bütün kolonların diğerleri ile ilişkilerini ve tahmin için kullanıldıklarını ön görebiliriz.
print(veriler.corr())

# linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# p value değerine ulaşıp gereksiz verileri silmek için baktığımız kısım
model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())
# p value bitiş

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# tahminler


# p value değerine ulaşıp gereksiz verileri silmek için baktığımız kısım
print("polynomial OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)), X)
print(model2.fit().summary())
# p value bitiş


print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1, 1)))

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

# p value değerine ulaşıp gereksiz verileri silmek için baktığımız kısım
print("svr OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli), x_olcekli)
print(model3.fit().summary())
# p value bitiş


print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

# Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4

# p value değerine ulaşıp gereksiz verileri silmek için baktığımız kısım
print("Decision Tree OLS")
model4 = sm.OLS(r_dt.predict(X), X)
print(model4.fit().summary())
# p value bitiş


print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

# Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())

# p value değerine ulaşıp gereksiz verileri silmek için baktığımız kısım
print("Random Forest OLS")
model5 = sm.OLS(rf_reg.predict(X), X)
print(model5.fit().summary())
# p value bitiş


print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))

# Ozet R2 değerleri
print('-----------------------')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

# 2 parametre elenirse bir artış oluyo
