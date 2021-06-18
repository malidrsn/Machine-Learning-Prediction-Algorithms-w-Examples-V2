from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# veri Önişleme
# Veri Yükleme
veriler = pd.read_csv("veriler.csv", sep=",")
yas = veriler.iloc[:, 1:4].values
ulke = veriler.iloc[:, 0:1].values
# print(ulke)
# encoder ulke için  Kategorik verilerden -> Nümerik verilere
from sklearn.preprocessing import LabelEncoder  # kategorik verileri numerik verilere eşitliyor

le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:, 0])
# print(ulke)
# her kolon başlığına sıfır yada 1" atar
from sklearn.preprocessing import OneHotEncoder  # dizi mantığı gibi 1 0 0 ekliyor dönüşüm yapıyor

ohe = OneHotEncoder(categories="auto")
ulke = ohe.fit_transform(ulke).toarray()
# print(ulke)

# encoder cinsiyet için  Kategorik verilerden -> Nümerik verilere
c = veriler.iloc[:, -1:].values
print(c)
# encoder Kategorik verilerden -> Nümerik verilere
from sklearn.preprocessing import LabelEncoder  # kategorik verileri numerik verilere eşitliyor

le = LabelEncoder()
c[:, 0] = le.fit_transform(c[:, 0])
print(c)

# her kolon başlığına sıfır yada 1" atar
from sklearn.preprocessing import OneHotEncoder  # dizi mantığı gibi 1 0 0 ekliyor dönüşüm yapıyor

ohe = OneHotEncoder(categories="auto")
c = ohe.fit_transform(c).toarray()
print("C :", c)

# verilerin birleştirilmesi numpy dizileri dataframe oluşturma
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
print("sonuc1", sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(22), columns=['boy', 'kilo', 'yas'])
print("sonuc2 :", sonuc2)

cinsiyet = veriler.iloc[:, -1:].values  # iloc verilerin alınmasını sağlıyor
# print("Cinsiyet",cinsiyet)

sonuc3 = pd.DataFrame(data=c[:, :1], index=range(22), columns=['cinsiyet'])
print("Sonuc3:", sonuc3)

# Data Frame birleştirme işlemleri
s = pd.concat([sonuc, sonuc2], axis=1)
print(s)
s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

print("--------------------------------")
# önemli test ve eğitim verileri olarlak bölmeye yarar eğitilecek 2/3 test 1/3 olacak şekilde
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)
print("x_train değerleri :", x_train)
print("x_test değerleri :", x_test)
print("y_train değerleri :", y_train)
print("y_test değerleri :", y_test)
print("--------------------------------")
# verilerin standartscaler küütphanesi ile ölçeklenmesi standart sapma kullandık
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# print(X_test)
# print(X_test)
print("--------------------------------")
# print(X_train)

# Lineer Regression için model oluşturma kısmı model inşasıdır
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)  # model inşa etme kısmı lineer model eğitme

y_pred = lr.predict(x_test)
print("y_pred:", y_pred)

# boy tahmini içinde kullanalım bunları
boy = s2.iloc[:, 3:4].values
# yeni eğitim kümesi için boy hariç yeni inşa yapıyoruz
sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]
veri = pd.concat([sol, sag], axis=1)
# print("veri :", veri) yeni veri toplulugu boy hariç gösterilmekte

a_train, a_test, b_train, b_test = train_test_split(veri, boy, train_size=0.67, random_state=0)

print("a_train değerleri :", a_train)
print("a_test değerleri :", a_test)
print("b_train değerleri :", b_train)
print("b_test değerleri :", b_test)

lr2 = LinearRegression()
lr.fit(a_train, b_train)  # model inşa etme kısmı lineer model eğitme
b_pred = lr.predict(a_test)  # a test kısmına bakarak b_pred tahmin ediyor x>y gibi
print("test_predict", b_pred)

# backward elimination
# model ve modeldeki değişkenlerin başarısı ile alakalı bir sistem kurmaktayız
#p-value ne kadar küçük ise o kadar iyi
import statsmodels.api as sm

# en başa 1 eklemek için kullanılır. bu da bizim ax+b deki b değeri içindir
X = np.append(arr=np.ones((22, 1)).astype(int), values=veri,
              axis=1)  # axis 1 demesi sütun olarak eklemesi demektir 0 satır
print("X Değeri:", X)

#ilk hali listenin X_list = veri.iloc[:, [0, 1, 2, 3, 4, 5]].values
X_list = veri.iloc[:, [0, 1, 2, 3, 4, 5]].values

# boy bağımlı değişken dizi bağımsız olan
regression_OLS = sm.OLS(endog=boy, exog=X_list).fit()
print(regression_OLS.summary())
# burada x5>0.5 olduğu için elenecek o yüzden üst kısımdan silecez

X_list = veri.iloc[:, [0, 1, 2, 3, 5]].values

# boy bağımlı değişken dizi bağımsız olan
regression_OLS = sm.OLS(endog=boy, exog=X_list).fit()
print(regression_OLS.summary())
# burada x5 istenilenin üstünde  olduğu için elenecek o yüzden üst kısımdan silecez


X_list = veri.iloc[:, [0, 1, 2, 3]].values

# boy bağımlı değişken dizi bağımsız olan
regression_OLS = sm.OLS(endog=boy, exog=X_list).fit()
print(regression_OLS.summary())
# burada x5>0.5 olduğu için elenecek o yüzden üst kısımdan silecez

# bu değerler silindikten sonra daha düzgün değerler kullanışlı değerler kalacağından daha sonra modelimizi tekrar oluşturup daha doğru sonuçlar alabiliriz.