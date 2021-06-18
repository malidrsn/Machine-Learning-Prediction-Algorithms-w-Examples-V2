from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

veriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values = np.nan,strategy ="mean")

yas = veriler.iloc[:,1:4].values
print(yas)

imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)