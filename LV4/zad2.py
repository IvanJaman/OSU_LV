''''
 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoriˇ cku
 varijable „Fuel Type“ kao ulaznu veliˇcinu. Pri tome koristite 1-od-K kodiranje kategoriˇckih
 veliˇ cina. Radi jednostavnosti nemojte skalirati ulazne veliˇ cine. Komentirajte dobivene rezultate.
 Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
 vozila radi?
'''

from sklearn.metrics import mean_absolute_error
import seaborn as sns
import sklearn.linear_model as lm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data_C02_emission.csv')

cols = ["Engine Size (L)", "Cylinders", "Fuel Type", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)",
                     "Fuel Consumption Comb (mpg)"]

X = df[cols]
y = df['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_encoded = ohe.fit_transform(X_train[['Fuel Type']])
X_test_encoded = ohe.transform(X_test[['Fuel Type']])

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out(['Fuel Type']), index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=ohe.get_feature_names_out(['Fuel Type']), index=X_test.index)

X_train = X_train.drop(columns=['Fuel Type']).join(X_train_encoded_df)
X_test = X_test.drop(columns=['Fuel Type']).join(X_test_encoded_df)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

print("Intercept (b):", linearModel.intercept_)
print("Coefficients (weights):", linearModel.coef_)

plt.scatter(df['Fuel Type'], df['CO2 Emissions (g/km)'], color="red")
plt.xlabel('Fuel Type')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

y_test_p = linearModel.predict(X_test)

plt.scatter(y_test, y_test_p)
plt.xlabel('Actual results')
plt.ylabel('Predicted results')
plt.show()

MAE = mean_absolute_error(y_test, y_test_p)
max_error = np.max(np.abs(y_test - y_test_p))
max_error_index = np.argmax(np.abs(y_test - y_test_p))

print(f"Maksimalna pogreška: {max_error} g/km")
print(f"Medijalna pogreška: {MAE} g/km")

max_error_index = np.argmax(np.abs(y_test - y_test_p))
vehicle_info = df.iloc[X_test.index[max_error_index]][['Make', 'Model']]
print(f"Vozilo s najvećom pogreškom: {vehicle_info['Make']} {vehicle_info['Model']}")

print("KOMENTAR:")
print("Prema rezultatima i dijagramu raspršenja, vidljivo je da regularni benzin (Z) ima najveći interval CO2 emisija, dok dizel (D) ima najmanji.")