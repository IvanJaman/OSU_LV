''''
Skripta zadatak_1.py uˇcitava podatkovni skup iz data_C02_emission.csv .
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju os-
talih numeriˇckih ulaznih veliˇcina. Detalje oko ovog podatkovnog skupa mogu se prona´ci u 3.
laboratorijskoj vježbi.
a) Odaberite željene numeriˇcke veliˇcine specificiranjem liste s nazivima stupaca. Podijelite
podatke na skup za uˇcenje i skup za testiranje u omjeru 80%-20%.
b) Pomo´cu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
o jednoj numeriˇckoj veliˇcini. Pri tome podatke koji pripadaju skupu za uˇcenje oznaˇcite
plavom bojom, a podatke koji pripadaju skupu za testiranje oznaˇcite crvenom bojom.
c) Izvršite standardizaciju ulaznih veliˇcina skupa za uˇcenje. Prikažite histogram vrijednosti
jedne ulazne veliˇcine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
transformirajte ulazne veliˇcine skupa podataka za testiranje.
d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
e) Izvršite procjenu izlazne veliˇcine na temelju ulaznih veliˇcina skupa za testiranje. Prikažite
pomo´cu dijagrama raspršenja odnos izme ¯du stvarnih vrijednosti izlazne veliˇcine i procjene
dobivene modelom.
f) Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika na
skupu podataka za testiranje.
g) Što se doga ¯da s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj
ulaznih veliˇcina?
'''
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import sklearn.linear_model as lm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# a) dio
numerical_columns = ["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)",
                     "Fuel Consumption Comb (mpg)"]
df = pd.read_csv('data_C02_emission.csv')

X = df[numerical_columns]
y = df['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# b) dio
plt.scatter(X_train['Engine Size (L)'], y_train,
            color='blue', label='Training Set')
plt.scatter(X_test['Engine Size (L)'], y_test, color='red', label='Test Set')

plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

# c) dio
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

plt.hist(X_train['Engine Size (L)'], bins=30, alpha=0.7, label='X_train')
plt.hist(X_train_n[:, 0], bins=30, alpha=0.7, label='X_train_n')

plt.xlabel('Engine Size (L)')
plt.ylabel('Frequency')
plt.show()

# d) dio
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print("Intercept (b):", linearModel.intercept_)
print("Coefficients (weights):", linearModel.coef_)

sns.pairplot(df[['Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)',
                 'Fuel Consumption Comb (mpg)', 'CO2 Emissions (g/km)']])
plt.show()

# e) dio
y_test_p = linearModel.predict(X_test_n)

plt.scatter(y_test, y_test_p)
plt.xlabel('Actual results')
plt.ylabel('Predicted results')
plt.show()

# f) dio
MAE = mean_absolute_error(y_test, y_test_p)
print(MAE)

# g) dio
print("Mjenjanjem test_size MAE ostaje približno jednaka.")