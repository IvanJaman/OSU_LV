'''
Skripta zadatak_1.py uˇcitava Social_Network_Ads.csv skup podataka [2].

 Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
 Podaci o korisnicima su spol, dob i procijenjena pla´ca. Razmatra se binarni klasifikacijski
 problem gdje su dob i procijenjena pla´ca ulazne veliˇcine, dok je kupovina (0 ili 1) izlazna
 veliˇcina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija
 plot_decision_region [1]. Podaci su podijeljeni na skup za uˇ cenje i skup za testiranje modela
 u omjeru 80%-20% te su standardizirani. Izgra¯ den je model logistiˇ cke regresije te je izraˇ cunata
 njegova toˇ cnost na skupu podataka za uˇ cenje i skupu podataka za testiranje. 

 Potrebno je:
    1. Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Izraˇcunajte toˇcnost
 klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje. Usporedite
 dobivene rezultate s rezultatima logistiˇ cke regresije. Što primje´ cujete vezano uz dobivenu
 granicu odluke KNN modela?

    2. Kako izgleda granica odluke kada je K = 1 i kada je K = 100?
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.1):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("LV6\Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.4f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.4f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.4f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# 1. zadatak
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("KNN Model: ")
print("Tocnost train: " + "{:0.4f}".format(accuracy_score(y_train, y_train_p_KNN)))
print("Tocnost test: " + "{:0.4f}".format(accuracy_score(y_test, y_test_p_KNN)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1 (Age)')
plt.ylabel('x_2 (Estimated Salary)')
plt.legend(loc='upper left')
plt.title("KNN Model - Tocnost train: " + "{:0.4f}".format(accuracy_score(y_train, y_train_p_KNN)))
plt.tight_layout()
plt.show()

print("\nUsporedba Logisticke regresije i KNN-a:")
print(f"Tocnost Logisticke regresije (Train): {accuracy_score(y_train, y_train_p):.4f}")
print(f"Tocnost Logisticke regresije (Test): {accuracy_score(y_test, y_test_p):.4f}")
print(f"KNN Tocnost (Train): {accuracy_score(y_train, y_train_p_KNN):.4f}")
print(f"KNN Tocnost (Test): {accuracy_score(y_test, y_test_p_KNN):.4f}")

# 2. zadatak

# tocnost za k=1
KNN_model = KNeighborsClassifier(n_neighbors = 1)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("KNN Model: ")
print("Tocnost train: " + "{:0.4f}".format(accuracy_score(y_train, y_train_p_KNN)))
print("Tocnost test: " + "{:0.4f}".format(accuracy_score(y_test, y_test_p_KNN)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1 (Age)')
plt.ylabel('x_2 (Estimated Salary)')
plt.legend(loc='upper left')
plt.title("KNN Model - Tocnost train: " + "{:0.4f} za K=1".format(accuracy_score(y_train, y_train_p_KNN)))
plt.tight_layout()
plt.show()

print("Za K=1 događa se overfitting, svaki je podatak ispravno klasificiran, no granica odluke je 'iscjepkana na regije'")

# tocnost za k=100
KNN_model = KNeighborsClassifier(n_neighbors = 100)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("KNN Model: ")
print("Tocnost train: " + "{:0.4f}".format(accuracy_score(y_train, y_train_p_KNN)))
print("Tocnost test: " + "{:0.4f}".format(accuracy_score(y_test, y_test_p_KNN)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1 (Age)')
plt.ylabel('x_2 (Estimated Salary)')
plt.legend(loc='upper left')
plt.title("KNN Model - Tocnost train: " + "{:0.4f} za k=100".format(accuracy_score(y_train, y_train_p_KNN)))
plt.tight_layout()
plt.show()

print("Za K=100 događa se underfitting, klasifikacija je pogrešna za velik broj podataka. Granica odluke je blizu linearne.")
