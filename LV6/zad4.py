'''
Pomo´ cu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
 algoritma SVM za problem iz Zadatka 1
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

data = pd.read_csv("LV6\Social_Network_Ads.csv")
print(data.info())

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

SVM_model = svm.SVC(kernel='rbf', gamma = 1, C=0.1)
SVM_model.fit(X_train_n, y_train)

y_train_p = SVM_model.predict(X_train_n)
y_test_p = SVM_model.predict(X_test_n)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train_n, y_train)

print(f"Optimalni hiperparametri (C, gamma): {grid_search.best_params_}")
print(f"Najbolja točnost na skupu za ucenje: {grid_search.best_score_:.3f}")

optimal_SVM_model = grid_search.best_estimator_
y_train_optimal = optimal_SVM_model.predict(X_train_n)
y_test_optimal = optimal_SVM_model.predict(X_test_n)

print("Optimalni SVM model (RBF Kernel): ")
print("Tocnost train: " + "{:0.4f}".format(accuracy_score(y_train, y_train_optimal)))
print("Tocnost test: " + "{:0.4f}".format(accuracy_score(y_test, y_test_optimal)))
