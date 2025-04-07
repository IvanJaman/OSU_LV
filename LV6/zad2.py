'''
 Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
 algoritma KNN za podatke iz Zadatka 1
'''

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("LV6\Social_Network_Ads.csv")
print(data.info())

X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

k_values = list(range(1, 21))  
mean_scores = []

for k in k_values:
    KNN_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(KNN_model, X_train_n, y_train, cv=5, scoring='accuracy')
    mean_scores.append(scores.mean())

optimal_K = k_values[np.argmax(mean_scores)]
print(f"Optimalna vrijednost K: {optimal_K}")
print(f"Najbolja srednja točnost (K={optimal_K}): {max(mean_scores):.4f}")

plt.plot(k_values, mean_scores, marker='o', linestyle='-', color='b')
plt.title('Točnost u funkciji vrijednosti K')
plt.xlabel('K - broj susjeda')
plt.ylabel('Srednja točnost (5-fold CV)')
plt.xticks(k_values)
plt.grid(True)
plt.show()

best_KNN_model = KNeighborsClassifier(n_neighbors=optimal_K)
best_KNN_model.fit(X_train_n, y_train)

y_test_p_KNN_best = best_KNN_model.predict(X_test_n)
print(f"Tocnost na testnom skupu za optimalni K={optimal_K}: {accuracy_score(y_test, y_test_p_KNN_best):.4f}")
