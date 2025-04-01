''''
Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
 ulazne veliˇ cine. Podaci su podijeljeni na skup za uˇ cenje i skup za testiranje modela.
 a) Prikažite podatke za uˇ cenje u x1−x2 ravnini matplotlib biblioteke pri ˇ cemu podatke obojite
 s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
 marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
 cmap kojima je mogu´ ce definirati boju svake klase.
 b) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.
 c) Prona¯ dite u atributima izgra¯ denog modela parametre modela. Prikažite granicu odluke
 nauˇcenog modela u ravnini x1 −x2 zajedno s podacima za uˇcenje. Napomena: granica
 odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.
 d) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije. Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Izraˇ cunate toˇ cnost,
 preciznost i odziv na skupu podataka za testiranje.
 e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznaˇ cite dobro klasificirane
 primjere dok pogrešno klasificirane primjere oznaˇ cite crnom bojom.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# a) dio
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Train Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test Data')

plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Vizualizacija podataka za treniranje i testiranje')
plt.legend()
plt.show()

# b) dio
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
y_test_p = LogRegression_model.predict(X_test)

# c) dio
intercept = LogRegression_model.intercept_
coefficients = LogRegression_model.coef_

print(f"Intercept: {intercept}")
print(f"Koeficijenti: {coefficients}")

x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
x2_decision_boundary = (-intercept - coefficients[0][0] * x1_range) / coefficients[0][1]

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Train Data')
plt.plot(x1_range, x2_decision_boundary, color='black', label='Decision Boundary')

plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Granica odluke modela logističke regresije u ravnini x1-x2')
plt.legend()
plt.show()

# d) dio
y_true = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 1, 0, 0])

print("Točnost: ", accuracy_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("Matrica zabune: ", cm)

disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
disp.plot()
plt.title('Matrica zabune')
plt.show()
print(classification_report(y_true, y_pred))

# e) dio
correctly_classified = y_test == y_test_p  
incorrectly_classified = ~correctly_classified  

plt.figure(figsize=(8, 6))
plt.scatter(X_test[correctly_classified, 0], X_test[correctly_classified, 1], c='green', marker='o', label='Correctly Classified')
plt.scatter(X_test[incorrectly_classified, 0], X_test[incorrectly_classified, 1], c='black', marker='x', label='Incorrectly Classified')

plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Dobro i pogrešno klasificirani testni primjeri u ravnini x1-x2')
plt.legend()
plt.show()