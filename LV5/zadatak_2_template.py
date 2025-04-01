''''
Skripta zadatak_2.py uˇcitava podatkovni skup Palmer Penguins [1]. Ovaj
 podatkovni skup sadrži mjerenja provedena na tri razliˇcite vrste pingvina (’Adelie’, ’Chins
trap’, ’Gentoo’) na tri razliˇcita otoka u podruˇcju Palmer Station, Antarktika. Vrsta pingvina
 odabrana je kao izlazna veliˇcina i pri tome su klase oznaˇcene s cjelobrojnim vrijednostima
 0, 1 i 2. Ulazne veliˇcine su duljina kljuna (’bill_length_mm’) i duljina peraje u mm (’flip
per_length_mm’). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
 funkcija plot_decision_region.
 a) Pomo´cu stupˇcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu
 pingvina) u skupu podataka za uˇcenje i skupu podataka za testiranje. Koristite numpy
 funkciju unique.
 b) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.
 c) Prona¯ dite u atributima izgra¯ denog modela parametre modela. Koja je razlika u odnosu na
 binarni klasifikacijski problem iz prvog zadatka?
 d) Pozovite funkciju plot_decision_region pri ˇcemu joj predajte podatke za uˇcenje i
 izgra ¯ deni model logistiˇ cke regresije. Kako komentirate dobivene rezultate?
 e) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije. Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Izraˇ cunajte toˇ cnost.
 Pomo´cu classification_report funkcije izraˇcunajte vrijednost ˇcetiri glavne metrike na skupu podataka za testiranje.
 f) Dodajte u model još ulaznih veliˇcina. Što se doga¯ da s rezultatima klasifikacije na skupu
 podataka za testiranje?
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm', 'bill_depth_mm',
                    'flipper_length_mm', 'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# a) dio
train_class_counts = np.unique(y_train, return_counts=True)
test_class_counts = np.unique(y_test, return_counts=True)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(train_class_counts[0], train_class_counts[1], color=['blue', 'red', 'green'])
plt.title('Broj primjera po klasama - Skup za treniranje')
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.xticks(train_class_counts[0], ['Adelie', 'Chinstrap', 'Gentoo'])

plt.subplot(1, 2, 2)
plt.bar(test_class_counts[0], test_class_counts[1], color=['blue', 'red', 'green'])
plt.title('Broj primjera po klasama - Skup za testiranje')
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.xticks(test_class_counts[0], ['Adelie', 'Chinstrap', 'Gentoo'])

plt.tight_layout()
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

print("Kod binarne klasifikacije imamo samo jedan intercept i dva koeficijenta, dok kod višeklasne imamo tri intercepta i šest koeficijentanta (po jedan int. i dva koef. za svaku klasu pingvina). Koeficijenti predstavljaju duljinu kljuna i peraja u mm.")

# d) dio
# plot_decision_regions(X_train, y_train.ravel(), LogRegression_model)
# plt.show()

# e) dio
cm = confusion_matrix(y_test, y_test_p)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Adelie', 'Chinstrap', 'Gentoo'], yticklabels=['Adelie', 'Chinstrap', 'Gentoo'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

accuracy = accuracy_score(y_test, y_test_p)
print(f"Točnost: {accuracy:.4f}")

report = classification_report(y_test, y_test_p, target_names=['Adelie', 'Chinstrap', 'Gentoo'])
print("Classification Report:")
print(report)

# f) dio
''''
Za dvije ulazne veličine ('bill_length_mm', 'flipper_length_mm'):
Classification Report
              precision    recall  f1-score   support

      Adelie       0.96      0.89      0.92        27
   Chinstrap       0.94      0.88      0.91        17
      Gentoo       0.89      1.00      0.94        25

    accuracy                           0.93        69
   macro avg       0.93      0.92      0.93        69
weighted avg       0.93      0.93      0.93        69'

Za četiri ulazne veličine ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'):
Classification Report:
              precision    recall  f1-score   support

      Adelie       1.00      0.93      0.96        27
   Chinstrap       0.89      1.00      0.94        17
      Gentoo       1.00      1.00      1.00        25

    accuracy                           0.97        69
   macro avg       0.96      0.98      0.97        69
weighted avg       0.97      0.97      0.97        69
'''

print("Uočavamo da je model točniji s 4 ulazna podatka.")