'''
Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
 te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
 ovih hiperparametara utjeˇce na granicu odluke te pogrešku na skupu podataka za testiranje?
 Mijenjajte tip kernela koji se koristi. Što primje´ cujete?
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

C=[0.1, 1, 10, 100]
gamma=[0.1, 1, 10, 100]

for i in range(0, 4):
    for j in range(0, 4):
        SVM_model = svm.SVC(kernel='rbf', gamma = gamma[i], C = C[j])
        SVM_model.fit(X_train_n, y_train)

        y_train_p = SVM_model.predict(X_train_n)
        y_test_p = SVM_model.predict(X_test_n)

        print("SVM (RBF Kernel): ")
        print("Tocnost train; gamma = {gamma[i]}, C = {C[j]}: " + "{:0.4f}".format(accuracy_score(y_train, y_train_p)))
        print("Tocnost test; gamma = {gamma[i]}, C = {C[j]}: " + "{:0.4f}".format(accuracy_score(y_test, y_test_p)))

        plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
        plt.xlabel('x_1 (Age)')
        plt.ylabel('x_2 (Estimated Salary)')
        plt.legend(loc='upper left')
        plt.title(f"SVM Model - RBF Kernel: gamma = {gamma[i]}, C = {C[j]}, Train Accuracy: {accuracy_score(y_train, y_train_p):.4f}")
        plt.tight_layout()
        plt.show()
