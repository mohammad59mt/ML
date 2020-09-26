from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale

import pandas as pd
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X = scale(X)

def func1():
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    print("Validation: {:.3f}".format(knn.score(X_val, y_val)))
    print("Test: {:.3f}".format(knn.score(X_test, y_test)))
    val = []
    test = []
    for i in range(1000):
        rng = np.random.RandomState(i)
        noise = rng.normal(scale=.1, size=X_train.shape)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train + noise, y_train)
        val.append(knn.score(X_val, y_val))
        test.append(knn.score(X_test, y_test))

    print("Validation: {:.3f}".format(np.max(val)))
    print("Test: {:.3f}".format(test[np.argmax(val)]))

def func3_parameterSelectionByValidationData():
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)
    val_scores = []
    neighbors = np.arange(1, 15, 2)
    for i in neighbors:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        val_scores.append(knn.score(X_val, y_val))
    print("best validation score: {:.3f}".format(np.max(val_scores)))
    best_n_neighbors = neighbors[np.argmax(val_scores)]
    print("best n_neighbors:", best_n_neighbors)
    knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    knn.fit(X_trainval, y_trainval)
    print("test-set score: {:.3f}".format(knn.score(X_test, y_test)))

def func4_crossValidation():
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    cross_val_scores = []
    neighbors = np.arange(1, 15, 2)
    for i in neighbors:
        knn = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn, X_train, y_train, cv=10)
        cross_val_scores.append(np.mean(scores))
    print("best cross-validation score: {:.3f}".format(np.max(cross_val_scores)))
    best_n_neighbors = neighbors[np.argmax(cross_val_scores)]
    print("best n_neighbors:", best_n_neighbors)
    knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    knn.fit(X_train, y_train)
    print("test-set score: {:.3f}".format(knn.score(X_test, y_test)))

def func5_parameterSettingWithGridSearch():
    from sklearn.model_selection import GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    param_grid = {'n_neighbors':  np.arange(1, 15, 2)}
    ''' Use  a more complicated method for parameter setting rather than fixed cros_validation'''
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,
                        cv=10, return_train_score=True)
    grid.fit(X_train, y_train)
    print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("best parameters: {}".format(grid.best_params_))
    print("test-set score: {:.3f}".format(grid.score(X_test, y_test)))

class Time_Series:
    def __init__(self):
        self.data = None
    def read_data(self, file_addrs):
        self.data = pd.read_csv(file_addrs, header=0)



# func3_parameterSelectionByValidationData()
# func4_crossValidation()
# func5_parameterSettingWithGridSearch()
