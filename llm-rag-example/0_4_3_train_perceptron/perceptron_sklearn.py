# perceptron_sklearn.py

from sklearn.linear_model import Perceptron
import numpy as np

X = np.array([
    [1.0, 1.0],
    [2.0, 1.5],
    [1.5, 2.0],
    [2.0, 2.5],
    [-1.0, -1.0],
    [-2.0, -1.5],
    [-1.5, -2.0],
    [-2.0, -2.5],
])

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

clf = Perceptron()
clf.fit(X, y)

print(clf.coef_, clf.intercept_)
print(clf.predict(X))
