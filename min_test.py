from concrete.ml.sklearn import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=3, n_bits=8)  # choose k and quantization
knn.fit(X, y)

# Compile once on representative inputs (calibration):
knn.compile(X[:32])  # any small sample with same shape/range

# Run fully under FHE (encrypt → execute → decrypt in one call):
y_pred = knn.predict(X[100:105], fhe="execute")          # labels
p_pred = knn.predict_proba(X[100:105], fhe="execute")    # class probs
