from concrete.ml.sklearn import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# --- Load small dataset ---
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --- Train Concrete-ML KNN ---
clf = KNeighborsClassifier(n_neighbors=3, n_bits=3)
t0 = time.time()
clf.fit(X_train, y_train)
clf.compile(X_train)
print("Fit+Compile time:", round(time.time() - t0, 2), "s")

# --- Evaluate ---
y_pred_clear = clf.predict(X_test)
y_pred_sim = clf.predict(X_test, fhe="simulate")

print("Accuracy (clear):", accuracy_score(y_test, y_pred_clear))
print("Accuracy (FHE simulate):", accuracy_score(y_test, y_pred_sim))

# --- Optional: true FHE execution on 3 samples (slow) ---
print("Running real FHE on 3 samples...")
y_exec = clf.predict(X_test[:3], fhe="execute")
print("FHE-exec preds:", y_exec.tolist())
