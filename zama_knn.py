#!/usr/bin/env python3
import argparse, sys, time, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ✅ correct import path
from concrete.ml.sklearn import KNeighborsClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-bits", type=int, default=3)
    p.add_argument("--pca-components", type=int, default=10,
                   help="Number of PCA components for dimensionality reduction (required for FHE).")
    p.add_argument("--fhe-samples", type=int, default=8,
                   help="Run real FHE on this many test samples (slow).")
    args = p.parse_args()

    print("=== Loading dataset (Breast Cancer) ===")
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)

    # Apply PCA for dimensionality reduction (required for FHE compilation)
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    scaler = StandardScaler()
    
    print(f"Original features: {X.shape[1]} -> PCA components: {args.pca_components}")
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    
    Xtr, Xte, ytr, yte = train_test_split(
        X_pca, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Train size: {len(Xtr)} | Test size: {len(Xte)}")
    print(f"k={args.k}, n_bits={args.n_bits}, PCA components={args.pca_components}")

    print("\n=== Fitting Concrete-ML KNN ===")
    t0 = time.time()
    clf = KNeighborsClassifier(n_neighbors=args.k, n_bits=args.n_bits)
    clf.fit(Xtr, ytr)
    fit_sec = time.time() - t0
    print(f"Fit time: {fit_sec:.3f}s")

    print("\n=== Compiling to FHE ===")
    t0 = time.time()
    clf.compile(Xtr)  # builds & caches the circuit
    comp_sec = time.time() - t0
    print(f"Compile time: {comp_sec:.3f}s")

    print("\n=== Predict (clear) ===")
    t0 = time.time()
    y_clear = clf.predict(Xte)
    clear_t = time.time() - t0
    acc_clear = accuracy_score(yte, y_clear)
    print(f"acc={acc_clear:.4f}, time={clear_t:.3f}s (full test)")

    print("\n=== Predict (FHE simulate) ===")
    t0 = time.time()
    y_sim = clf.predict(Xte, fhe="simulate")
    sim_t = time.time() - t0
    acc_sim = accuracy_score(yte, y_sim)
    print(f"acc={acc_sim:.4f}, time={sim_t:.3f}s (full test)")

    n_exec = min(args.fhe_samples, len(Xte))
    print(f"\n=== Predict (FHE execute) on {n_exec} samples ===")
    X_exec, y_exec = Xte[:n_exec], yte[:n_exec]
    t0 = time.time()
    y_exec_pred = clf.predict(X_exec, fhe="execute")  # includes keygen on first call
    exec_t = time.time() - t0
    acc_exec = accuracy_score(y_exec, y_exec_pred)
    print(f"acc={acc_exec:.4f}, time={exec_t:.3f}s (includes keygen)")

    agree = (y_sim == y_clear).mean()
    print("\n=== Summary ===")
    print(f"Fit: {fit_sec:.3f}s | Compile: {comp_sec:.3f}s")
    print(f"Clear: acc={acc_clear:.4f}, time={clear_t:.3f}s")
    print(f"FHE-sim: acc={acc_sim:.4f}, time={sim_t:.3f}s")
    print(f"FHE-exec (n={n_exec}): acc={acc_exec:.4f}, time={exec_t:.3f}s")
    print(f"Clear vs Sim agreement: {agree*100:.2f}%")
    print("\nTips: Use --pca-components to reduce dimensions for FHE; lower --n-bits or --k to speed up FHE.")

if __name__ == "__main__":
    main()
