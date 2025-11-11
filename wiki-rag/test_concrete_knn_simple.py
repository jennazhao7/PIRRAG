#!/usr/bin/env python3
"""
Ultra-simple test for Concrete ML KNN - minimal parameters for debugging

This version uses very small parameters to ensure FHE compilation succeeds.
Use this if test_concrete_knn_minimal.py fails with NoParametersFound error.
"""

import time
import numpy as np
from concrete.ml.sklearn import KNeighborsClassifier

# ---------- helpers ----------

def try_compile(knn: KNeighborsClassifier, X_calib: np.ndarray, p_errors=(0.01, 0.02, 0.05)):
    """
    Try compiling with several p_error values (passed as kwargs).
    Returns (True, used_p_error, compile_time) on success; (False, last_exception, None) on failure.
    """
    last_err = None
    for p_err in p_errors:
        try:
            t0 = time.time()
            # IMPORTANT: p_error must be a kwarg here (NOT in fhe.Configuration)
            knn.compile(X_calib, p_error=p_err)
            return True, p_err, time.time() - t0
        except Exception as e:
            last_err = e
            print(f"  ✗ compile failed with p_error={p_err}: {e}")
    return False, last_err, None


def build_and_fit_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int, n_bits: int) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_bits=n_bits)
    knn.fit(X, y)
    return knn


def main():
    print("=" * 70)
    print("CONCRETE-ML KNN — robust compile")
    print("=" * 70)

    # -------- base tiny dataset --------
    base = {
        "n_docs": 10,
        "dim": 10,
        "n_neighbors": 3,
        "n_bits": 2,
    }

    rng = np.random.default_rng(42)
    X_full = (rng.normal(size=(base["n_docs"], base["dim"])).astype(np.float32) * 0.3)
    y_full = np.arange(base["n_docs"], dtype=np.int64)

    # Query close to first document
    query_full = (X_full[0] + rng.normal(size=base["dim"]).astype(np.float32) * 0.1)[None, :]

    print("Initial parameters:")
    for k, v in base.items():
        print(f"  {k}: {v}")
    print(f"\nData ranges: X in [{X_full.min():.3f}, {X_full.max():.3f}] | query in [{query_full.min():.3f}, {query_full.max():.3f}]")

    # -------- fallback configurations (we'll try in order) --------
    # Each config optionally reduces dim, neighbors, or n_bits to widen the feasible parameter space.
    configs = [
        {"dim": base["dim"], "n_neighbors": 3, "n_bits": 2},
        {"dim": base["dim"], "n_neighbors": 2, "n_bits": 2},
        {"dim": base["dim"], "n_neighbors": 2, "n_bits": 1},
        {"dim": 8,            "n_neighbors": 2, "n_bits": 2},
        {"dim": 6,            "n_neighbors": 2, "n_bits": 1},
    ]

    success = False
    used = None
    last_err = None

    for cfg in configs:
        dim = cfg["dim"]
        nnz = cfg["n_neighbors"]
        nbits = cfg["n_bits"]

        # Slice columns if we reduce dimension
        X = X_full[:, :dim]
        query = query_full[:, :dim]

        print("\n--- Trying configuration:", cfg, "---")
        print(f"Shapes: X={X.shape}, query={query.shape}")

        # Fit KNN for this config
        knn = build_and_fit_knn(X, y_full, n_neighbors=nnz, n_bits=nbits)

        # Calibration set: small batch from the training distribution (up to 8 rows)
        X_calib = X[: min(8, X.shape[0])]

        print("Compiling FHE circuit (p_error via kwargs)…")
        ok, info, dt = try_compile(knn, X_calib, p_errors=(0.01, 0.02, 0.05))
        if ok:
            print(f"✓ Compilation successful with config {cfg} in {dt:.2f}s (p_error={info})")
            success, used = True, (cfg, info, dt, knn, X, query)
            break
        else:
            last_err = info
            print(f"Compilation failed for config {cfg}. Trying next fallback…")

    if not success:
        print("\n❌ Compilation failed after all fallbacks.")
        print("Consider reducing dimension further, lowering n_bits to 1, or ensuring your Concrete/Concrete-ML versions are recent.")
        raise last_err

    # Unpack the successful artifacts
    cfg, p_err, compile_time, knn, X, query = used

    # -------- predictions --------
    print("\nTesting predictions (clear/simulate/execute)…")
    pred_clear = knn.predict(query)
    print(f"  Clear-text:   {int(pred_clear[0])}")

    pred_sim = knn.predict(query, fhe="simulate")
    print(f"  FHE simulate: {int(pred_sim[0])}")

    pred_fhe = knn.predict(query, fhe="execute")
    print(f"  FHE execute:  {int(pred_fhe[0])}")

    # -------- manual verification --------
    distances = np.linalg.norm(X - query, axis=1)
    nearest = np.argsort(distances)[: cfg["n_neighbors"]]
    print("\nManual verification:")
    print(f"  Top-{cfg['n_neighbors']} nearest indices: {nearest.tolist()}")
    print(f"  KNN result (label): {int(pred_fhe[0])}")
    print(f"  Match: {'✓' if int(pred_fhe[0]) in nearest else '✗'}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    # Optional: quick version print (helps debugging env mismatches)
    try:
        import concrete
        import concrete.ml as cml
        print(f"[versions] concrete={concrete.__version__} concrete-ml={cml.__version__}")
    except Exception:
        pass
    main()