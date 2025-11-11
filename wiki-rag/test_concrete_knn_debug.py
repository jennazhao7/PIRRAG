#!/usr/bin/env python3
"""
Debug-focused version: Step-by-step Concrete ML KNN with detailed output

This version adds extensive debugging output to understand exactly what
Concrete ML is doing at each step.
"""

import numpy as np
from concrete.ml.sklearn import KNeighborsClassifier
import time

def debug_print(title, data, max_items=5):
    """Helper to print debug info"""
    print(f"\n{'='*70}")
    print(f"DEBUG: {title}")
    print(f"{'='*70}")
    if isinstance(data, np.ndarray):
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Min/Max: [{data.min():.3f}, {data.max():.3f}]")
        if len(data.shape) == 1:
            print(f"  First {min(max_items, len(data))} values: {data[:max_items]}")
        elif len(data.shape) == 2:
            print(f"  First row (first {min(max_items, data.shape[1])} dims): {data[0, :max_items]}")
    else:
        print(f"  Value: {data}")

# ============================================================================
# SETUP: Minimal wiki-rag scenario
# ============================================================================

print("\n" + "="*70)
print("CONCRETE ML KNN DEBUG TEST (Wiki-RAG Scenario)")
print("="*70)

# Create minimal test data
n_docs = 10  # Very small for easy debugging
dim = 10     # Small dimension for easy debugging

np.random.seed(42)

# Documents with clear structure
doc_embeddings = np.random.randn(n_docs, dim).astype(np.float32)
doc_embeddings[0:4] += np.array([2.0, 1.0] * 5)  # Group 1
doc_embeddings[4:7] += np.array([-1.0, 2.0] * 5)  # Group 2
doc_embeddings[7:10] += np.array([0.5, -1.5] * 5)  # Group 3

doc_labels = np.arange(n_docs, dtype=np.int64)

# Query similar to Group 1
query = np.array([2.1, 1.1] * 5).astype(np.float32).reshape(1, -1)

debug_print("Document Embeddings", doc_embeddings)
debug_print("Document Labels (Indices)", doc_labels)
debug_print("Query Embedding", query)

# ============================================================================
# TRAIN KNN
# ============================================================================

print("\n" + "="*70)
print("TRAINING CONCRETE ML KNN")
print("="*70)

knn = KNeighborsClassifier(n_neighbors=3, n_bits=2)
print(f"\n  Model parameters:")
print(f"    n_neighbors: {knn.n_neighbors}")
print(f"    n_bits: {knn.n_bits}")

print(f"\n  Fitting model...")
t0 = time.time()
knn.fit(doc_embeddings, doc_labels)
fit_time = time.time() - t0
print(f"  ✓ Fit completed in {fit_time:.3f}s")

# Check what was stored
print(f"\n  Model state:")
print(f"    Training data shape: {knn._fit_X.shape if hasattr(knn, '_fit_X') else 'N/A'}")
print(f"    Training labels: {knn._fit_y if hasattr(knn, '_fit_y') else 'N/A'}")

# ============================================================================
# COMPILE FHE
# ============================================================================

print("\n" + "="*70)
print("COMPILING FHE CIRCUIT")
print("="*70)

print(f"  Compiling with sample input...")
debug_print("Compilation Input Sample", doc_embeddings[:1])

t0 = time.time()
knn.compile(doc_embeddings)
compile_time = time.time() - t0
print(f"  ✓ Compilation completed in {compile_time:.3f}s")

# ============================================================================
# PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("PREDICTIONS")
print("="*70)

# Clear-text
print("\n[1] Clear-text prediction:")
debug_print("Input to predict()", query)
t0 = time.time()
pred_clear = knn.predict(query)
clear_time = time.time() - t0
print(f"  Result: {pred_clear[0]}")
print(f"  Time: {clear_time:.4f}s")

# FHE simulate
print("\n[2] FHE simulation:")
t0 = time.time()
pred_sim = knn.predict(query, fhe="simulate")
sim_time = time.time() - t0
print(f"  Result: {pred_sim[0]}")
print(f"  Time: {sim_time:.4f}s")

# FHE execute
print("\n[3] FHE execute:")
print("  (This is the actual encrypted computation)")
t0 = time.time()
pred_fhe = knn.predict(query, fhe="execute")
fhe_time = time.time() - t0
print(f"  Result: {pred_fhe[0]}")
print(f"  Time: {fhe_time:.4f}s")

# ============================================================================
# MANUAL VERIFICATION
# ============================================================================

print("\n" + "="*70)
print("MANUAL VERIFICATION (Clear-text)")
print("="*70)

# Compute distances manually
distances = np.linalg.norm(doc_embeddings - query, axis=1)
nearest_k = np.argsort(distances)[:knn.n_neighbors]

print(f"\n  Distances to all documents:")
for i, dist in enumerate(distances):
    marker = " ← TOP" if i in nearest_k else ""
    print(f"    Doc {i}: distance = {dist:.4f}{marker}")

print(f"\n  Top-{knn.n_neighbors} nearest neighbors:")
for i, idx in enumerate(nearest_k):
    print(f"    {i+1}. Document {idx} (distance: {distances[idx]:.4f})")

print(f"\n  Expected result: Document {nearest_k[0]} (closest)")
print(f"  KNN result: Document {pred_fhe[0]}")

if pred_fhe[0] == nearest_k[0]:
    print(f"  ✓ Match! KNN correctly identified the nearest neighbor")
elif pred_fhe[0] in nearest_k:
    print(f"  ⚠ KNN returned one of the top-{knn.n_neighbors}, but not the closest")
    print(f"     (This is expected - KNN returns most common class among k neighbors)")
else:
    print(f"  ⚠ KNN result differs (may be due to quantization/FHE precision)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DEBUG SUMMARY")
print("="*70)

print(f"""
What happened:
1. Created {n_docs} document embeddings ({dim}D each)
2. Created 1 query embedding ({dim}D)
3. Trained KNN with n_neighbors={knn.n_neighbors}, n_bits={knn.n_bits}
4. Compiled FHE circuit
5. Predicted document index: {pred_fhe[0]}

What this means for wiki-rag:
- This document index ({pred_fhe[0]}) would be used to retrieve the document
- wiki-rag would return: all_documents[{pred_fhe[0]}]
- This is the "most common class" among {knn.n_neighbors} nearest neighbors

Limitation:
- KNN returns ONE index (classification)
- Not top-{knn.n_neighbors} indices (retrieval)
- For top-k, wiki-rag uses a hybrid approach (FHE + clear-text)
""")

print("="*70)


