#!/usr/bin/env python3
"""
Minimal test for Concrete ML KNN in wiki-rag context

This script isolates the Concrete ML KNN part and mimics what it would see
in the actual wiki-rag pipeline. Everything is manual and minimal for easy debugging.

What this simulates:
- Document embeddings (50D after PCA, mimicking wiki-rag)
- Query embedding (50D after PCA, mimicking wiki-rag)
- Concrete ML KNN classification
- Document retrieval by index
"""

import numpy as np
from concrete.ml.sklearn import KNeighborsClassifier
import time

# ============================================================================
# SIMULATE WIKI-RAG SCENARIO
# ============================================================================

print("=" * 70)
print("MINIMAL CONCRETE ML KNN TEST (Wiki-RAG Scenario)")
print("=" * 70)

# ----------------------------------------------------------------------------
# Step 1: Simulate document embeddings (what wiki-rag would have after PCA)
# ----------------------------------------------------------------------------
print("\n[Step 1] Creating mock document embeddings...")
print("-" * 70)

# Simulate documents (instead of 100k for easy debugging)
# IMPORTANT: Concrete ML KNN has strict limits on training set size (~10-15 samples)
# Testing with 18 documents and 20 dimensions to find the actual limits
# Each document has 20D embedding (reduced from 50D for FHE compatibility)
n_documents = 18  # Testing if 18 documents works (pushing the limit)
embedding_dim = 20  # Reduced from 50D - testing with lower dimensions

# Create some realistic-looking embeddings (clustered around topics)
# Note: Keep values in a reasonable range for FHE (not too large)
np.random.seed(42)
document_embeddings = np.random.randn(n_documents, embedding_dim).astype(np.float32) * 0.5

# Add some structure: group documents into topics
# Topic 1: Documents 0-5 (similar to each other)
topic1_offset = np.array([1.0, 0.75, -0.5] * 6 + [1.0, 0.75])[:embedding_dim]
document_embeddings[0:6] += topic1_offset

# Topic 2: Documents 6-11 (similar to each other)
topic2_offset = np.array([-0.75, 1.0, 0.5] * 6 + [-0.75, 1.0])[:embedding_dim]
document_embeddings[6:12] += topic2_offset

# Topic 3: Documents 12-17 (similar to each other)
topic3_offset = np.array([0.25, -1.0, 0.75] * 6 + [0.25, -1.0])[:embedding_dim]
document_embeddings[12:18] += topic3_offset

# Normalize to keep values in reasonable range for FHE
document_embeddings = document_embeddings / np.std(document_embeddings) * 0.5

print(f"✓ Created {n_documents} document embeddings")
print(f"  Shape: {document_embeddings.shape}")
print(f"  Range: [{document_embeddings.min():.2f}, {document_embeddings.max():.2f}]")

# ----------------------------------------------------------------------------
# Step 2: Create document labels (document indices)
# ----------------------------------------------------------------------------
print("\n[Step 2] Creating document labels (indices)...")
print("-" * 70)

# In wiki-rag, we use document indices as "class labels"
# This allows KNN to predict which document index is most similar
document_indices = np.arange(n_documents, dtype=np.int64)

print(f"✓ Document labels: {document_indices}")
print(f"  These represent document indices: 0, 1, 2, ..., {n_documents-1}")

# ----------------------------------------------------------------------------
# Step 3: Simulate query embedding (what wiki-rag would have after PCA)
# ----------------------------------------------------------------------------
print("\n[Step 3] Creating mock query embedding...")
print("-" * 70)

# Simulate a query that's similar to Topic 1 documents
# In real wiki-rag: query → embed → PCA → 20D vector (reduced)
# Use similar values to Topic 1 but in the same normalized range
query_base = np.array([1.05, 0.8, -0.45] * 6 + [1.05, 0.8])[:embedding_dim]
query_embedding = (query_base + np.random.randn(embedding_dim).astype(np.float32) * 0.1)
query_embedding = query_embedding.astype(np.float32).reshape(1, -1)  # Shape: (1, 20)

print(f"✓ Query embedding created")
print(f"  Shape: {query_embedding.shape}")
print(f"  Range: [{query_embedding.min():.2f}, {query_embedding.max():.2f}]")
print(f"  (Similar to Topic 1 documents, should match documents 0-5)")
print(f"\n  NOTE: Testing with {n_documents} documents and {embedding_dim}D (pushing limits)")

# ----------------------------------------------------------------------------
# Step 4: Train Concrete ML KNN with progressive fallback
# ----------------------------------------------------------------------------
print("\n[Step 4] Training Concrete ML KNN...")
print("-" * 70)

# Progressive fallback configurations - try larger first, fallback to smaller
# Each config: (dim, n_neighbors, n_bits)
# NOTE: With 18 documents and 20D, we start with 20D and reduce if needed
configs = [
    (embedding_dim, 5, 2),   # Original: 20D, 5 neighbors, 2 bits
    (embedding_dim, 3, 2),   # Reduce neighbors: 20D, 3 neighbors, 2 bits
    (embedding_dim, 2, 2),   # Further reduce: 20D, 2 neighbors, 2 bits
    (15, 3, 2),              # Reduce dim: 15D, 3 neighbors, 2 bits
    (15, 2, 2),              # Further reduce: 15D, 2 neighbors, 2 bits
    (10, 3, 2),              # Smaller: 10D, 3 neighbors, 2 bits
    (10, 2, 2),              # Even smaller: 10D, 2 neighbors, 2 bits
    (8, 2, 2),               # Very small: 8D, 2 neighbors, 2 bits
    (6, 2, 2),               # Minimal: 6D, 2 neighbors, 2 bits
    (6, 2, 1),               # Minimal with 1 bit: 6D, 2 neighbors, 1 bit
]

fhe_knn = None
used_config = None
compile_samples = None
used_p_error = None

for dim, n_neighbors, n_bits in configs:
    print(f"\n  Trying configuration: dim={dim}, n_neighbors={n_neighbors}, n_bits={n_bits}")
    
    # Slice embeddings if reducing dimension
    if dim < embedding_dim:
        embeddings_subset = document_embeddings[:, :dim].astype(np.float32)
        query_subset = query_embedding[:, :dim].astype(np.float32)
    else:
        embeddings_subset = document_embeddings
        query_subset = query_embedding
    
    print(f"    Embeddings shape: {embeddings_subset.shape}")
    
    # Train KNN
    start_time = time.time()
    fhe_knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_bits=n_bits)
    fhe_knn.fit(embeddings_subset, document_indices)
    fit_time = time.time() - start_time
    print(f"    ✓ Training completed in {fit_time:.3f}s")
    
    # Try to compile
    calibration_size = min(8, len(embeddings_subset))
    compile_samples = embeddings_subset[:calibration_size].astype(np.float32)
    
    print(f"    Compiling FHE circuit...")
    p_errors = [0.01, 0.02, 0.05]
    compile_success = False
    
    for p_err in p_errors:
        try:
            fhe_knn.compile(compile_samples, p_error=p_err)
            used_p_error = p_err
            used_config = (dim, n_neighbors, n_bits)
            compile_success = True
            print(f"    ✓ Compilation successful with p_error={p_err}")
            break
        except Exception as e:
            if p_err == p_errors[-1]:
                print(f"    ✗ Compilation failed with all p_error values")
                break
            continue
    
    if compile_success:
        print(f"\n✓ Successfully compiled with: dim={dim}, n_neighbors={n_neighbors}, n_bits={n_bits}, p_error={used_p_error}")
        # Update query embedding to match the dimension we used
        query_embedding = query_subset
        break
    else:
        print(f"  ✗ Configuration failed, trying next...")

if fhe_knn is None or not compile_success:
    raise RuntimeError(
        "FHE compilation failed with all configurations. "
        "The dataset/parameters may be too large for FHE. "
        "Try reducing the number of documents or dimensions further."
    )

print(f"\n  Final configuration:")
print(f"    Dimensions: {used_config[0]} (reduced from {embedding_dim})")
print(f"    n_neighbors: {used_config[1]}")
print(f"    n_bits: {used_config[2]}")
print(f"    p_error: {used_p_error}")

# Note: query_embedding may have been reduced to match the compiled dimension
# ----------------------------------------------------------------------------
# Step 6: Test predictions (clear, simulate, execute)
# ----------------------------------------------------------------------------
print("\n[Step 6] Testing predictions...")
print("-" * 70)

# Test 1: Clear-text prediction (fastest, for debugging)
print("\n[Test 1] Clear-text prediction (no FHE)...")
start_time = time.time()
pred_clear = fhe_knn.predict(query_embedding)
clear_time = time.time() - start_time
print(f"  Predicted document index: {pred_clear[0]}")
print(f"  Time: {clear_time:.4f}s")

# Test 2: FHE simulation (faster than real FHE)
print("\n[Test 2] FHE simulation (faster, for development)...")
start_time = time.time()
pred_sim = fhe_knn.predict(query_embedding, fhe="simulate")
sim_time = time.time() - start_time
print(f"  Predicted document index: {pred_sim[0]}")
print(f"  Time: {sim_time:.4f}s")

# Test 3: Real FHE execution (slowest, fully encrypted)
print("\n[Test 3] Real FHE execution (fully encrypted)...")
print("  (This is the actual encrypted computation)")
start_time = time.time()
pred_fhe = fhe_knn.predict(query_embedding, fhe="execute")
fhe_time = time.time() - start_time
print(f"  Predicted document index: {pred_fhe[0]}")
print(f"  Time: {fhe_time:.4f}s")

# ----------------------------------------------------------------------------
# Step 7: Verify results and show what wiki-rag would do
# ----------------------------------------------------------------------------
print("\n[Step 7] Results and verification...")
print("-" * 70)

print(f"\n  Predictions:")
print(f"    Clear-text:   {pred_clear[0]}")
print(f"    FHE simulate: {pred_sim[0]}")
print(f"    FHE execute:  {pred_fhe[0]}")

# Check if all predictions match
if pred_clear[0] == pred_sim[0] == pred_fhe[0]:
    print(f"\n  ✓ All predictions match! (Good)")
else:
    print(f"\n  ⚠ Predictions differ (may be due to quantization/FHE precision)")

# Show what wiki-rag would do with this result
print(f"\n  What wiki-rag would do:")
print(f"    - Get document at index {pred_fhe[0]}")
print(f"    - Return: all_documents[{pred_fhe[0]}]")
print(f"    - This is the 'most common class' among {used_config[1]} nearest neighbors")

# ----------------------------------------------------------------------------
# Step 8: Debug: Show what KNN actually sees
# ----------------------------------------------------------------------------
print("\n[Step 8] Debug: What Concrete ML KNN sees...")
print("-" * 70)

print(f"\n  Input to KNN.predict():")
print(f"    Query embedding shape: {query_embedding.shape}")
print(f"    Query embedding (first 5 dims): {query_embedding[0, :5]}")
print(f"    Query embedding (last 5 dims): {query_embedding[0, -5:]}")

print(f"\n  Training data:")
print(f"    Document embeddings shape: {document_embeddings.shape}")
print(f"    Document labels (indices): {document_indices}")

print(f"\n  What KNN does internally:")
print(f"    1. Finds {used_config[1]} nearest neighbors (in encrypted space)")
print(f"    2. Looks at their class labels (document indices)")
print(f"    3. Returns most common class (document index)")
print(f"    4. Result: {pred_fhe[0]} (most common among {used_config[1]} neighbors)")

# ----------------------------------------------------------------------------
# Step 9: Manual distance computation (for comparison)
# ----------------------------------------------------------------------------
print("\n[Step 9] Manual verification (clear-text distances)...")
print("-" * 70)

# Compute distances manually (for debugging)
# Use the same dimension that was used for compilation
if used_config[0] < embedding_dim:
    doc_emb_for_dist = document_embeddings[:, :used_config[0]]
else:
    doc_emb_for_dist = document_embeddings

distances = np.linalg.norm(doc_emb_for_dist - query_embedding, axis=1)
nearest_indices = np.argsort(distances)[:used_config[1]]

print(f"  Top-{used_config[1]} nearest neighbors (by distance):")
for i, idx in enumerate(nearest_indices):
    print(f"    {i+1}. Document {idx}: distance = {distances[idx]:.4f}")

print(f"\n  Most common document index in top-{used_config[1]}: {nearest_indices[0]}")
print(f"  (KNN returns this, or most common if there are ties)")

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
This test simulates what Concrete ML KNN sees in wiki-rag:

1. Document embeddings: {n_documents} documents, {used_config[0]}D each
   (After PCA reduction: 768D → {used_config[0]}D, reduced from {embedding_dim}D for FHE compatibility)

2. Query embedding: 1 query, {used_config[0]}D
   (After PCA reduction: 768D → {used_config[0]}D)

3. Concrete ML KNN:
   - Finds {used_config[1]} nearest neighbors
   - Returns most common document index: {pred_fhe[0]}
   - Configuration: dim={used_config[0]}, n_neighbors={used_config[1]}, n_bits={used_config[2]}, p_error={used_p_error}

4. What wiki-rag does:
   - Uses this index to retrieve document: all_documents[{pred_fhe[0]}]
   - Returns the Wikipedia article

Key point: KNN returns ONE document index (classification), not top-k indices.
""")

print("=" * 70)
print("Test completed!")
print("=" * 70)

