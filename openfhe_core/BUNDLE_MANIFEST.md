# OpenFHE Bundle Manifest

This manifest defines the minimal file set to share with collaborators for
single-query OpenFHE FHE execution.

## Required files

- `openfhe_core/CMakeLists.txt`
- `openfhe_core/README.md`
- `openfhe_core/include/io_utils.h`
- `openfhe_core/src/io_utils.cpp`
- `openfhe_core/src/openfhe_keygen.cpp`
- `openfhe_core/src/openfhe_encrypt_query.cpp`
- `openfhe_core/src/openfhe_compute_distances.cpp`
- `openfhe_core/src/openfhe_decrypt_topk.cpp`
- `prototype/fhe_backend.py`
- `prototype/fhe_query_client.py`
- `prototype/fhe_query_server.py`

## Optional files

Only needed for text-to-embedding query mode:

- `prototype/rag_utils.py`

## Runtime data (not source)

Collaborators must provide:

- `centroids.npy`
- one query vector `.npy` (or text query + embedding dependencies)
