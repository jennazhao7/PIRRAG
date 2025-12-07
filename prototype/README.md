# FHE Query Prototype

A complete prototype for privacy-preserving query encryption and encrypted distance computation using TenSEAL (FHE).

## Overview

This prototype implements:
1. **Client-side**: Encrypts queries using FHE (TenSEAL/CKKS)
2. **Server-side**: Computes encrypted distances to centroids without decrypting
3. **Client-side**: Decrypts results and selects top-k

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Client: Encrypt a Query

```bash
python fhe_query_client.py \
    --query "What is machine learning?" \
    --output-dir ./encrypted_queries
```

This creates:
- `encrypted_query.bin` - Encrypted query vector
- `encrypted_norm_squared.bin` - Encrypted ||q||²
- `context_public.json` - Public key (for server)
- `query_metadata.json` - Metadata

### 2. Server: Compute Encrypted Distances

```bash
python fhe_query_server.py \
    --centroids-path ./data/centroids.npy \
    --encrypted-query ./encrypted_queries/encrypted_query.bin \
    --encrypted-norm ./encrypted_queries/encrypted_norm_squared.bin \
    --output-dir ./encrypted_distances
```

This computes encrypted distances to all centroids and saves them.

### 3. Client: Decrypt and Get Top-K

```bash
python fhe_query_client.py \
    --decrypt-distances ./encrypted_distances \
    --top-k 100 \
    --results-dir ./decrypted_results
```

This decrypts distances and returns top-100 smallest.

## Complete Workflow Example

```bash
# Step 1: Encrypt query
python fhe_query_client.py --query "Your question here" --output-dir ./encrypted_queries

# Step 2: Server computes distances (on server machine)
python fhe_query_server.py \
    --centroids-path ./data/centroids.npy \
    --encrypted-query ./encrypted_queries/encrypted_query.bin \
    --encrypted-norm ./encrypted_queries/encrypted_norm_squared.bin \
    --output-dir ./encrypted_distances

# Step 3: Client decrypts results
python fhe_query_client.py \
    --decrypt-distances ./encrypted_distances \
    --top-k 100 \
    --results-dir ./decrypted_results
```

## File Structure

```
prototype/
├── fhe_query_client.py      # Client-side encryption/decryption
├── fhe_query_server.py       # Server-side distance computation
├── rag_utils.py             # Embedding utilities
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── example_workflow.sh      # Example script
└── data/
    ├── centroids.npy        # IVF cluster centroids (4096 x 768)
    └── lists.json           # Inverted lists mapping clusters to vectors
```

## Requirements

- Python 3.8+
- TenSEAL (for FHE operations)
- NumPy
- LangChain Community (for embeddings)
- Sentence Transformers (for BGE model)

## Security Notes

- **Client** keeps the secret key (never shared)
- **Server** only receives public key (cannot decrypt)
- All computations on server are homomorphic (encrypted)
- Query privacy is preserved throughout

## Advanced Usage

### Custom Embedding Model

Modify `rag_utils.py` to use a different embedding model:

```python
class PromptedBGE(HuggingFaceEmbeddings):
    def __init__(self, model_name="BAAI/bge-large-en"):
        super().__init__(model_name=model_name)
```

### Adjust FHE Parameters

In `fhe_query_client.py`, modify encryption parameters:

```python
client = FHEQueryClient(
    poly_modulus_degree=16384,  # Larger = more secure, slower
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
```

## Troubleshooting

**Error: Model not found**
- The embedding model downloads automatically on first use
- Ensure internet connection for first run

**Error: TenSEAL not found**
- Install: `pip install tenseal`
- For GPU support: `pip install tenseal[gpu]` (optional)

**Error: Context file not found**
- Client creates encryption context automatically
- Make sure client runs before server

## License

See parent repository for license information.

## PIR Readme

**Guide:**
Generate uniform_index_1024.txt by running pickle_processor.py
and change to correct path on line 268,269

Generate faiss.json by running faiss_processor.py with command line arguments:
--input ../index.faiss --batch-size 1 --output ../faiss.json --format json

Run server_exe twice with the following command line arguments:
-port 50051 -database /path/to/uniform_index_1024.txt
-port 50052 -database /path/to/faiss.json

Change the correct paths in:
Line 496 of fhe_query_client.py with correct path
Line 650 of fhe_query_client.py

Go ham
