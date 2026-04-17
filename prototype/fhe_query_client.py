#!/usr/bin/env python3
"""
Client-side single-query FHE wrapper.

The crypto backend is now pluggable; default is OpenFHE C++ binaries.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from fhe_backend import BackendError, create_backend


class FHEQueryClient:
    """Client-side query embedding + backend encryption/decryption wrapper."""

    def __init__(
        self,
        context_path: Path = None,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        embedding_model_name: str = "BAAI/bge-base-en",
        backend: str = "openfhe_cpp",
    ):
        self.context_path = Path(context_path) if context_path else Path("./fhe_context")
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.backend_name = backend
        if self.backend_name == "openfhe_cpp" and self.poly_modulus_degree < 16384:
            print(
                "OpenFHE backend requires ring dimension >= 16384 on this install; "
                "upgrading poly_modulus_degree to 16384."
            )
            self.poly_modulus_degree = 16384
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.backend = create_backend(backend)
        self.backend.ensure_context(
            self.context_path, self.poly_modulus_degree, self.coeff_mod_bit_sizes
        )

    def embed_query(self, query_text: str) -> np.ndarray:
        if self.embeddings is None:
            from rag_utils import PromptedBGE
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embeddings = PromptedBGE(model_name=self.embedding_model_name)
        embedding = self.embeddings.embed_query(query_text)
        return np.array(embedding, dtype=np.float32)

    def process_query(
        self, query: Union[str, np.ndarray], output_path: Path
    ) -> Tuple[Path, Path, dict]:
        if isinstance(query, str):
            print(f"Embedding query: '{query[:80]}'")
            query_vector = self.embed_query(query)
        else:
            query_vector = np.array(query, dtype=np.float32).flatten()

        metadata = self.backend.encrypt_query(
            context_path=self.context_path,
            query_vector=query_vector,
            output_path=Path(output_path),
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        query_file = Path(output_path) / "encrypted_query.bin"
        norm_file = Path(output_path) / "encrypted_norm_squared.bin"
        return query_file, norm_file, metadata

    def save_query_metadata(self, metadata: dict, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_file = output_path / "query_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return metadata_file

    def decrypt_distances(
        self, encrypted_distances_dir: Path, top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.backend.decrypt_top_k(
            context_path=self.context_path,
            encrypted_distances_dir=Path(encrypted_distances_dir),
            top_k=top_k,
        )

    def save_decrypted_results(
        self,
        top_k_distances: np.ndarray,
        top_k_indices: np.ndarray,
        output_path: Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        results = {
            "top_k": int(len(top_k_distances)),
            "distances": top_k_distances.tolist(),
            "centroid_indices": top_k_indices.tolist(),
            "min_distance": float(top_k_distances.min()) if len(top_k_distances) else 0.0,
            "max_distance": float(top_k_distances.max()) if len(top_k_distances) else 0.0,
            "mean_distance": float(top_k_distances.mean()) if len(top_k_distances) else 0.0,
            "backend": self.backend_name,
        }
        results_file = output_path / "top_k_results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        np.save(output_path / "top_k_distances.npy", top_k_distances)
        np.save(output_path / "top_k_indices.npy", top_k_indices)
        return results_file


def main():
    parser = argparse.ArgumentParser(description="Client-side FHE query encryption")
    parser.add_argument("--query", type=str, required=False)
    parser.add_argument("--context-path", type=str, default="./fhe_context")
    parser.add_argument("--output-dir", type=str, default="./encrypted_queries")
    parser.add_argument(
        "--poly-modulus-degree", type=int, default=8192, choices=[8192, 16384]
    )
    parser.add_argument("--query-vector", type=str, default=None)
    parser.add_argument("--decrypt-distances", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--results-dir", type=str, default="./decrypted_results")
    parser.add_argument(
        "--backend",
        type=str,
        default="openfhe_cpp",
        choices=["openfhe_cpp", "tenseal"],
        help="Crypto backend implementation",
    )
    args = parser.parse_args()

    try:
        client = FHEQueryClient(
            context_path=args.context_path,
            poly_modulus_degree=args.poly_modulus_degree,
            backend=args.backend,
        )
    except BackendError as exc:
        raise SystemExit(f"Failed to initialize backend '{args.backend}': {exc}") from exc

    if args.decrypt_distances:
        top_k_distances, top_k_indices = client.decrypt_distances(
            Path(args.decrypt_distances), top_k=args.top_k
        )
        results_file = client.save_decrypted_results(
            top_k_distances, top_k_indices, Path(args.results_dir)
        )
        print(f"Decrypted {len(top_k_distances)} distances.")
        print(f"Results saved to: {results_file}")
        return

    if not args.query and not args.query_vector:
        parser.error("--query or --query-vector is required for encryption mode")

    query_value: Union[str, np.ndarray]
    if args.query_vector:
        query_value = np.load(args.query_vector).astype(np.float32)
    else:
        query_value = args.query

    query_file, norm_file, metadata = client.process_query(
        query=query_value, output_path=Path(args.output_dir)
    )
    metadata_file = client.save_query_metadata(metadata, Path(args.output_dir))
    print(f"Encrypted query file: {query_file}")
    print(f"Encrypted norm file: {norm_file}")
    print(f"Metadata file: {metadata_file}")


if __name__ == "__main__":
    main()

