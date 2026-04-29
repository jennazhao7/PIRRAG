#!/usr/bin/env python3

from pathlib import Path

import numpy as np

import fhe_query_client as client_mod
import fhe_query_server as server_mod


class DummyEmbeddings:
    def embed_query(self, text):
        return [0.25, 0.5, 0.75]


class DummyBackend:
    def ensure_context(self, context_path, poly_modulus_degree, coeff_mod_bit_sizes):
        Path(context_path).mkdir(parents=True, exist_ok=True)

    def encrypt_query(
        self,
        context_path,
        query_vector,
        output_path,
        poly_modulus_degree,
        coeff_mod_bit_sizes,
    ):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "encrypted_query.bin").write_bytes(b"query")
        (output_path / "encrypted_norm_squared.bin").write_bytes(b"norm")
        return {
            "query_dim": int(query_vector.shape[0]),
            "plaintext_norm_squared": float(np.sum(query_vector**2)),
            "plaintext_norm": float(np.sqrt(np.sum(query_vector**2))),
            "poly_modulus_degree": poly_modulus_degree,
            "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
            "backend": "dummy",
        }

    def compute_distances(
        self,
        context_path,
        centroids_path,
        encrypted_query_path,
        encrypted_norm_path,
        output_path,
    ):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "encrypted_distance_0000.bin").write_bytes(b"d0")
        (output_path / "distances_metadata.json").write_text(
            '{"n_distances": 1, "n_centroids": 1, "centroid_dim": 3}'
        )

    def decrypt_top_k(self, context_path, encrypted_distances_dir, top_k):
        return np.array([0.123], dtype=np.float32), np.array([0], dtype=np.int32)


def _patch_backend(monkeypatch):
    monkeypatch.setattr(client_mod, "create_backend", lambda name: DummyBackend())
    monkeypatch.setattr(server_mod, "create_backend", lambda name: DummyBackend())
    monkeypatch.setattr(client_mod, "PromptedBGE", lambda model_name: DummyEmbeddings())


def test_single_query_contract(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    context = tmp_path / "ctx"
    output = tmp_path / "encrypted_queries"
    distance_output = tmp_path / "encrypted_distances"
    result_output = tmp_path / "decrypted_results"

    query_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    centroids_path = tmp_path / "centroids.npy"
    np.save(centroids_path, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    client = client_mod.FHEQueryClient(context_path=context, backend="openfhe_cpp")
    server = server_mod.FHEQueryServer(
        context_path=context,
        centroids_path=centroids_path,
        backend="openfhe_cpp",
    )

    query_file, norm_file, metadata = client.process_query(query_vector, output)
    metadata_file = client.save_query_metadata(metadata, output)

    assert query_file.exists()
    assert norm_file.exists()
    assert metadata_file.exists()
    assert metadata["backend"] == "dummy"

    server.compute_distances(query_file, norm_file, distance_output)
    assert (distance_output / "distances_metadata.json").exists()
    assert (distance_output / "encrypted_distance_0000.bin").exists()

    distances, indices = client.decrypt_distances(distance_output, top_k=1)
    assert distances.shape == (1,)
    assert indices.shape == (1,)

    results_file = client.save_decrypted_results(distances, indices, result_output)
    assert results_file.exists()
    assert (result_output / "top_k_distances.npy").exists()
    assert (result_output / "top_k_indices.npy").exists()

