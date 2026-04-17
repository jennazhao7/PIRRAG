#!/usr/bin/env python3
"""
FHE backend abstraction for single-query workflow.

This module keeps the existing file contract used by prototype scripts while
allowing multiple crypto engines.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class BackendError(RuntimeError):
    """Raised when backend setup or execution fails."""


@dataclass
class OpenFHEBinarySet:
    keygen: Path
    encrypt_query: Path
    compute_distances: Path
    decrypt_topk: Path


class SingleQueryBackend:
    """Interface for single-query encryption/server/decryption operations."""

    def ensure_context(
        self,
        context_path: Path,
        poly_modulus_degree: int,
        coeff_mod_bit_sizes: List[int],
    ) -> None:
        raise NotImplementedError

    def encrypt_query(
        self,
        context_path: Path,
        query_vector: np.ndarray,
        output_path: Path,
        poly_modulus_degree: int,
        coeff_mod_bit_sizes: List[int],
    ) -> dict:
        raise NotImplementedError

    def compute_distances(
        self,
        context_path: Path,
        centroids_path: Path,
        encrypted_query_path: Path,
        encrypted_norm_path: Path,
        output_path: Path,
        batch_size: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    def decrypt_top_k(
        self,
        context_path: Path,
        encrypted_distances_dir: Path,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class OpenFHECppBackend(SingleQueryBackend):
    """OpenFHE C++ backend using CLI executables."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root
            else Path(__file__).resolve().parents[1]
        )
        self.core_dir = self.repo_root / "openfhe_core"
        self.build_dir = self.core_dir / "build"
        self._binaries: Optional[OpenFHEBinarySet] = None

    def _run(self, args: List[str], cwd: Optional[Path] = None) -> None:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise BackendError(
                "OpenFHE command failed.\n"
                f"Command: {' '.join(args)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    def _find_binaries(self) -> Optional[OpenFHEBinarySet]:
        candidates = [
            self.build_dir / "bin",
            self.build_dir,
        ]
        for base in candidates:
            keygen = base / "openfhe_keygen"
            encrypt = base / "openfhe_encrypt_query"
            server = base / "openfhe_compute_distances"
            decrypt = base / "openfhe_decrypt_topk"
            if all(path.exists() for path in [keygen, encrypt, server, decrypt]):
                return OpenFHEBinarySet(
                    keygen=keygen,
                    encrypt_query=encrypt,
                    compute_distances=server,
                    decrypt_topk=decrypt,
                )
        return None

    def _ensure_binaries(self) -> OpenFHEBinarySet:
        if self._binaries is not None:
            return self._binaries

        found = self._find_binaries()
        if found:
            self._binaries = found
            return found

        self.build_dir.mkdir(parents=True, exist_ok=True)
        self._run(
            ["cmake", "-S", str(self.core_dir), "-B", str(self.build_dir)],
            cwd=self.repo_root,
        )
        self._run(
            ["cmake", "--build", str(self.build_dir), "-j"],
            cwd=self.repo_root,
        )

        found = self._find_binaries()
        if not found:
            raise BackendError(
                "OpenFHE binaries were not found after build. "
                f"Expected under {self.build_dir}."
            )
        self._binaries = found
        return found

    def ensure_context(
        self,
        context_path: Path,
        poly_modulus_degree: int,
        coeff_mod_bit_sizes: List[int],
    ) -> None:
        bins = self._ensure_binaries()
        context_path = Path(context_path)
        context_path.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                str(bins.keygen),
                "--context-dir",
                str(context_path),
                "--poly-modulus-degree",
                str(poly_modulus_degree),
                "--coeff-mod-bit-sizes",
                ",".join(str(x) for x in coeff_mod_bit_sizes),
            ]
        )

    def encrypt_query(
        self,
        context_path: Path,
        query_vector: np.ndarray,
        output_path: Path,
        poly_modulus_degree: int,
        coeff_mod_bit_sizes: List[int],
    ) -> dict:
        self.ensure_context(context_path, poly_modulus_degree, coeff_mod_bit_sizes)
        bins = self._ensure_binaries()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        vector_file = output_path / "query_vector.txt"
        np.savetxt(vector_file, query_vector.astype(np.float64), fmt="%.12g")

        self._run(
            [
                str(bins.encrypt_query),
                "--context-dir",
                str(context_path),
                "--input-vector",
                str(vector_file),
                "--output-dir",
                str(output_path),
            ]
        )

        norm_sq = float(np.sum(query_vector**2))
        return {
            "query_dim": int(query_vector.shape[0]),
            "plaintext_norm_squared": norm_sq,
            "plaintext_norm": float(np.sqrt(norm_sq)),
            "poly_modulus_degree": poly_modulus_degree,
            "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
            "backend": "openfhe_cpp",
        }

    def _convert_centroids_to_text(self, centroids_path: Path, work_dir: Path) -> Path:
        centroids = np.load(centroids_path).astype(np.float64)
        if centroids.ndim != 2:
            raise BackendError(
                f"Expected 2D centroids array at {centroids_path}, got {centroids.ndim}D"
            )
        out_file = work_dir / "centroids.txt"
        with out_file.open("w", encoding="utf-8") as f:
            for row in centroids:
                f.write(" ".join(f"{float(v):.12g}" for v in row))
                f.write("\n")
        return out_file

    def compute_distances(
        self,
        context_path: Path,
        centroids_path: Path,
        encrypted_query_path: Path,
        encrypted_norm_path: Path,
        output_path: Path,
        batch_size: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        bins = self._ensure_binaries()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        centroids_txt = self._convert_centroids_to_text(Path(centroids_path), output_path)

        cmd = [
            str(bins.compute_distances),
            "--context-dir",
            str(context_path),
            "--centroids-file",
            str(centroids_txt),
            "--encrypted-query",
            str(encrypted_query_path),
            "--encrypted-norm",
            str(encrypted_norm_path),
            "--output-dir",
            str(output_path),
        ]
        if batch_size is not None and batch_size > 0:
            cmd.extend(["--batch-size", str(batch_size)])
        if num_threads is not None and num_threads > 0:
            cmd.extend(["--num-threads", str(num_threads)])
        self._run(cmd)

    def decrypt_top_k(
        self,
        context_path: Path,
        encrypted_distances_dir: Path,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bins = self._ensure_binaries()
        encrypted_distances_dir = Path(encrypted_distances_dir)
        self._run(
            [
                str(bins.decrypt_topk),
                "--context-dir",
                str(context_path),
                "--encrypted-distances-dir",
                str(encrypted_distances_dir),
                "--top-k",
                str(top_k),
                "--output-json",
                str(encrypted_distances_dir / "top_k_results.json"),
            ]
        )

        with (encrypted_distances_dir / "top_k_results.json").open(
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        distances = np.array(data["distances"], dtype=np.float32)
        indices = np.array(data["centroid_indices"], dtype=np.int32)
        return distances, indices


class TensealBackend(SingleQueryBackend):
    """Legacy TenSEAL backend kept as a migration fallback."""

    def __init__(self):
        try:
            import tenseal as ts  # type: ignore
        except ImportError as exc:
            raise BackendError(
                "TenSEAL backend requested but tenseal is not installed."
            ) from exc
        self.ts = ts

    def _context_file(self, context_path: Path) -> Path:
        return Path(context_path) / "context.json"

    def _public_context_file(self, context_path: Path) -> Path:
        return Path(context_path) / "context_public.json"

    def ensure_context(
        self,
        context_path: Path,
        poly_modulus_degree: int,
        coeff_mod_bit_sizes: List[int],
    ) -> None:
        context_path = Path(context_path)
        context_path.mkdir(parents=True, exist_ok=True)
        context_file = self._context_file(context_path)
        public_file = self._public_context_file(context_path)
        if context_file.exists() and public_file.exists():
            return

        context = self.ts.context(
            self.ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        context.generate_galois_keys()
        context.global_scale = 2**40
        with context_file.open("wb") as f:
            f.write(context.serialize(save_secret_key=True))
        with public_file.open("wb") as f:
            f.write(context.serialize(save_secret_key=False))

    def _load_context(self, context_path: Path):
        with self._context_file(context_path).open("rb") as f:
            return self.ts.context_from(f.read())

    def _load_public_context(self, context_path: Path):
        with self._public_context_file(context_path).open("rb") as f:
            return self.ts.context_from(f.read())

    def encrypt_query(
        self,
        context_path: Path,
        query_vector: np.ndarray,
        output_path: Path,
        poly_modulus_degree: int,
        coeff_mod_bit_sizes: List[int],
    ) -> dict:
        self.ensure_context(context_path, poly_modulus_degree, coeff_mod_bit_sizes)
        context = self._load_context(Path(context_path))
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        encrypted_query = self.ts.ckks_vector(context, query_vector.astype(float).tolist())
        squared_norm = float(np.sum(query_vector**2))
        encrypted_norm = self.ts.ckks_vector(context, [squared_norm])

        with (output_path / "encrypted_query.bin").open("wb") as f:
            f.write(encrypted_query.serialize())
        with (output_path / "encrypted_norm_squared.bin").open("wb") as f:
            f.write(encrypted_norm.serialize())
        with (output_path / "context_public.json").open("wb") as f:
            f.write(context.serialize(save_secret_key=False))

        return {
            "query_dim": int(query_vector.shape[0]),
            "plaintext_norm_squared": squared_norm,
            "plaintext_norm": float(np.sqrt(squared_norm)),
            "poly_modulus_degree": poly_modulus_degree,
            "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
            "backend": "tenseal",
        }

    def compute_distances(
        self,
        context_path: Path,
        centroids_path: Path,
        encrypted_query_path: Path,
        encrypted_norm_path: Path,
        output_path: Path,
        batch_size: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        context = self._load_public_context(Path(context_path))
        centroids = np.load(centroids_path).astype(np.float32)
        centroid_norms = np.sum(centroids**2, axis=1).astype(np.float32)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with Path(encrypted_query_path).open("rb") as f:
            encrypted_query = self.ts.ckks_vector_from(context, f.read())
        with Path(encrypted_norm_path).open("rb") as f:
            encrypted_norm = self.ts.ckks_vector_from(context, f.read())

        for i, centroid in enumerate(centroids):
            dot = (encrypted_query * centroid.tolist()).sum()
            centroid_norm = self.ts.ckks_vector(context, [float(centroid_norms[i])])
            distance = (encrypted_norm + centroid_norm) - (dot + dot)
            with (output_path / f"encrypted_distance_{i:04d}.bin").open("wb") as f:
                f.write(distance.serialize())

        with (output_path / "distances_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_distances": int(len(centroids)),
                    "n_centroids": int(len(centroids)),
                    "centroid_dim": int(centroids.shape[1]),
                    "backend": "tenseal",
                },
                f,
                indent=2,
            )

    def _parse_n_distances(self, metadata_path: Path) -> int:
        with metadata_path.open("r", encoding="utf-8") as f:
            text = f.read()
        match = re.search(r'"n_distances"\s*:\s*(\d+)', text)
        if not match:
            raise BackendError(f"Could not parse n_distances from {metadata_path}")
        return int(match.group(1))

    def decrypt_top_k(
        self,
        context_path: Path,
        encrypted_distances_dir: Path,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        context = self._load_context(Path(context_path))
        encrypted_distances_dir = Path(encrypted_distances_dir)
        n_distances = self._parse_n_distances(
            encrypted_distances_dir / "distances_metadata.json"
        )
        distances = []
        for i in range(n_distances):
            dist_file = encrypted_distances_dir / f"encrypted_distance_{i:04d}.bin"
            if not dist_file.exists():
                continue
            with dist_file.open("rb") as f:
                ct = self.ts.ckks_vector_from(context, f.read())
            distances.append(float(ct.decrypt()[0]))

        all_distances = np.array(distances, dtype=np.float32)
        top_indices = np.argsort(all_distances)[:top_k]
        top_distances = all_distances[top_indices]
        return top_distances, top_indices.astype(np.int32)


def create_backend(name: str = "openfhe_cpp") -> SingleQueryBackend:
    name = (name or "openfhe_cpp").strip().lower()
    if name == "openfhe_cpp":
        return OpenFHECppBackend()
    if name == "tenseal":
        return TensealBackend()
    raise BackendError(f"Unknown backend: {name}")

