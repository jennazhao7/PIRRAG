import faiss
import numpy as np
from typing import Optional, Tuple
import struct


def faiss_to_text(
        faiss_path: str,
        output_path: str,
        format: str = "csv",
        max_vectors: Optional[int] = None,
        include_ids: bool = True
) -> Tuple[int, int]:
    """
    Read FAISS index and output vector mappings to a text file.

    Args:
        faiss_path: Path to FAISS index file
        output_path: Path to output text file
        format: Output format ('csv', 'tsv', 'json', or 'npy')
        max_vectors: Maximum number of vectors to export (None = all)
        include_ids: Whether to include vector IDs in output

    Returns:
        Tuple of (num_vectors, dimension)
    """
    # Load FAISS index
    print(f"Loading FAISS index from {faiss_path}...")
    index = faiss.read_index(faiss_path)

    num_vectors = index.ntotal
    dimension = index.d

    print(f"Index contains {num_vectors} vectors of dimension {dimension}")

    # Limit number of vectors if specified
    if max_vectors is not None:
        num_vectors = min(num_vectors, max_vectors)

    # Check if index supports reconstruction
    if not hasattr(index, 'reconstruct') and not hasattr(index, 'reconstruct_n'):
        print("Warning: This index type does not support vector reconstruction")
        print("You may need to store vectors separately or use a different index type")
        return num_vectors, dimension

    print(f"Exporting {num_vectors} vectors to {output_path}...")

    if format == "csv":
        export_csv(index, output_path, num_vectors, dimension, include_ids, delimiter=',')
    elif format == "tsv":
        export_csv(index, output_path, num_vectors, dimension, include_ids, delimiter='\t')
    elif format == "json":
        export_json(index, output_path, num_vectors, dimension, include_ids)
    elif format == "npy":
        export_npy(index, output_path, num_vectors, dimension)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Export complete!")
    return num_vectors, dimension


def export_csv(index, output_path, num_vectors, dimension, include_ids, delimiter=','):
    """Export to CSV/TSV format."""
    with open(output_path, 'w') as f:
        # Write header
        if include_ids:
            header = f"vector_id{delimiter}" + delimiter.join([f"dim_{i}" for i in range(dimension)])
        else:
            header = delimiter.join([f"dim_{i}" for i in range(dimension)])
        f.write(header + '\n')

        # Write vectors
        for i in range(num_vectors):
            try:
                vector = index.reconstruct(i)
                if include_ids:
                    line = f"{i}{delimiter}" + delimiter.join([str(x) for x in vector])
                else:
                    line = delimiter.join([str(x) for x in vector])
                f.write(line + '\n')
            except Exception as e:
                print(f"Warning: Could not reconstruct vector {i}: {e}")
                continue

            # Progress update
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{num_vectors} vectors...")


def export_json(index, output_path, num_vectors, dimension, include_ids):
    """Export to JSON format."""
    import json

    vectors_data = []

    for i in range(num_vectors):
        try:
            vector = index.reconstruct(i)
            if include_ids:
                vectors_data.append({
                    "id": i,
                    "vector": vector.tolist()
                })
            else:
                vectors_data.append(vector.tolist())
        except Exception as e:
            print(f"Warning: Could not reconstruct vector {i}: {e}")
            continue

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{num_vectors} vectors...")

    with open(output_path, 'w') as f:
        json.dump({
            "num_vectors": len(vectors_data),
            "dimension": dimension,
            "vectors": vectors_data
        }, f, indent=2)


def export_npy(index, output_path, num_vectors, dimension):
    """Export to NumPy binary format."""
    vectors = np.zeros((num_vectors, dimension), dtype=np.float32)

    for i in range(num_vectors):
        try:
            vectors[i] = index.reconstruct(i)
        except Exception as e:
            print(f"Warning: Could not reconstruct vector {i}: {e}")
            continue

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{num_vectors} vectors...")

    np.save(output_path, vectors)


def faiss_to_text_batch(
        faiss_path: str,
        output_path: str,
        batch_size: int = 1000,
        format: str = "csv"
) -> Tuple[int, int]:
    """
    Export FAISS index using batch reconstruction (more efficient).

    Args:
        faiss_path: Path to FAISS index file
        output_path: Path to output text file
        batch_size: Number of vectors to process at once
        format: Output format

    Returns:
        Tuple of (num_vectors, dimension)
    """
    # Load index
    index = faiss.read_index(faiss_path)
    num_vectors = index.ntotal
    dimension = index.d

    print(f"Index: {num_vectors} vectors of dimension {dimension}")
    print(f"Exporting in batches of {batch_size}...")

    with open(output_path, 'w') as f:
        # Write header
        if format in ["csv", "tsv"]:
            delimiter = ',' if format == "csv" else '\t'
            header = f"vector_id{delimiter}" + delimiter.join([f"dim_{i}" for i in range(dimension)])
            f.write(header + '\n')
        else:
            delimiter = ""
        # Process in batches
        for start_idx in range(0, num_vectors, batch_size):
            end_idx = min(start_idx + batch_size, num_vectors)
            batch_count = end_idx - start_idx

            try:
                # Reconstruct batch
                if hasattr(index, 'reconstruct_n'):
                    vectors = index.reconstruct_n(start_idx, batch_count)
                else:
                    # Fallback to individual reconstruction
                    vectors = np.zeros((batch_count, dimension), dtype=np.float32)
                    for i in range(batch_count):
                        vectors[i] = index.reconstruct(start_idx + i)

                # Write batch to file
                for i, vector in enumerate(vectors):
                    actual_id = start_idx + i
                    line = f"{actual_id}{delimiter}" + delimiter.join([str(x) for x in vector])
                    f.write(line + '\n')

                print(f"  Processed {end_idx}/{num_vectors} vectors...")

            except Exception as e:
                print(f"Error processing batch starting at {start_idx}: {e}")
                continue

    return num_vectors, dimension


def faiss_index_info(faiss_path: str) -> dict:
    """
    Get information about a FAISS index without exporting.

    Args:
        faiss_path: Path to FAISS index file

    Returns:
        Dictionary with index information
    """
    index = faiss.read_index(faiss_path)

    info = {
        'num_vectors': index.ntotal,
        'dimension': index.d,
        'index_type': type(index).__name__,
        'is_trained': index.is_trained,
        'metric_type': index.metric_type,
    }

    # Check reconstruction support
    info['supports_reconstruction'] = hasattr(index, 'reconstruct')

    # Try to get additional info for specific index types
    if hasattr(index, 'nlist'):
        info['nlist'] = index.nlist
    if hasattr(index, 'nprobe'):
        info['nprobe'] = index.nprobe

    return info


def faiss_sample_to_text(
        faiss_path: str,
        output_path: str,
        sample_size: int = 100,
        random_seed: Optional[int] = 42
) -> None:
    """
    Export a random sample of vectors from FAISS index.

    Args:
        faiss_path: Path to FAISS index file
        output_path: Path to output text file
        sample_size: Number of vectors to sample
        random_seed: Random seed for reproducibility
    """
    index = faiss.read_index(faiss_path)
    num_vectors = index.ntotal
    dimension = index.d

    if random_seed is not None:
        np.random.seed(random_seed)

    # Sample random indices
    sample_indices = np.random.choice(num_vectors, size=min(sample_size, num_vectors), replace=False)
    sample_indices.sort()

    print(f"Sampling {len(sample_indices)} vectors from {num_vectors} total...")

    with open(output_path, 'w') as f:
        # Write header
        f.write(f"vector_id," + ",".join([f"dim_{i}" for i in range(dimension)]) + '\n')

        # Write sampled vectors
        for idx in sample_indices:
            try:
                vector = index.reconstruct(int(idx))
                line = f"{idx}," + ",".join([str(x) for x in vector])
                f.write(line + '\n')
            except Exception as e:
                print(f"Warning: Could not reconstruct vector {idx}: {e}")
                continue

    print(f"Sample exported to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export FAISS index to text file')
    parser.add_argument('--input', '-i', required=True, help='Input FAISS index file')
    parser.add_argument('--output', '-o', required=True, help='Output text file')
    parser.add_argument('--format', '-f', default='csv', choices=['csv', 'tsv', 'json', 'npy'],
                        help='Output format')
    parser.add_argument('--max-vectors', '-m', type=int, default=None,
                        help='Maximum number of vectors to export')
    parser.add_argument('--batch-size', '-b', type=int, default=1000,
                        help='Batch size for processing')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Export random sample of N vectors')
    parser.add_argument('--info', action='store_true',
                        help='Only show index information')

    args = parser.parse_args()

    if args.info:
        # Just show index info
        info = faiss_index_info(args.input)
        print("\n=== FAISS Index Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")
    elif args.sample:
        # Export sample
        faiss_sample_to_text(args.input, args.output, sample_size=args.sample)
    else:
        # Full export
        if args.batch_size > 1:
            num_vecs, dim = faiss_to_text_batch(
                args.input,
                args.output,
                batch_size=args.batch_size,
                format=args.format
            )
        else:
            num_vecs, dim = faiss_to_text(
                args.input,
                args.output,
                format=args.format,
                max_vectors=args.max_vectors
            )

        print(f"\nExported {num_vecs} vectors of dimension {dim}")
