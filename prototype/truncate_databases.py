import pickle
import os
from typing import Union, Any

from langchain_community.docstore import InMemoryDocstore


def truncate_file(
        input_path: str,
        output_path: str,
        max_entries: int,
        file_type: str = "auto"
) -> None:
    """
    Truncate a FAISS index or pickle file to a specified number of entries.

    Parameters:
    -----------
    input_path : str
        Path to the input file (FAISS index or pickle file)
    output_path : str
        Path where the truncated file will be saved
    max_entries : int
        Maximum number of entries to keep
    file_type : str, optional
        Type of file: 'faiss', 'pickle', or 'auto' (default: 'auto')
        If 'auto', the type is inferred from file extension

    Returns:
    --------
    None

    Examples:
    ---------
    >>> truncate_file('large_index.faiss', 'small_index.faiss', 1000)
    >>> truncate_file('data.pkl', 'truncated_data.pkl', 500, file_type='pickle')
    """

    # Determine file type
    if file_type == "auto":
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.faiss', '.index']:
            file_type = 'faiss'
        elif ext in ['.pkl', '.pickle']:
            file_type = 'pickle'
        else:
            raise ValueError(
                f"Cannot auto-detect file type from extension '{ext}'. "
                "Please specify file_type as 'faiss' or 'pickle'"
            )

    if file_type == 'faiss':
        truncate_faiss(input_path, output_path, max_entries)
    elif file_type == 'pickle':
        truncate_pickle(input_path, output_path, max_entries)
    else:
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'faiss' or 'pickle'")


def truncate_faiss(input_path: str, output_path: str, max_entries: int) -> None:
    """
    Truncate a FAISS index to a specified number of entries.

    Parameters:
    -----------
    input_path : str
        Path to the input FAISS index file
    output_path : str
        Path where the truncated FAISS index will be saved
    max_entries : int
        Maximum number of vectors to keep in the index
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "FAISS is not installed. Install it with: pip install faiss-cpu "
            "or pip install faiss-gpu"
        )

    # Load the FAISS index
    index = faiss.read_index(input_path)

    # Get current number of vectors
    n_total = index.ntotal

    if max_entries >= n_total:
        print(f"Index has {n_total} entries, which is <= {max_entries}. Copying file as-is.")
        # Just copy the file
        import shutil
        shutil.copy2(input_path, output_path)
        return

    # Create a new index with the same parameters
    dimension = index.d

    # Determine the index type and create a new one
    if isinstance(index, faiss.IndexFlat):
        new_index = faiss.IndexFlatL2(dimension)
    elif isinstance(index, faiss.IndexFlatIP):
        new_index = faiss.IndexFlatIP(dimension)
    elif isinstance(index, faiss.IndexIVFFlat):
        # For IVF indices, we need to preserve the quantizer
        new_index = faiss.IndexIVFFlat(index.quantizer, dimension, index.nlist)
        new_index.nprobe = index.nprobe
    else:
        # For other index types, try to clone and clear
        new_index = faiss.clone_index(index)
        new_index.reset()

    # Extract the first max_entries vectors
    vectors = index.reconstruct_n(0, min(max_entries, n_total))

    # Add to new index (train first if needed)
    if hasattr(new_index, 'is_trained') and not new_index.is_trained:
        new_index.train(vectors)

    new_index.add(vectors)

    # Save the truncated index
    faiss.write_index(new_index, output_path)

    print(f"Truncated FAISS index from {n_total} to {new_index.ntotal} entries")
    print(f"Saved to: {output_path}")


def truncate_pickle(input_path: str, output_path: str, max_entries: int) -> None:
    """
    Truncate a pickle file to a specified number of entries.
    Assumes the pickled object is a list, tuple, dict, or other sequence/collection.

    Parameters:
    -----------
    input_path : str
        Path to the input pickle file
    output_path : str
        Path where the truncated pickle will be saved
    max_entries : int
        Maximum number of entries to keep
    """
    # Load the pickle file
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different data types
    if isinstance(data, (list, tuple)):
        print("Truncated pickle from list")
        print(type(data[0]))
        print(type(data[1]))
        original_len = len(data[1])
        print(data[1][4999])
        (entrydocstore, keysdict) = data
        entrydict = entrydocstore.__dict__
        print(type(entrydict))
        truncated_data_entries = {}
        truncated_data_keys = {}
        for i in range(max_entries):
            truncated_data_keys[i] = keysdict[i]
            key = keysdict[i]
            truncated_data_entries[key] = entrydocstore.search(keysdict[i])
        print(truncated_data_keys[4999])
        #truncated_data_entries = InMemoryDocstore(list(list(entrydict.values())[0].values())[:max_entries])
            #InMemoryDocstore(dict(list(entrydict.items())[:max_entries])))
        truncated_data = (InMemoryDocstore(truncated_data_entries), truncated_data_keys)
        print(type(truncated_data))
        print("What the fuck")
    elif isinstance(data, dict):
        truncated_data = dict(list(data.items())[:max_entries])
        print("Truncated pickle from dict")
        original_len = len(data)
    elif hasattr(data, '__len__') and hasattr(data, '__getitem__'):
        # Generic sequence-like object
        truncated_data = [data[i] for i in range(min(max_entries, len(data)))]
        original_len = len(data)
    else:
        raise TypeError(
            f"Unsupported data type in pickle file: {type(data)}. "
            "Expected list, tuple, dict, or sequence-like object."
        )

    if max_entries >= original_len:
        print(f"Pickle contains {original_len} entries, which is <= {max_entries}. Copying file as-is.")
        import shutil
        shutil.copy2(input_path, output_path)
        return

    # Save the truncated data
    with open(output_path, 'wb') as f:
        pickle.dump(truncated_data, f)

    print(f"Truncated pickle from {original_len} to {len(truncated_data)} entries")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python truncate_file.py <input_file> <output_file> <max_entries> [file_type]")
        print("Example: python truncate_file.py index.faiss truncated.faiss 1000")
        print("Example: python truncate_file.py data.pkl truncated.pkl 500 pickle")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_entries = int(sys.argv[3])
    file_type = sys.argv[4] if len(sys.argv) > 4 else "auto"

    truncate_file(input_file, output_file, max_entries, file_type)
