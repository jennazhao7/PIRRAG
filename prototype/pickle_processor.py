import pickle
import numpy as np
from typing import Any, Tuple, List


def uniformize_pickle_entries(
        input_pickle_path: str,
        output_pickle_path: str,
        pad_value: Any = 0,
        truncate: bool = True
) -> Tuple[int, int]:
    """
    Process a pickle file to create uniform-size entries.

    Args:
        input_pickle_path: Path to input pickle file
        output_pickle_path: Path to output pickle file
        pad_value: Value to use for padding (default: 0)
        truncate: If True, truncate longer entries; if False, pad to max size

    Returns:
        tuple: (uniform_size, number_of_entries)
    """
    # Load the pickle file
    with open(input_pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different data types
    if isinstance(data, dict):
        entries = list(data.values())
        keys = list(data.keys())
        is_dict = True
    elif isinstance(data, (list, tuple)):
        entries = list(data)
        keys = None
        is_dict = False
    elif isinstance(data, np.ndarray):
        entries = list(data)
        keys = None
        is_dict = False
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Determine the uniform size
    if truncate:
        # Use the minimum size (truncate all to smallest)
        uniform_size = min(len(entry) if hasattr(entry, '__len__') else 1
                           for entry in entries)
    else:
        # Use the maximum size (pad all to largest)
        uniform_size = max(len(entry) if hasattr(entry, '__len__') else 1
                           for entry in entries)

    # Process entries to uniform size
    uniform_entries = []
    for entry in entries:
        if isinstance(entry, (list, tuple)):
            if len(entry) > uniform_size:
                uniform_entry = list(entry[:uniform_size])
            else:
                uniform_entry = list(entry) + [pad_value] * (uniform_size - len(entry))
        elif isinstance(entry, np.ndarray):
            if len(entry) > uniform_size:
                uniform_entry = entry[:uniform_size]
            else:
                padding = np.full(uniform_size - len(entry), pad_value, dtype=entry.dtype)
                uniform_entry = np.concatenate([entry, padding])
        elif isinstance(entry, str):
            if len(entry) > uniform_size:
                uniform_entry = entry[:uniform_size]
            else:
                uniform_entry = entry + pad_value * (uniform_size - len(entry))
        else:
            # For single values or unsupported types, convert to list
            uniform_entry = [entry] + [pad_value] * (uniform_size - 1)

        uniform_entries.append(uniform_entry)

    # Reconstruct data structure
    if is_dict:
        output_data = dict(zip(keys, uniform_entries))
    else:
        # Try to maintain original type
        if isinstance(data, np.ndarray):
            output_data = np.array(uniform_entries)
        elif isinstance(data, tuple):
            output_data = tuple(uniform_entries)
        else:
            output_data = uniform_entries

    # Save to output pickle file
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(output_data, f)

    num_entries = len(uniform_entries)

    print(f"Processed {num_entries} entries")
    print(f"Uniform size: {uniform_size}")
    print(f"Output saved to: {output_pickle_path}")

    return uniform_size, num_entries


# Example usage and variants:

def uniformize_pickle_with_stats(
        input_pickle_path: str,
        output_pickle_path: str,
        target_size: int = None,
        pad_value: Any = 0
) -> Tuple[int, int, dict]:
    """
    Uniformize pickle with detailed statistics.

    Args:
        input_pickle_path: Path to input pickle file
        output_pickle_path: Path to output pickle file
        target_size: Specific target size (if None, uses max size)
        pad_value: Value to use for padding

    Returns:
        tuple: (uniform_size, number_of_entries, statistics_dict)
    """
    with open(input_pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Convert to list of entries
    if isinstance(data, dict):
        entries = list(data.values())
        keys = list(data.keys())
        is_dict = True
    elif isinstance(data,list):
        entries = list(data) if not isinstance(data, np.ndarray) else list(data)
        keys = None
        is_dict = False
    else:
        (entrydocstore, keysdict) = data
        entrydict = entrydocstore.__dict__
        dockeys = list(list(entrydict.values())[0].keys())
        entrydocs = list(list(entrydict.values())[0].values())
        print(len(entrydocs))
        #print(list(entrydict.values())[0])
        entries = []
        for value in entrydocs:
            entries.append(value.page_content[0:64])
        print(entries[0])
        print(len(entries[0]))
        keys = list(keysdict.keys())
        print(len(keys))
        is_dict = False
        print("What the fuck")


    # Calculate statistics
    sizes = [len(entry) if hasattr(entry, '__len__') else 1 for entry in entries]
    stats = {
        'min_size': min(sizes),
        'max_size': max(sizes),
        'mean_size': np.mean(sizes),
        'median_size': np.median(sizes),
        'original_sizes': sizes
    }

    # Determine target size
    if target_size is None:
        target_size = stats['max_size']

    # Uniformize entries
    uniform_entries = []
    for entry in entries:
        if isinstance(entry, np.ndarray):
            if len(entry) > target_size:
                uniform_entry = entry[:target_size]
            else:
                padding = np.full(target_size - len(entry), pad_value, dtype=entry.dtype)
                uniform_entry = np.concatenate([entry, padding])
        elif isinstance(entry, (list, tuple)):
            if len(entry) > target_size:
                uniform_entry = list(entry[:target_size])
            else:
                uniform_entry = list(entry) + [pad_value] * (target_size - len(entry))
        else:
            '''
            uniform_entry = []
            for character in entry:
                uniform_entry.append(character)
            while len(uniform_entry) < target_size:
                uniform_entry.append(pad_value)
            '''
            uniform_entry = ""
            for character in entry:
                uniform_entry += character
            while len(uniform_entry) < target_size:
                uniform_entry += pad_value

        uniform_entries.append(uniform_entry)

    # Save output
    if is_dict:
        output_data = dict(zip(keys, uniform_entries))
    else:
        output_data = np.array(uniform_entries) if isinstance(data, np.ndarray) else uniform_entries

    #with open(output_pickle_path, 'wb') as f:
    #    pickle.dump(output_data, f)

    output_data = ""
    for e in uniform_entries:
        output_data += e
        output_data += chr(31)

    with open(output_pickle_path, 'w') as f:
        f.write(output_data)

    print(f"\n=== Uniformization Statistics ===")
    print(f"Number of entries: {len(entries)}")
    print(f"Original size range: {stats['min_size']} - {stats['max_size']}")
    print(f"Target uniform size: {target_size}")
    print(f"Mean original size: {stats['mean_size']:.2f}")
    print(f"Median original size: {stats['median_size']:.2f}")

    return target_size, len(entries), stats


if __name__ == "__main__":
    """
    # Example 1: Basic usage
    print("Example 1: Basic uniformization")

    # Create sample data
    sample_data = {
        'vec1': [1, 2, 3],
        'vec2': [4, 5, 6, 7, 8],
        'vec3': [9, 10],
        'vec4': [11, 12, 13, 14]
    }

    with open('sample_input.pkl', 'wb') as f:
        pickle.dump(sample_data, f)



    # Uniformize (pad to max size)
    size, count = uniformize_pickle_entries(
        'sample_input.pkl',
        'sample_output_padded.pkl',
        pad_value=0,
        truncate=False
    )

    print(f"\nResult: {count} entries of size {size}")

    # Load and verify
    with open('sample_output_padded.pkl', 'rb') as f:
        result = pickle.load(f)
    print(f"Sample output: {result}")

    # Example 2: With statistics
    print("\n" + "=" * 50)
    print("Example 2: With detailed statistics")
"""
    size, count, stats = uniformize_pickle_with_stats(
        #'/home/ajanusze/wiki-pir-rag/index.pkl',
        #'/home/ajanusze/wiki-pir-rag/uniform_index.pkl',
        '/Users/antoniajanuszewicz/PycharmProjects/wiki-pir-rag/index.pkl',
        '/Users/antoniajanuszewicz/PycharmProjects/wiki-pir-rag/uniform_index.txt',
        target_size=None,  # Will use max size
        pad_value=0
    )
