import json
import torch
import numpy as np

def tensor_to_serializable(obj):
    """Recursively convert torch.Tensor or numpy array to lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return obj


def dump_json(data, max_list_len=20, indent=4):
    """
    Hybrid pretty printer:
    - Uses json.dumps() for normal fields
    - Compresses large lists/tensors into single lines like `[1, 2, ..., 99]`
    """
    data = tensor_to_serializable(data)
    compact_data = {}

    for key, value in data.items():
        if isinstance(value, list) and len(value) > max_list_len:
            # compress long lists
            head = ", ".join(map(str, value[:max_list_len // 2]))
            tail = ", ".join(map(str, value[-max_list_len // 2:]))
            compact_data[key] = f"[{head}, ..., {tail}] (len={len(value)})"
        else:
            compact_data[key] = value

    # Now safely pretty-print everything else
    print(json.dumps(compact_data, indent=indent))
