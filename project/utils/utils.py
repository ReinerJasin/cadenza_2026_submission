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

# Function to Measure activation Memory
import torch
import gc
import os
import psutil

def bytes_to_mb(x: int) -> float:
    """Convert bytes to megabytes."""
    return round(x / 1024 / 1024, 2)


def check_memory_usage(device: torch.device = None):
    """
    Measure approximate memory usage:
    - GPU memory (CUDA) if available
    - CPU process memory if on MPS or CPU
    - Activation tensor memory (rough estimate)
    """
    gc.collect()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    usage_info = {}

    # --- 1. GPU Memory (CUDA) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        usage_info["gpu_allocated_MB"] = bytes_to_mb(used)
        usage_info["gpu_reserved_MB"] = bytes_to_mb(reserved)

    # --- 2. MPS or CPU Memory (fallback) ---
    else:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        usage_info["cpu_used_MB"] = bytes_to_mb(mem_info.rss)  # Resident memory

        # Approx. activation memory (very rough!)
        total_tensor_mem = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == device.type:
                    total_tensor_mem += obj.nelement() * obj.element_size()
            except:
                pass
        usage_info["tensor_memory_MB"] = bytes_to_mb(total_tensor_mem)

    return usage_info

# utils.py (continue)

def stop_if_memory_high(threshold_percent: float = 90.0):
    """
    Stop the program if memory usage exceeds threshold_percent.
    Works for CPU and MPS. On CUDA, you can use memory_allocated().
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    total_memory = psutil.virtual_memory().total
    used_percent = (mem_info.rss / total_memory) * 100

    if used_percent > threshold_percent:
        raise MemoryError(
            f"🚨 Memory usage too high: {used_percent:.2f}% > {threshold_percent}%! Stopping training..."
        )
