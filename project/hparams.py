import os
import sys
import torch
from pathlib import Path

class Hparams:
    PROJECT_ROOT = Path(__file__).parent.resolve()
    
    args = {
        'data_root': f"{PROJECT_ROOT}/dataset/cadenza_data",
        # 'save_model_dir': './results/lr1e-3',
        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # 'dataset_root': './data_mini/',
        # 'sampling_rate': 16000, # Please keep the sampling rate unchanged
        # 'sample_length': 5,  # in second
        'num_workers': 0,  # Number of additional thread for data loading. When running on laptop, set to 0.
        # 'annotation_path': './data_mini/annotations.json',      # Previously Missing and causing an error

        # 'frame_size': 0.02,
        'batch_size': 8,  
    }