import os
import sys
import torch
from pathlib import Path

class Hparams:
    PROJECT_ROOT = Path(__file__).parent.resolve()
    
    data_loader_args = {
        'data_root': f"{PROJECT_ROOT}/dataset/cadenza_data",
        # 'save_model_dir': './results/lr1e-3',
        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # 'dataset_root': './data_mini/',
        # 'sampling_rate': 16000, # Please keep the sampling rate unchanged
        # 'sample_length': 5,  # in second
        'num_workers': 0,  # Number of additional thread for data loading. When running on laptop, set to 0.
        # 'annotation_path': './data_mini/annotations.json',      # Previously Missing and causing an error

        # 'frame_size': 0.02,
        'batch_size': 2,
    }
    
    vq_args = {
        'vq_initial_loss_weight': 2,
        'vq_warmup_steps': 5000,
        'vq_final_loss_weight': 0.5,
    }
    
    train_args = {
        'num_epochs': 1000,
        'starting_steps': 0,
        'num_examples': 100,
        'model_id': 'test2',
        'num_batch_repeats': 1
    }
    
    optimizer_args = {
        'learning_rate': 5e-4,
    }