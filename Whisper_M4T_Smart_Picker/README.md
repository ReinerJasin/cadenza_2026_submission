# Smart Model-Picker

Compare OpenAI's Whisper and Meta's Seamless M4T v2 models on audio transcription, with a neural network that predicts which model performs better based on audio features.

## Overview

This project compares two state-of-the-art speech recognition models on the Cadenza dataset and trains a neural network to predict the better-performing model for each audio file.

**Key Features:**
- Dual model transcription (Whisper + Seamless M4T v2)
- Audio feature extraction (BPM, ZCR, Spectral Centroid, Spectral Rolloff)
- Neural network model selector
- Performance analysis by hearing loss severity
- Persistent JSON caching for incremental processing
- Batch processing with configurable file ranges

## Important: Data Configuration

**You Must Update the Data Paths Before Running!**

The notebook expects the following data structure with **FLAC audio files**:

```
your_data_directory/
├── metadata/
│   └── train_metadata.json
└── train/
    └── signals/
        ├── audio_file_1.flac
        ├── audio_file_2.flac
        └── ...
```



```python
# CHANGE THESE PATHS TO YOUR DATA LOCATION
METADATA_PATH = Path("/content/drive/MyDrive/Cadenza/cadenza_data/metadata/train_metadata.json")
AUDIO_DIR_PRIMARY = Path("/content/drive/MyDrive/Cadenza/cadenza_data/train/signals")
RESULTS_FILE = "/content/drive/MyDrive/Cadenza/transcription_results_cache.json"

# Configure file range to process
START_FILE_INDEX = 0      # Start from this file
END_FILE_INDEX = 1000     # Process up to this file (exclusive)
```

## Requirements

```bash
pip install openai-whisper transformers>=4.35.0 librosa soundfile
pip install scikit-learn torch torchvision torchaudio accelerate sentencepiece
```

**Hardware:**
- GPU with CUDA (recommended) or CPU with 8GB+ RAM

##  Quick Start

1. **Mount Drive (if using Colab):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Update Data Paths:** Edit Cell 3 configuration (see above)

3. **Run All Cells:** Execute sequentially from Cell 1 to Cell 8

## Models

- **Whisper**: OpenAI's encoder-decoder transformer (configurable: tiny/base/small/medium/large)
- **Seamless M4T v2**: Meta's multilingual translation model (facebook/seamless-m4t-v2-large)

## Neural Network

Predicts which model performs better using 4 audio features:
- Input → 64 neurons (ReLU) → Dropout → 32 neurons (ReLU) → Dropout → 2 classes (Softmax)

Results are cached in JSON format by signal_id. Previously processed files are automatically skipped.

## Outputs

- Transcription results cache (JSON)
- Performance comparisons by hearing loss level
- Confusion matrix for model predictions
- Training curves (loss/accuracy)
- Trained neural network model
