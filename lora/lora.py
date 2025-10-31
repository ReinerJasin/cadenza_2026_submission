
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from dataclasses import dataclass
from typing import Any

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import evaluate

# ============================================================
# 1. Configuration
# ============================================================
# Repo root = parent of this file's directory (since lora.py lives in ./lora/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow override via env var if you want
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "project" / "dataset" / "cadenza_data"))

TRAIN_AUDIO_DIR = DATA_ROOT / "train" / "signals"
VALID_AUDIO_DIR = DATA_ROOT / "valid" / "signals"
TRAIN_METADATA_PATH = DATA_ROOT / "metadata" / "train_metadata.json"
VALID_METADATA_PATH = DATA_ROOT / "metadata" / "valid_metadata.json"  # optional

MODEL_NAME = "openai/whisper-small"  # or "openai/whisper-large-v2"
LANGUAGE = "English"
TASK = "transcribe"

BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
TARGET_SR = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
    
# Print device here
print("🔍 CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🧠 GPU device:", torch.cuda.get_device_name(0))
    print("🔥 Current device index:", torch.cuda.current_device())
else:
    print("⚠️ Using CPU — training will be very slow")

# ============================================================
# 2. Dataset Definition
# ============================================================
class AudioPromptDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, processor, target_sr=16000):
        with open(metadata_path, "r") as f:
            data = json.load(f)
        # handle nested list like [[{...}]]
        self.metadata = data[0] if isinstance(data, list) and isinstance(data[0], list) else data
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        signal_id = entry["signal"]
        prompt = entry.get("prompt") or entry.get("original_prompt", "")

        audio_path = self.audio_dir / f"{signal_id}.flac"
        waveform, sr = torchaudio.load(audio_path)

        # convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample to target_sr
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # extract input features
        input_features = self.processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=self.target_sr
        ).input_features[0]

        # tokenize the text
        labels = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)

        return {"input_features": input_features, "labels": labels}


# ============================================================
# 3. Collator Function
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # remove duplicate BOS if exists
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================================
# 4. Load Processor and Dataset
# ============================================================
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

train_dataset = AudioPromptDataset(TRAIN_METADATA_PATH, TRAIN_AUDIO_DIR, processor, TARGET_SR)
valid_dataset = (
    AudioPromptDataset(VALID_METADATA_PATH, VALID_AUDIO_DIR, processor, TARGET_SR)
    if VALID_METADATA_PATH.exists()
    else None
)
collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# ============================================================
# 5. Load Whisper Model (16-bit) and Prepare for LoRA
# ============================================================
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    # torch_dtype=torch.float8_e4m3fn if torch.cuda.is_available() else torch.float8_e4m3fnuz,
    dtype=torch.float16,
    device_map="auto"
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model = prepare_model_for_kbit_training(model)

# LoRA setup
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)

print("Trainable parameters:")
model.print_trainable_parameters()

# ============================================================
# 6. Evaluation Metric (WER)
# ============================================================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ============================================================
# 7. Trainer Setup
# ============================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_lora_results",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch" if valid_dataset else "no",
    save_strategy="epoch",
    logging_steps=10,
    logging_dir="./logs",
    fp16=True,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="none",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset if valid_dataset else None,
    data_collator=collator,
    tokenizer=processor,
    compute_metrics=compute_metrics if valid_dataset else None,
)

model.config.use_cache = False  # disable for training

# ============================================================
# 8. Train and Save
# ============================================================
from types import MethodType

def safe_forward(self, *args, **kwargs):
    # Drop 'input_ids' if it appears (Trainer sometimes injects it)
    kwargs.pop("input_ids", None)
    return self.base_model.forward(*args, **kwargs)

# Attach to PEFT model
model.forward = MethodType(safe_forward, model)

trainer.train()

SAVE_DIR = "./whisper_lora_finetuned"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"✅ Training complete! Model saved to: {SAVE_DIR}")