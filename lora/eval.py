from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import numpy as np
import torch
import torchaudio
from pathlib import Path
from .lora import AudioPromptDataset

DATA_ROOT = Path(r"/content/drive/MyDrive/cadenza_data")

TEST_AUDIO_DIR = DATA_ROOT / "valid" / "signals"
TEST_METADATA_PATH = DATA_ROOT / "metadata" / "valid_metadata.json"

test_dataset = AudioPromptDataset(TEST_METADATA_PATH, TEST_AUDIO_DIR, processor, TARGET_SR)

# --- load saved adapter on top of base ---
SAVE_DIR = "./whisper_lora_finetuned"
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, SAVE_DIR)  # LoRA adapters
processor = WhisperProcessor.from_pretrained(SAVE_DIR)

model.to(device)
model.eval()

# (Optional) If you want language/task tokens automatically:
# ids = processor.get_decoder_prompt_ids(language=LANGUAGE.lower(), task=TASK)
# model.generation_config.forced_decoder_ids = ids  # or leave None if you trained that way

def transcribe_sample(audio_path, max_new_tokens=128):
    waveform, sr = torchaudio.load(audio_path)

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    # extract features -> numpy array [80, T]
    feats = processor.feature_extractor(
        waveform.squeeze().numpy(), sampling_rate=TARGET_SR
    ).input_features[0]  # np.ndarray

    # convert to torch tensor with batch dim and move to device
    input_features = torch.from_numpy(np.array(feats)).unsqueeze(0).to(device)

    # fp16 autocast helps on CUDA when model is half / mixed-precision
    with torch.no_grad():
        # If you trained with fp16, you can also do:
        # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
        generated_ids = model.generate(
            input_features=input_features,     # <-- KEY FIX: use keyword input_features
            max_new_tokens=max_new_tokens
        )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

for i in range(min(5, len(test_dataset))):
    sid = test_dataset.metadata[i]["signal"]
    audio_path = TEST_AUDIO_DIR / f"{sid}.flac"
    pred_text = transcribe_sample(audio_path)
    ref_text  = test_dataset.metadata[i].get("prompt", "")
    print(f"🎵 {sid}\nPrediction: {pred_text}\nReference : {ref_text}\n")
