from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio
import torch

# Path where you saved it
MODEL_PATH = "./whisper_lora_finetuned"

# Load model and processor
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_PATH = "./sample.flac"   # change this to your test file
waveform, sr = torchaudio.load(AUDIO_PATH)

# convert to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# resample if needed
TARGET_SR = 16000
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

input_features = processor.feature_extractor(
    waveform.squeeze().numpy(),
    sampling_rate=TARGET_SR,
    return_tensors="pt"
).input_features.to(model.device)

predicted_ids = model.generate(input_features)
transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("🔊 Predicted transcription:\n", transcription)

import evaluate
wer = evaluate.load("wer")

reference = "your ground truth text here"
error = 100 * wer.compute(predictions=[transcription], references=[reference])
print(f"WER: {error:.2f}%")
