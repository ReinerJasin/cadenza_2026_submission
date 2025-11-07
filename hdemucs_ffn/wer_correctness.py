from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import torch
import torchaudio
import re

# Load Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

wer_metric = evaluate.load("wer")

def transcribe(audio, sr):
    
    # Convert to mono if needed
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr
        # print(f"Resampled from {sr} Hz to 16000 Hz as expected by whisper model")
        
    # Convert to numpy for whisper processor
    audio_np = audio.squeeze().numpy()
        
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        pred_ids = model.generate(**inputs)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def compute_correctness(pred, ref):
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    
    # Normalized before counting Word Error Rate
    wer = wer_metric.compute(predictions=[pred_norm], references=[ref_norm])
    # wer = wer_metric.compute(predictions=[pred], references=[ref])
    
    correctness = max(0.0, 1 - wer)
    return correctness


if __name__ == "__main__":
    # Example test
    
    # Load audio with noises
    AUDIO_PATH = "/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data/train/signals/bdb1b2a6bdea4bcf6d03e11f.flac"
    REF_TEXT = "Just come away and let it"
    # REF_TEXT = "#"

    # Load audio
    audio, sr = torchaudio.load(AUDIO_PATH)
    print(f"Loaded audio ({audio.shape[0]} channel(s), {sr} Hz)")

    # Transcribe the noisy audio
    print("Transcribing the noisy audio...")
    prediction = transcribe(audio, sr)
    print(f"Predicted: {prediction}")

    # Check correctness compared to ground truth (actual lyrics)
    correctness = compute_correctness(prediction, REF_TEXT)
    print(f"Correctness: {correctness:.2%}")