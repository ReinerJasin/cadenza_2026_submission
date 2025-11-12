import json
from pathlib import Path
import torchaudio
import torch
from tqdm import tqdm

from hdemucs import separate_vocals
from wer_correctness import compute_correctness
from extract_bpm import compute_bpm
from stoi import compute_stoi
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Paths
DATA_ROOT = Path("/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data")
SIGNAL_DIR = DATA_ROOT / "valid" / "signals"
VOCAL_CACHE = DATA_ROOT / "valid" / "vocals_cache"  # add this folder to cache separated vocals
INPUT_JSON = DATA_ROOT / "metadata" / "valid_metadata.json"
OUTPUT_JSON = DATA_ROOT / "metadata" / "valid_signals_metadata_augmented.json"

START_IDX = 0
END_IDX = 1000

VOCAL_CACHE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)

# Whisper loads once
print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.eval()


def transcribe_with_whisper(audio, sr):
    """Transcribe using preloaded Whisper"""
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    target_sr = 16000
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
        sr = target_sr
    audio_np = audio.squeeze().numpy()
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt").to(model.device)
    with torch.no_grad():
        pred_ids = model.generate(**inputs)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


def main():
    with open(INPUT_JSON, "r") as f:
        meta = json.load(f)
        
    meta = meta[START_IDX:END_IDX]
    print(f"Using subset of {len(meta)} entries (index 0–1000)")

    augmented = []
    for row in tqdm(meta, desc="Processing clips", unit="clip"):
        signal_id = row["signal"]
        audio_path = SIGNAL_DIR / f"{signal_id}.flac"
        vocal_path = VOCAL_CACHE / f"{signal_id}_vocals.wav"

        if not audio_path.exists():
            print(f"⚠️ Skipping {signal_id} — file not found")
            continue

        try:
            # Load audio
            audio, sr = torchaudio.load(str(audio_path))
            if audio.ndim > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Vocal separation (with caching)
            if vocal_path.exists():
                vocals, sr_v = torchaudio.load(vocal_path)
            else:
                vocals, sr_v = separate_vocals(str(audio_path))
                torchaudio.save(vocal_path, vocals, sr_v)

            # Transcribe vocals
            pred_text = transcribe_with_whisper(vocals, sr_v)
            ref_text = row.get("prompt", "")
            asr_correctness = compute_correctness(pred_text, ref_text)

            # BPM
            bpm = float(compute_bpm(audio, sr))
            if bpm <= 0:
                bpm = 1.0
            inv_bpm = 1.0 / bpm

            # STOI (truncate to 10 sec for speed)
            max_len = min(audio.shape[1], sr * 10)
            stoi_score = float(compute_stoi(audio[:, :max_len], vocals[:, :max_len], sr))

            # Label
            label = float(row.get("fixed_correctness", row.get("correctness", 0.0)))

            # Append
            augmented.append(
                {
                    "signal": signal_id,
                    "audio_path": str(audio_path),
                    "prompt": row.get("prompt", ""),
                    "asr_pred": pred_text,
                    "asr_correctness": asr_correctness,
                    "bpm": bpm,
                    "inv_bpm": inv_bpm,
                    "stoi": stoi_score,
                    "label": label,
                    "hearing_loss": row.get("hearing_loss", None),
                    "raw_correctness_metadata": row.get("correctness", None),
                }
            )

        except Exception as e:
            print(f"Error on {signal_id}: {e}")
            continue

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"\nSaved {len(augmented)} processed entries to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()



## OLD CODE

# # preprocess_build_features.py
# import json
# from pathlib import Path

# import torchaudio
# import torch

# from tqdm import tqdm

# from hdemucs import separate_vocals
# from wer_correctness import transcribe, compute_correctness
# from extract_bpm import compute_bpm
# from stoi import compute_stoi

# # adjust these to your repo layout
# DATA_ROOT = Path("/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data")
# SIGNAL_DIR = DATA_ROOT / "train" / "unprocessed"    # .../train/unprocessed/<signal>_unproc.flac
# INPUT_JSON = DATA_ROOT / "metadata" / "train_metadata.json"
# OUTPUT_JSON = DATA_ROOT / "metadata" / "train_metadata_augmented.json"


# def main():
#     with open(INPUT_JSON, "r") as f:
#         meta = json.load(f)

#     augmented = []
#     for i, row in enumerate(tqdm(meta, desc="Processing clips", unit="clip")):
#         signal_id = row["signal"]
#         audio_path = SIGNAL_DIR / f"{signal_id}_unproc.flac"

#         # 1. load audio
#         audio, sr = torchaudio.load(str(audio_path))

#         # 2. separate vocals
#         vocals, _ = separate_vocals(str(audio_path))

#         # 3. transcribe separated vocals (could also use original audio)
#         pred_text = transcribe(vocals, sr)  # we pass vocals here
#         ref_text = row.get("prompt", "")    # your metadata has "prompt"
#         asr_correctness = compute_correctness(pred_text, ref_text)

#         # 4. bpm from original audio
#         bpm = float(compute_bpm(audio, sr))
#         # safeguard
#         if bpm <= 0:
#             bpm = 1.0
#         inv_bpm = 1.0 / bpm

#         # 5. stoi between original and vocals
#         stoi_score = float(compute_stoi(audio, vocals, sr))

#         # 6. label from metadata
#         # if you later "manually" fix correctness, just add "fixed_correctness" to json
#         label = float(row.get("fixed_correctness", row.get("correctness", 0.0)))

#         augmented.append(
#             {
#                 "signal": signal_id,
#                 "audio_path": str(audio_path),
#                 "prompt": row.get("prompt", ""),
#                 "asr_pred": pred_text,
#                 "asr_correctness": asr_correctness,
#                 "bpm": bpm,
#                 "inv_bpm": inv_bpm,
#                 "stoi": stoi_score,
#                 "label": label,
#                 # keep originals so you don't lose info
#                 "hearing_loss": row.get("hearing_loss", None),
#                 "raw_correctness_metadata": row.get("correctness", None),
#             }
#         )
#         print(f"[{i+1}/{len(meta)}] processed {signal_id}")

#     with open(OUTPUT_JSON, "w") as f:
#         json.dump(augmented, f, indent=2)

#     print(f"saved augmented metadata to {OUTPUT_JSON}")


# if __name__ == "__main__":
#     main()
