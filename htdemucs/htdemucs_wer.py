import json
import csv
import re
from jiwer import wer

# -------------------------------
# Helper: normalize text
# -------------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase everything
    text = text.lower()
    # Remove punctuation: keep only letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# File paths
# -------------------------------
CLEAN_FILE = "project/dataset/cadenza_data/metadata/valid_unproc_metadata_augmented.json"     # unprocessed (clean)
NOISY_FILE = "project/dataset/cadenza_data/metadata/valid_signals_metadata_augmented.json"    # noisy version

OUTPUT_FILE = "wer_result.csv"

# -------------------------------
# Load JSON files (no recoding)
# -------------------------------
with open(CLEAN_FILE, "r") as f:   # ← remove encoding="utf-8"
    clean_meta = json.load(f)

with open(NOISY_FILE, "r") as f:
    noisy_meta = json.load(f)

print(f"✅ Loaded {len(clean_meta)} clean samples and {len(noisy_meta)} noisy samples")

# -------------------------------
# Build dictionaries: signal_id → asr_pred
# -------------------------------
clean_dict = {x["signal"]: x["asr_pred"] for x in clean_meta}
noisy_dict = {x["signal"]: x["asr_pred"] for x in noisy_meta}

# -------------------------------
# Compute normalized WER
# -------------------------------
wer_results = []
for signal_id, ref_text in noisy_dict.items():
    if signal_id in clean_dict:
        hyp_text = clean_dict[signal_id]
        # Normalize both texts before computing WER
        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(hyp_text)
        try:
            wer_value = wer(ref_norm, hyp_norm)
        except Exception:
            wer_value = None
        wer_results.append({
            "signal": signal_id,
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "wer": wer_value
        })

# -------------------------------
# Compute average WER
# -------------------------------
valid_wers = [x["wer"] for x in wer_results if x["wer"] is not None]
avg_wer = sum(valid_wers) / len(valid_wers) if valid_wers else None
print(f"📊 Average normalized WER (clean vs noisy): {avg_wer:.4f}")

# -------------------------------
# Save to CSV (still standard UTF-8 output for compatibility)
# -------------------------------
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["signal", "wer", "ref_text", "hyp_text", "ref_norm", "hyp_norm"]
    )
    writer.writeheader()
    writer.writerows(wer_results)

print(f"💾 Results saved to {OUTPUT_FILE}")
print("✅ Done! (Original file encoding kept; all text lowercased & punctuation removed)")

import chardet

file_path = "project/dataset/cadenza_data/metadata/valid_unproc_metadata_augmented.json"     # unprocessed (clean)
# file_path = "valid_unproc_metadata_augmented.json"

with open(file_path, "rb") as f:
    raw = f.read()

result = chardet.detect(raw)
print(result)

import pandas as pd

df = pd.read_csv("wer_result.csv")

print(df.columns)

avg_wer = df["wer"].mean(skipna=True)

print(f"📊 Average WER from wer_result.csv: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
