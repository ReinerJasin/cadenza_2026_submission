DATA_ROOT = Path(r"/content/drive/MyDrive/cadenza_data")

TEST_AUDIO_DIR = DATA_ROOT / "valid" / "signals"
TEST_METADATA_PATH = DATA_ROOT / "metadata" / "valid_metadata.json"

test_dataset = AudioPromptDataset(TEST_METADATA_PATH, TEST_AUDIO_DIR, processor, TARGET_SR)

predictions, references = [], []
for i in range(len(test_dataset)):
    entry = test_dataset.metadata[i]
    audio_path = TEST_AUDIO_DIR / f"{entry['signal']}.flac"
    pred = transcribe_sample(audio_path)
    predictions.append(pred)
    references.append(entry.get("prompt", ""))

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"WER on valid set: {wer * 100:.2f}%")