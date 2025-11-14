# HTDemucs Signal Isolation Pipeline

## Pipeline Diagram
<img src="htdemucs.png" style="background-color:white;padding-top:5px">

## Overview
Our approach is inspired by the original Cadenza Challenge pipeline, where Hybrid Transformers Demucs (HTDemucs) was used to estimate the ground truth signals in the dataset. This technique then converts the method from an intrusive approach where we can do direct comparison between the processed signals and unprocessed signals, into a non-intrusive approach, by only passing the vocal estimated feature which technically mask the ground truth.

During the development of our older approach, the HDemucs Feed Forward Neural Network. We take on similar approach in isolating the vocal part of the unprocessed signals. While manually inspecting the output of the isolated vocal track, we found that HTDemucs does not just isolating the vocal, but it also amplifies them, resulting in a much clearer audio and louder vocal even if the input audio is applied with mild or moderate noise.

This discovery inspires us to take on a new approach, which is to feed the Whisper-small model by OpenAI and use the transcription result from both isolated vocals to be evaluated by the Word Error Rate (WER) metric.

## Why is HTDemucs important?
The cadenza dataset contains audio of musics form different styles, which are applied with random noises and additional effects to obscure the audio to reduce the intelligibility level.

HTDemucs, which is an attention-based version of the Hybrid Demucs (HDemucs) is very helpful in isolating different frequencies from the audio source. By specifying the model with parameter, we can extract only part and remove the other components of the audio, including harmonic or percussive musical component. The result is a refined vocal track with the following key observations:
1. Vocal Clarity improves due to harmonic separation.
2. Speech amplitudes increases which make the lyrics more audible.
3. Effectiveness on noisy audio allows us to develop this modular approach.

We believe that this capabiliy can work well when combined with OpenAI's Whisper model that is capable on identifying speech but often degrades when there are music interference.

## Pipeline Summary
Our HTDemucs Signal Isolation pipeline follow these key steps:
1. Isolate vocals using HYDemucs
2. Transcribe isolated vocals with Whisper-small
3. Compute correctness using WER
4. Save the result for manual inspection 

## Running the HTDemucs Vocal Isolation
**(Optional)** If you want to create your own transcription result in the metadata folder by running the following script in the terminal. Make sure that you activate the correct virtual environment.

```python htdemucs/preprocess_build_features.py```

You can run the HTDemucs inference script to compute corrrectness based on our previous result or your custom transcription, just make sure that these files exist in your project:
* `project/dataset/cadenza_data/metadata/valid_signals_metadata_augmented.json`
* `project/dataset/cadenza_data/metadata/valid_unproc_metadata_augmented.json`

Then you can run this command in the terminal:
```python htdemucs/htdemucs_wer.py```

## HTDemucs Inference Examples
To give you a better understanding about how our pipeline performs. we provide the following inference examples, which includes:
* Playable noisy vs HTDemucs separated vocals
* Playable unprocessed vs HTDemucs separated vocals
* Side-by-side waverform, spectrogram, and MFCC comparison
* Whisper's transcription result
* Correctness score

### 🟩 Noise Level: No Loss
Noisy Signals — Before & After HTDemucs

Audio Name: 1d887341bdf775c49d8a0c30

Unprocessed Signals — Before & After HTDemucs

Audio Name: 1d887341bdf775c49d8a0c30_unproc

Ground Truth

Lyrics: “you can hold my hand”

HT Demucs result (noisy audio)

Whisper Prediction: “You can know my hand.”

Correctness: 0.8

HT Demucs result (unprocessed audio)

Whisper Prediction: “You can know my hand”

Correctness: 0.8

### 🟨 Noise Level: Mild
Noisy Signals — Before & After HTDemucs

Audio Name: 1d887341bdf775c49d8a0c30

Unprocessed Signals — Before & After HTDemucs

Audio Name: 1d887341bdf775c49d8a0c30_unproc

Ground Truth

Lyrics: “you can hold my hand”

HT Demucs result (noisy audio)

Whisper Prediction: “You can know my hand.”

Correctness: 0.8

HT Demucs result (unprocessed audio)

Whisper Prediction: “You can know my hand”

Correctness: 0.8

### 🟧 Noise Level: Moderate
Noisy Signals — Before & After HTDemucs

Audio Name: 1d887341bdf775c49d8a0c30

Unprocessed Signals — Before & After HTDemucs

Audio Name: 1d887341bdf775c49d8a0c30_unproc

Ground Truth

Lyrics: “you can hold my hand”

HT Demucs result (noisy audio)

Whisper Prediction: “You can know my hand.”

Correctness: 0.8

HT Demucs result (unprocessed audio)

Whisper Prediction: “You can know my hand”

Correctness: 0.8