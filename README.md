# [Submission] ICASSP 2026 Cadenza Challenge: Predicting Lyric Intelligibility
This repository is made for our submission to the ICASSP 2026 Cadenza Challenge.

For more details about the challenge, please visit [cadenza challenge 2026 website](https://cadenzachallenge.org/docs/clip1/intro).

## Team Members

1. Reiner Anggriawan Jasin
    * Student ID: A0314502W
    * Contact: E1503344@u.nus.edu

2. Ram Gopalakrishnan
    * Student ID: A0314499R
    * Contact: -

3. Qiao Jiayi
    * Student ID: A0332228L
    * Contact: E1583084@u.nus.edu

4. Ye Guoquan
    * Student ID: A0188947A
    * Contact: -

## Environment Setup Step

### 1. Create a virtual environment
```
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\venv\Scripts\Activate
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Download Dataset for training / fine tuning (Optional)
1. Visit this google drive: https://drive.google.com/drive/folders/11-28gBufhvrfl5w5Mlc65ItEDo_wWe7j?usp=sharing
2. Download the dataset.
3. Extract the zip file.
3. Move to this directory: `/cadenza_2026_submission/project/dataset/cadenza_data`
4. Your audio dataset will be accessible as `../project/dataset/cadenza_data/train/signals/example_audio.flac` (adjust the sub-folder for validation and metadata)

## 📂 Sub-Projects

This repository contains multiple modules. Please do the `environment setup` before running each sub-projects.

Click any folder below to open it and view its own README:

* [data_analysis_and_sastoi](data_analysis_and_sastoi/) : Exploratory data analysis of the data set.
* [LoRA (new)](new_lora/) : Most updated implementation of the LoRA fine tuning method.
* [HT Demucs](htdemucs/) : Implementation of Vocal Signal Isolation technique.
* [Smart Model Picker](Whisper_M4T_Smart_Picker/) : Implementation of Smart Whisper/M4T model picker.

Previous implementation:
* [LoRA (old)](lora/) : Previous implementation on lora. It is recommended to use the LoRA (new) code.
* [HT Demucs + FFN](hdemucs_ffn/) : A feedforward neural network approach to predict lyric intelligibility.
* [Self-attention Speech-to-text from Scratch](project/speech-to-text.ipynb) : Implementation of speech-to-text model from sratch using Whisper Encoder + Self-attention + RVQ form scratch.
* [Custom Hearing Loss Simulator](hearing_loss_sim/) : Attempt on simulating hearing_loss for dataset augmentation purposes.

## USE of LLM/AI Tools
Large Language Model (LLM) tools were used selectively during this project to support technical tasks, but not to generate research ideas or claim originality for concepts that were not developed by us. The core methodology, system design, and experimental approach were conceived, implemented, and validated by the authors.

**Tools used:**
ChatGPT (developed by OpenAI)

**Purpose of use:** <br/>
LLM assistance was used strictly for the following supportive tasks:

* Debugging specific error messages in Python, PyTorch, and dataset path configurations.
* Improving clarity of code segments by rewriting helper functions, refactoring logic, or generating minimal working examples (e.g., DataLoader, collate functions, evaluation snippets).
* Formatting sections of the report, simplifying wording, and proofreading grammar.
* Providing quick references for library functions or API usage when documentation was unclear.

**What was not done with AI tools:**

* AI tools were not used to generate research ideas, algorithmic innovations, system architecture decisions, or any conceptual contribution.
* AI tools were not used to draft or invent the methodology presented in the paper. <br/>
All techniques, model adaptations, and experimental strategies were designed by the authors.

**Verification of originality and correctness:** <br/>
All AI-assisted outputs—whether code, explanations, or text—were manually checked, tested, or rewritten before inclusion.

Specifically:
* Code suggestions were validated by running them locally, checking correctness, and ensuring compatibility with our dataset and pipeline.
* Written content was reviewed for accuracy and adjusted to match our actual implementation.
* No AI-generated material was used without verification; inaccurate or irrelevant outputs were removed or corrected.

Thus, while LLMs supported the workflow as an auxiliary tool for debugging and writing clarity, the intellectual contribution, system design, and experimentation remain the authors’ own original work.