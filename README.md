# cadenza_2026_submission

- Important Links:
  - Project Directory (Google Drive): https://drive.google.com/drive/folders/1yylkF_uyWTKFt6AFnc_Cb0u58buqg5Ko?usp=sharing

  - Dataset (Google Drive): https://drive.google.com/drive/folders/11-28gBufhvrfl5w5Mlc65ItEDo_wWe7j?usp=drive_link

  - Cadenza Website: https://cadenzachallenge.org/docs/clip1/intro

## How to Run:

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
# If the repo ships requirements.txt
pip install -r requirements.txt
```

### 3. Prepare the dataset (Locally)

Find the data_root path in the `hparams.py` and just replace it with your relative path to the dataset.

### 4. Adjust hyperparameters for training

Go to `project/utils/hparams.py` and check every hyperparameter and adjust it to your need.

> Please pay attention to the model_id. you can start with something like `test1`. If you use the same id, the train on that model will continue. so to start again from scratch, make sure to change the model_id every time you want to train. <br /><br />
There is a small issue, if the code already make checkpoint of the model in the first epoch, and we stop the training process and continue it by re-running the code. It will skip the first epoch and start from the second epoch directly.

### 5. Run the Code to train
```
python project/train.py
```

Can also use full path of train.py