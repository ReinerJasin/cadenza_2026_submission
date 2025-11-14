
### 4. Adjust hyperparameters for training (for `project` directory)

Go to `project/utils/hparams.py` and check every hyperparameter and adjust it to your need.

> Please pay attention to the model_id. you can start with something like `test1`. If you use the same id, the train on that model will continue. so to start again from scratch, make sure to change the model_id every time you want to train. <br /><br />
There is a small issue, if the code already make checkpoint of the model in the first epoch, and we stop the training process and continue it by re-running the code. It will skip the first epoch and start from the second epoch directly.

### 5. Run the Code to train
```
python project/train.py
```

Can also use full path of train.py