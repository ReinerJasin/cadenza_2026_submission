# LoRA Fine-Tuning on Whisper-Small

Low-Rank Adaptation (LoRA) is an adaptation method in finetuning deep learning models that is achieved by freezing the pre-trained weight of the model and injecting trainable rank decomposition matrixes in each layer of the transformer architecture to reduce the number of parameters to be finetuned later on [https://arxiv.org/pdf/2106.09685]. LoRA was originally intended for a natural language processing scenario, where it was able to reduce the number of trainable parameters by 10,000x and the GPU memory requirement by 3x.

## Whisper Data Pre-Processing Steps

Whisper requires us to implement the data pre-processing to ensure the input format is consistent. As our data is not is whisper's expected format, we have to do extra steps in pre-processing the data. The steps can be seen in the following points:
1. Load Audio.
2. Convert from Stereo to Mono.
3. Resample from 44.1 kHz to 16 kHz.
4. Extract log-Mel spectrograms. (done using `WhisperFeatureExtractor`)
5. Tokenize text (done using `WhisperTokenizer`).
6. Dynamic Padding per batch.

After these steps are done, then the dataset is ready for fine tuning.

## LoRA Configuration
We adjust the `LoraConfig` with the following parameters:
<table>
    <tr>
        <td>
            <b>Parameter</b>
        </td>
        <td>
            <b>Value</b>
        </td>
    </tr>
    <tr>
        <td>rank (r)</td>
        <td>32</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>64</td>
    </tr>
    <tr>
        <td>target modules</td>
        <td>q_proj, v_proj, fc1, fc2</td>
    </tr>
    <tr>
        <td>dropout</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>lora_only</td>
    </tr>
    <tr>
        <td>task_type</td>
        <td>SEQ_2_SEQ_LM</td>
    </tr>
</table>

## How to Fine Tune with LoRA
To run the fine tune script by yourself, head to `new_lora/lora_fine_tune.ipynb` and just follow along the code.

Overview content:
1. Load Whisper-small
2. Prepare model for LoRA
3. Attach LoRA adapters
4. Create dataset and dataloaders
5. Train using Seq2SeqTrainer with the following parameter:
    * batch_size = 4
    * learning rate = 1e-3
    * warmup steps = 50
    * FP16 enabled
6. Save outputs of:
    * training logs and checkpoints to `project/whisper_lora_exp4_results/`
    * final model + processor to `project/whisper_lora_exp4_finetuned/`
7. Load model for inference
8. Calculates Word Error Rate (WER)

## Why LoRA?
Initially in our development, we implement the LoRA adaptation method to finetune a speech-to-text model created by OpenAI named Whisper. As of now, the Whisper model has multiple variants with different model sizes and speed. We decided to go with the Whisper-small variant, due to its relatively faster speed and more compact memory requirements. Whisper model follows the regular encoder-decoder model, but is adapted for speech recognition and transcription purposes. The Whisper-small model consists of 12 layers in each encoder and decoder part. In the hidden size part, the dimension of the embeddings and hidden vectors are 768. During the attention mechanism, it splits each attention layer into 12 attention heads. **In total, the Whisper-small model has ≈244 million trainable parameters.**

Specifically for the LoRA layer, we utilize the parameter-efficient fine-tuning (PEFT) approach and its library for easier fine tuning. In our configuration of the LoRA layer, we use a rank size of 16, a value that we think is a sweet spot to balance the capacity and precision of the embedding. We use alpha of 32, which is twice the size of the rank so that we have enough magnitude to update the weight with less risk of gradient explosion. Our LoRA is applied only on the Q and V projections because these are the parts of the attention mechanism that control what information we should pay attention more into in a specific context. For the dropout layer, we incorporate 0.05 value which is considered a small amount  for large pretrained models to allow generalization on the result and prevent overfitting. We also disable the bias because we do not want extra parameters that can increase the complexity. We also specify the task_type to “SEQ_2_SEQ_LM” to make sure that the Whisper-small model uses the weight for the right task.

LoRA mechanism can be seen as an adapter that will add value to the final embeddings to convert its value instead of having to finetune the whole parameters which is computationally more expensive. Fine tuning the Whisper-small model requires us to update the weight of the ≈244 million trainable parameters that we have. While implementing the LoRA allows us to freeze the pretrained weights and only learn parameters from the following calculations:

$total\;module = LoRA\;module + self\;attention\;LoRA\;module + cross;attention;LoRA;module$

$total\;params = 2 \cdot num\;of\;embedding \cdot LoRA\;Rank \cdot layer \cdot total\;module$

By this calculation, **we only need to update the weights of 1,769,472 trainable parameters**. This number is **138.23x less than the original parameter of the Whisper-small model**.

## Inference Result Example
When we run the code, the final result will be the prediction of the whisper model.

<audio controls>
    <source src="../project/dataset/cadenza_data/valid/signals/6fd06b828c785b77e19dfdcb.flac" type="audio/flac">
</audio>

[Listen to the audio](../project/dataset/cadenza_data/valid/signals/6fd06b828c785b77e19dfdcb.flac)

* Audio name: 6fd06b828c785b77e19dfdcb
    * Noise level: No Loss
    * Inference Speed: 1.00 sec
    * Ground truth: head down all shivering as if to yell
    * Prediction: hate it as shiver and six t...
    * WER: 1.000


<audio controls>
    <source src="../project/dataset/cadenza_data/valid/signals/5965000d1139d388ebcded80.flac" type="audio/flac">
</audio>

[Listen to the audio](../project/dataset/cadenza_data/valid/signals/5965000d1139d388ebcded80.flac)

* Audio name: 5965000d1139d388ebcded80
    * Noise level: Mild
    * Inference Speed: 1.40 sec
    * Ground truth: we had time to spend together
    * Prediction: we are had time to spend together
    * WER: 0.167

<audio controls>
    <source src="../project/dataset/cadenza_data/valid/signals/795996ed006725199e5efc0b.flac" type="audio/flac">
</audio>

[Listen to the audio](../project/dataset/cadenza_data/valid/signals/795996ed006725199e5efc0b.flac)

* Audio name: 795996ed006725199e5efc0b
    * Noise level: Moderate
    * Inference Speed: 0.45 sec
    * Ground truth: I had to walk before I made
    * Prediction: i have to go be on me for all may
    * WER: 

## Performance Metrics
Our LoRA implementation able to reach WER on the valid set at **69.00%**

