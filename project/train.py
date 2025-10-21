import os
import time
import warnings
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

from dataset.audio_dataset import get_data_loader, get_tokenizer
from model.transcribe_model import TranscribeModel
from hparams import Hparams

# from pathlib import Path

# PROJECT_ROOT = Path(__file__).parent.resolve()

# HYPERPARAMETERS
vq_initial_loss_weight = 2
vq_warmup_steps = 5000
vq_final_loss_weight = 0.5
num_epochs = 1000
starting_steps = 0
num_examples = 100
model_id = 'test1'
num_batch_repeats = 1

# BATCH_SIZE = 64
LEARNING_RATE = 5e-4

# CTC Loss Function
def run_loss_function(log_probs, target, blank_token):
    # Add log_softmax to ensure proper probability distribution
    
    loss_function = nn.CTCLoss(blank=blank_token)
    # input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    input_lengths = [log_probs.size(1)] * log_probs.size(0)

    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first = log_probs.permute(1,0,2)
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    
    return loss 

# Main Training Loop
def main():
    log_dir = f"runs/speech2text_training/{model_id}"
    
    if os.path.exists(log_dir):
        import shutil
        
        shutil.rmtree(log_dir)
        
    writer = SummaryWriter(log_dir)
    
    tokenizer = get_tokenizer(save_path="project/dataset/tokenizer.json")
    
    blank_token = tokenizer.token_to_id("□")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")
    
    if os.path.exists(f"models/{model_id}/model_latest.pth"):
        print(f"Loading model form models/{model_id}/model_latest.pth")
        model = TranscribeModel.load(f"models/{model_id}/model_latest.pth").to(device)
    else:
        model = TranscribeModel(
            num_codebooks=8,
            codebook_size=256,
            embedding_dim=256,
            num_transformer_layers=6,
            vocab_size=len(tokenizer.get_vocab()),
            strides=[6, 6, 6],
            initial_mean_pooling_kernel_size=4,
            max_seq_length=400
        ).to(device)
        
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model is initiated and ready to train!")
    print(f'Number of trainable parameters: {num_trainable_params}\n')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = get_data_loader(split='train', args=Hparams.args)
    
    # ctc_losses = []
    # vq_losses = []
    num_batches = len(train_loader)
    total_steps = num_epochs * num_batches
    steps = starting_steps
    
    # =============================
    # Training Loop
    # made by chatgpt for training step for better log output
    # =============================
    start_time = time.time()

    # Loop by the number of epoch
    for epoch in range(num_epochs):
        
        # List to sroe loss of ctc and vector quantizer for every epoch
        epoch_ctc_losses = []
        epoch_vq_losses = []

        # tqdm progress bar per epoch
        progress_bar = tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Loop by the number of batch in train loader
        for idx, batch in enumerate(progress_bar):
            # Repeat per batch
            for _ in range(num_batch_repeats):
                audio = batch["waveform"]
                target = batch["padded_response"]
                text = [m["response"] for m in batch["metadata"]]

                # Pad audio if the sequence of the padded_response is longer
                if target.shape[1] > audio.shape[1]:
                    audio = torch.nn.functional.pad(
                        audio, (0, 0, 0, target.shape[1] - audio.shape[1])
                    )

                # Use device (cuda or mps)
                audio = audio.to(device)
                target = target.to(device)

                # Reset old gradients
                optimizer.zero_grad()
                
                # Forward Pass
                output, vq_loss = model(audio)
                
                # Calculate CTC loss
                ctc_loss = run_loss_function(output, target, blank_token)

                # Linear warmup schedule for VQ loss weight (total loss = CTC + VQ)
                vq_loss_weight = max(
                    vq_final_loss_weight,
                    vq_initial_loss_weight - (vq_initial_loss_weight - vq_final_loss_weight) * (steps / vq_warmup_steps)
                )

                if vq_loss is None:
                    loss = ctc_loss
                else:
                    loss = ctc_loss + vq_loss_weight * vq_loss

                # Handle invalid loss
                if torch.isinf(loss):
                    print(f"ALERT!!! Loss is inf, skipping step | audio: {audio.shape} | target: {target.shape}")
                    continue

                # Backpropagation
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                # Update parameter weight
                optimizer.step()

                # Update tracking
                steps += 1
                epoch_ctc_losses.append(ctc_loss.item())
                epoch_vq_losses.append(vq_loss.item() if vq_loss is not None else 0.0)

                # Calculate ETA
                elapsed = time.time() - start_time
                steps_done = steps - starting_steps
                eta = elapsed / steps_done * (total_steps - steps_done)
                eta_min = eta / 60

                # Update tqdm progress bar info
                progress_bar.set_postfix({
                    "CTC": f"{ctc_loss.item():.4f}",
                    "VQ": f"{vq_loss.item() if vq_loss is not None else 0:.4f}",
                    "Total": f"{loss.item():.4f}",
                    "ETA(min)": f"{eta_min:.1f}"
                })

                # TensorBoard logging every 20 steps
                if steps % 20 == 0:
                    avg_ctc_loss = sum(epoch_ctc_losses) / len(epoch_ctc_losses)
                    avg_vq_loss = sum(epoch_vq_losses) / len(epoch_vq_losses)
                    avg_loss = avg_ctc_loss + (vq_loss_weight * avg_vq_loss)

                    writer.add_scalar("Loss/CTC", avg_ctc_loss, steps)
                    writer.add_scalar("Loss/VQ", avg_vq_loss, steps)
                    writer.add_scalar("Loss/Total", avg_loss, steps)

                # Occasionally print predictions
                if steps % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred_ids = torch.argmax(output, dim=-1)
                        pred_texts = [tokenizer.decode(ids.tolist()) for ids in pred_ids]
                        true_texts = [t for t in text]

                        print("\n" + "="*70)
                        print(f"Prediction Examples (Step {steps})")
                        for ex_idx in range(min(3, len(pred_texts))):
                            print(f"\nExample {ex_idx + 1}:")
                            print(f"Model Output : {pred_texts[ex_idx]}")
                            print(f"Ground Truth : {true_texts[ex_idx]}")
                        print("="*70 + "\n")

                    model.train()

                # Save checkpoint periodically
                if steps % 250 == 0:
                    os.makedirs(f"models/{model_id}", exist_ok=True)
                    model.save(f"models/{model_id}/model_latest.pth")
                    print(f"Checkpoint saved at step {steps}")

        # Log epoch-level losses
        avg_epoch_ctc = sum(epoch_ctc_losses) / len(epoch_ctc_losses)
        avg_epoch_total = avg_epoch_ctc + vq_loss_weight * (sum(epoch_vq_losses) / len(epoch_vq_losses))
        writer.add_scalar("EpochLoss/CTC", avg_epoch_ctc, epoch + 1)
        writer.add_scalar("EpochLoss/Total", avg_epoch_total, epoch + 1)

        print(f"Epoch {epoch+1}/{num_epochs} complete | Avg CTC Loss: {avg_epoch_ctc:.4f} | Avg Total Loss: {avg_epoch_total:.4f}")

    # Save model after training
    print("Training complete!")
    os.makedirs(f"models/{model_id}", exist_ok=True)
    model.save(f"models/{model_id}/model_final.pth")
    print(f"Final model saved to models/{model_id}/model_final.pth")
    writer.close()
    
if __name__ == "__main__":
    main()