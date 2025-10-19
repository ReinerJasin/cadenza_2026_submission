import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn

torch.autograd.set_detect_anomaly(True)

from dataset.audio_dataset import get_data_loader, get_tokenizer
from model.module.transcribe_model import TranscribeModel

from hparams import Hparams

# from pathlib import Path


# PROJECT_ROOT = Path(__file__).parent.resolve()

vq_initial_loss_weight = 10
vq_warmup_steps = 1000
vq_final_loss_weight = 0.5
num_epochs = 1000
starting_steps = 0
num_examples = 100
model_id = 'test1'
num_batch_repeats = 1

BATCH_SIZE = 64
LEARNING_RATE = 0.005

def run_loss_function(log_probs, target, blank_token):
    # Add log_softmax to ensure proper probability distribution
    
    loss_function = nn.CTCLoss(blank=blank_token)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first = log_probs.permute(1,0,2)
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    
    return loss 

def main():
    log_dir = f"runs/speech2text_training/{model_id}"
    
    if os.path.exists(log_dir):
        import shutil
        
        shutil.rmtree(log_dir)
        
    writer = SummaryWriter(log_dir)
    
    tokenizer = get_tokenizer()
    blank_token = tokenizer.token_to_id("□")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    if os.path.exists(f"models/{model_id}/model_latest.pth"):
        print(f"Loading model form models/{model_id}/model_latest.pth")
        model = TranscribeModel.load(f"models/{model_id}/model_latest.pth").to(device)
    else:
        model = TranscribeModel(
            num_codebooks=2,
            codebook_size=32,
            embedding_dim=16,
            num_transformer_layers=2,
            vocab_size=len(tokenizer.get_vocab()),
            strides=[6, 6, 6],
            initial_mean_pooling_kernel_size=4,
            max_seq_length=400
        ).to(device)
        
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_trainable_params}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = get_data_loader(split='train', args=Hparams.args)
    
    ctc_losses = []
    vq_losses = []
    num_batches = len(train_loader)
    steps = starting_steps
    
    for i in range(num_epochs):
        for idx, batch in enumerate(train_loader):
            for repeat_batch in range (num_batch_repeats):
                audio = batch["waveform"]                           # waveform tensor
                target = batch["padded_response"]                   # target token IDs
                text = [m["response"] for m in batch["metadata"]]   # original text (optional)
                
                if target.shape[1] > audio.shape[1]:
                    print(f'Padding audio, target is longer than audio. Audio shape: {audio.shape}, target shape: {target.shape}')
                    
                    audio = torch.nn.functional.pad(
                        audio, (0, 0, 0, target.shape[1] - audio.shape[1])
                    )
                    
                    print(f'After padding: {audio.shape}')
                    
                audio = audio.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output, vq_loss = model(audio)
                ctc_loss = run_loss_function(output, target, blank_token)
                
                # Calculate vq_loss_weight using linear warmup schedule
                vq_loss_weight = max(
                    vq_final_loss_weight,
                    vq_initial_loss_weight - (vq_initial_loss_weight - vq_final_loss_weight) * (steps / vq_warmup_steps)
                )
                
                if vq_loss is None:
                    loss = ctc_loss
                else:
                    loss = ctc_loss + vq_loss_weight * vq_loss
                    
                if torch.isinf(loss):
                    print(f'Loss is ing, skipping step {audio.shape} {target.shape}')
                    continue
                loss.backward()
                
                # Increase gradient clipping threshold
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0
                )   # Changed form 1.0
                
                optimizer.step()
                
                ctc_losses.append(ctc_loss.item())
                vq_losses.append(vq_loss.item())
                steps += 1
                
                # Log to tensorboard every step
                
                if steps % 20 == 0:
                    avg_ctc_loss = sum(ctc_losses) / len(ctc_losses)
                    avg_vq_loss = sum(vq_losses) / len(vq_losses)
                    
                    avg_loss = avg_ctc_loss + (vq_loss_weight * avg_vq_loss)

                    print(
                        f"[Epoch {i+1}/{num_epochs}] Step {steps}/{num_batches*num_epochs} | "
                        f"CTC Loss: {avg_ctc_loss:.4f} | VQ Loss: {avg_vq_loss:.4f} | Total Loss: {avg_loss:.4f}"
                    )

                    writer.add_scalar("Loss/CTC", avg_ctc_loss, steps)
                    writer.add_scalar("Loss/VQ", avg_vq_loss, steps)
                    writer.add_scalar("Loss/Total", avg_loss, steps)

                    ctc_losses.clear()
                    vq_losses.clear()
                    
                if steps % 250 == 0:
                    os.makedirs(f"models/{model_id}", exist_ok=True)
                    model.save(f"models/{model_id}/model_latest.pth")
                    print(f"Saved checkpoint at step {steps}")

    print("Training complete!")
    os.makedirs(f"models/{model_id}", exist_ok=True)
    model.save(f"models/{model_id}/model_final.pth")
    print(f"Final model saved to models/{model_id}/model_final.pth")
    writer.close()
    
if __name__ == "__main__":
    main()