# inference_example.py
import torch
from ffn_model import IntelligibilityPredictor

def load_model(path="intelligibility_ffn.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelligibilityPredictor(input_dim=4)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_intelligibility(model, device, asr_corr, bpm, stoi_score):
    # recompute inv_bpm
    if bpm <= 0:
        bpm = 1.0
    inv_bpm = 1.0 / bpm

    feat = torch.tensor([[asr_corr, inv_bpm, bpm, stoi_score]], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(feat)  # [1]
    return float(pred.item())

if __name__ == "__main__":
    model, device = load_model()
    score = predict_intelligibility(model, device,
                                    asr_corr=0.82,
                                    bpm=120.0,
                                    stoi_score=0.75)
    print(f"Predicted intelligibility: {score:.4f}")
