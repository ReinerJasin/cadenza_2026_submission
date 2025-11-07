# test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import LyricDataset
from ffn_model import IntelligibilityPredictor

# 🔧 Path to your validation or test metadata JSON
VALID_JSON = "/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data/metadata/valid_metadata_augmented.json"

# 🔧 Path to the trained model weights
MODEL_PATH = "/Users/reiner/Documents/GitHub/cadenza_2026_submission/intelligibility_ffn.pt"


def evaluate():
    # 1. Load dataset
    dataset = LyricDataset(VALID_JSON)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelligibilityPredictor(input_dim=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Define loss
    criterion = nn.MSELoss()

    # 4. Evaluation loop
    total_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for features, label in loader:
            features = features.to(device)
            label = label.to(device)

            output = model(features)
            loss = criterion(output, label)

            total_loss += loss.item()
            predictions.extend(output.cpu().tolist())
            labels.extend(label.cpu().tolist())

    avg_loss = total_loss / len(loader)
    rmse = avg_loss ** 0.5

    # 5. Display results
    print(f"✅ Evaluation complete!")
    print(f"📊 Average MSE Loss: {avg_loss:.6f}")
    print(f"📊 Average RMSE Loss: {rmse:.6f}")
    print(f"📈 Example predictions vs labels:")

    for i in range(min(5, len(predictions))):
        print(f"  Pred: {predictions[i]:.4f} | True: {labels[i]:.4f}")


if __name__ == "__main__":
    evaluate()
