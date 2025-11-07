# train.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset_loader import LyricDataset
from ffn_model import IntelligibilityPredictor

AUG_JSON = "/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data/metadata/train_metadata_augmented.json"
NUM_EPOCH = 500

def main():
    dataset = LyricDataset(AUG_JSON, start_idx=0, end_idx=10)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = IntelligibilityPredictor(input_dim=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(NUM_EPOCH):
        model.train()
        total_loss = 0.0

        for features, label in loader:
            features = features.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(features)           # [B]
            loss = criterion(output, label)    # both [B]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.5f}")

    torch.save(model.state_dict(), "intelligibility_ffn.pt")
    print("✅ model saved to intelligibility_ffn.pt")


if __name__ == "__main__":
    main()
