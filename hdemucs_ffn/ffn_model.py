import torch
import torch.nn as nn


class IntelligibilityPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        """
        input_dim = 4 because we pass:
        [0] asr_correctness
        [1] inv_bpm (will go through GELU activation)
        [2] bpm
        [3] stoi
        """
        super().__init__()
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, 4]
        # apply GELU to inv_bpm only (index 1)
        inv_bpm = x[:, 1].unsqueeze(1)          # [B, 1]
        inv_bpm = self.gelu(inv_bpm)            # GELU(1/BPM)

        # rebuild feature vector with activated inv_bpm
        x = torch.cat([x[:, 0:1], inv_bpm, x[:, 2:]], dim=1)  # still [B, 4]

        # normalize all features together
        x = self.bn(x)

        out = self.net(x)  # [B, 1]
        return out.squeeze(1)  # return [B]
