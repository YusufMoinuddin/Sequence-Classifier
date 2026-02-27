# models/cnn_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqCNN(nn.Module):
    """
    Simple 1D CNN for 8-mer one-hot encoded DNA.

    Input:  x of shape (batch_size, 4, seq_len)
    Output: logits of shape (batch_size,)
    """
    def __init__(self, in_channels: int = 4, hidden_channels: int = 16, seq_len: int = 8):
        super().__init__()
        # use padding=1 (kernel=3) instead of "same" for broader PyTorch compat
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_channels, 1)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, L)
        returns logits: (B,)
        """
        h = F.relu(self.conv1(x))          # (B,16,L)
        h = F.relu(self.conv2(h))          # (B,16,L)
        # Global max pooling over sequence length
        h = F.max_pool1d(h, kernel_size=h.shape[-1]).squeeze(-1)  # (B,16)
        logits = self.fc(h).squeeze(-1)    # (B,)
        return logits
