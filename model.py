import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, inputs, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Sequence-to-sequence output: [batch_size, sequence_length]
        return self.head(out).squeeze(2)
