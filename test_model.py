
import torch
import torch.nn as nn

# Simple test model
class SimpleASR(nn.Module):
    def __init__(self, input_dim=80, vocab_size=100, hidden_dim=256):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        out, _ = self.encoder(x)
        return self.decoder(out)

# Test the model
model = SimpleASR()
dummy_input = torch.randn(2, 100, 80)  # (batch, time, features)
output = model(dummy_input)
print(f" Model test successful! Output shape: {output.shape}")
print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
