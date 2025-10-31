
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(" Starting ASR Training Demo...")

# Simple model
class DemoASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(80, 128, batch_first=True)
        self.decoder = nn.Linear(128, 50)  # 50 character vocab
    
    def forward(self, x):
        out, _ = self.encoder(x)
        return self.decoder(out)

# Initialize model and optimizer
model = DemoASR()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Simulate training for a few steps
for epoch in range(3):
    # Dummy data
    audio_features = torch.randn(4, 50, 80)  # (batch, time, features)
    targets = torch.randint(0, 50, (4, 50))  # (batch, time)
    
    # Forward pass
    outputs = model(audio_features)
    loss = criterion(outputs.reshape(-1, 50), targets.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/3, Loss: {loss.item():.4f}")

print(" Training demo completed successfully!")
print(" This was just a demo. For real training, you'll need actual audio data.")
