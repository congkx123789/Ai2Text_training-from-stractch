"""
Simple Vietnamese ASR Demo
A basic demonstration of the ASR system functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

print("=" * 50)
print("Vietnamese ASR System Demo")
print("=" * 50)

# Step 1: Check environment
print("\n1. Environment Check:")
print(f"   Python version: {torch.__version__}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Current directory: {os.getcwd()}")

# Step 2: Create simple ASR model
print("\n2. Creating ASR Model:")

class SimpleASR(nn.Module):
    """Simple ASR model for demonstration."""
    
    def __init__(self, input_dim=80, vocab_size=100, hidden_dim=256):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch, time, features)
        out, _ = self.encoder(x)
        out = self.dropout(out)
        return self.decoder(out)

# Initialize model
model = SimpleASR(input_dim=80, vocab_size=100, hidden_dim=256)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Model created with {total_params:,} parameters")

# Step 3: Simulate training
print("\n3. Training Simulation:")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(5):
    # Simulate batch of audio features and targets
    batch_size = 4
    seq_len = 50
    
    # Dummy audio features (mel spectrograms)
    audio_features = torch.randn(batch_size, seq_len, 80)
    
    # Dummy target transcripts (character indices)
    targets = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(audio_features)  # (batch, seq_len, vocab_size)
    
    # Compute loss
    loss = criterion(outputs.reshape(-1, 100), targets.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   Epoch {epoch+1}/5: Loss = {loss.item():.4f}")

# Step 4: Test inference
print("\n4. Inference Test:")

model.eval()
with torch.no_grad():
    # Single audio sample
    test_audio = torch.randn(1, 30, 80)  # 1 sample, 30 time steps, 80 features
    
    # Get predictions
    predictions = model(test_audio)
    predicted_chars = torch.argmax(predictions, dim=-1)
    
    print(f"   Input shape: {test_audio.shape}")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Predicted character indices: {predicted_chars[0][:10].tolist()}...")

# Step 5: Create sample data structure
print("\n5. Creating Sample Data Structure:")

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create sample CSV
sample_csv = """file_path,transcript,split,speaker_id
data/raw/audio1.wav,xin chao viet nam,train,speaker_01
data/raw/audio2.wav,toi la sinh vien,train,speaker_02
data/raw/audio3.wav,hom nay troi dep,val,speaker_01
data/raw/audio4.wav,chung toi hoc tieng viet,test,speaker_02"""

with open("data/sample_data.csv", "w", encoding="utf-8") as f:
    f.write(sample_csv)

print("   Sample data CSV created: data/sample_data.csv")

# Step 6: Save model checkpoint
print("\n6. Saving Model Checkpoint:")

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 5,
    'loss': loss.item(),
    'model_config': {
        'input_dim': 80,
        'vocab_size': 100,
        'hidden_dim': 256
    }
}

torch.save(checkpoint, "checkpoints/demo_model.pt")
print("   Model saved to: checkpoints/demo_model.pt")

# Step 7: Summary
print("\n" + "=" * 50)
print("Demo Completed Successfully!")
print("=" * 50)
print("\nWhat was demonstrated:")
print("- Basic ASR model architecture (LSTM + Linear)")
print("- Training loop with dummy data")
print("- Inference/prediction")
print("- Model checkpointing")
print("- Directory structure creation")

print("\nNext steps for real usage:")
print("1. Replace dummy data with real audio files and transcripts")
print("2. Implement proper audio preprocessing (mel spectrograms)")
print("3. Use Vietnamese text tokenization")
print("4. Add CTC loss for sequence alignment")
print("5. Implement proper evaluation metrics (WER, CER)")

print("\nFiles created:")
print("- data/sample_data.csv (sample dataset)")
print("- checkpoints/demo_model.pt (model checkpoint)")
print("- Directory structure for full project")

print("\nYour Vietnamese ASR system is ready for development!")


