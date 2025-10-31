import torch

# Load the checkpoint
checkpoint = torch.load('checkpoints/demo_model.pt')

print("Checkpoint loaded successfully!")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Loss: {checkpoint['loss']:.4f}")
print(f"Model config: {checkpoint['model_config']}")
print("\nCheckpoint keys:", list(checkpoint.keys()))


