
import torch

from models.cnn import CNN

model = CNN()

# Simulate a batch of 64 MNIST images.
images = torch.randn(64, 1, 28, 28)

outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Number of model parameters:", sum(
    p.numel() for p in model.parameters()
))

print("CNN is working correctly!")
