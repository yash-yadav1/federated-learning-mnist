from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

# Download MNIST training data.
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download MNIST test data.
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("MNIST downloaded successfully!")
print("Training examples:", len(train_dataset))
print("Test examples:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First 10 labels:", labels[:10].tolist())
