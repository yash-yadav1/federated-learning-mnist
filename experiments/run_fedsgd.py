import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

#Make the project root importable when running this file directly.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import CNN
from algorithms.fedsgd import fedsgd_round


# ---------------- Configuration ----------------

SEED = 42
NUM_CLIENTS = 10
NUM_ROUNDS = 150

LEARNING_RATE = 0.04
BATCH_SIZE = 1024
EVALUATION_BATCH_SIZE = 1024

DEVICE = torch.device("cpu")

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "fedsgd_iid_lr001.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def create_iid_clients(train_dataset, num_clients):
    """
    Randomly distribute training indices among clients.

    Each client receives approximately the same number of examples.
    """
    indices = np.random.permutation(len(train_dataset))
    client_indices = np.array_split(indices, num_clients)

    client_datasets = [
        Subset(train_dataset, indices.tolist())
        for indices in client_indices
    ]

    return client_datasets

def evaluate(model, test_loader, device):
    """Evaluate the global model on the MNIST test dataset."""
    model.eval()

    loss_function = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            total_loss += loss_function(outputs, labels).item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return accuracy, average_loss

def main():
    set_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    # Load the already-downloaded MNIST training data.

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    train_dataset = Subset(train_dataset, list(range(12000)))
    # Load the common test set.
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
    )

    # Divide the training data among ten IID clients.
    client_datasets = create_iid_clients(
        train_dataset,
        NUM_CLIENTS,
    )

    print("Dataset loaded successfully.")
    print("Training examples:", len(train_dataset))
    print("Test examples:", len(test_dataset))
    print("Number of clients:", NUM_CLIENTS)

    for client_id, dataset in enumerate(client_datasets, start=1):
        print(f"Client {client_id}: {len(dataset)} examples")

    # Initialize the global model.
    global_model = CNN().to(DEVICE)

    # Start writing the experimental results.
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "round",
            "algorithm",
            "distribution",
            "test_accuracy",
            "test_loss",
        ])

        # Evaluate the initial model before training.
        accuracy, loss = evaluate(
            global_model,
            test_loader,
            DEVICE,
        )

        writer.writerow([
            0,
            "FedSGD",
            "IID",
            accuracy,
            loss,
        ])

        print(
            f"Round 0 | Accuracy: {accuracy:.4f} | "
            f"Loss: {loss:.4f}"
        )

        # Execute communication rounds.
        for round_number in range(1, NUM_ROUNDS + 1):
            print(f"\nStarting round {round_number}")

            global_model = fedsgd_round(
                global_model=global_model,
                client_datasets=client_datasets,
                learning_rate=LEARNING_RATE,
                device=str(DEVICE),
            )

            accuracy, loss = evaluate(
                global_model,
                test_loader,
                DEVICE,
            )

            writer.writerow([
                round_number,
                "FedSGD",
                "IID",
                accuracy,
                loss,
            ])

            file.flush()

            print(
                f"Round {round_number} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Loss: {loss:.4f}"
            )

    print("\nExperiment completed.")
    print("Results saved to:", RESULTS_FILE)

if __name__ == "__main__":
    main()