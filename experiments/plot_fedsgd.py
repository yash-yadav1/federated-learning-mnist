import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = PROJECT_ROOT / "results" / "fedsgd_iid_lr001.csv"
PLOT_DIR = PROJECT_ROOT / "results" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)

# Convert accuracy from a fraction to a percentage.
df["accuracy_percent"] = df["test_accuracy"] * 100

# Plot accuracy against communication rounds.
plt.figure(figsize=(10, 5))
plt.plot(
    df["round"],
    df["accuracy_percent"],
    marker=".",
    markersize=4,
    label="FedSGD - IID",
)

plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy (%)")
plt.title("FedSGD on IID MNIST: Test Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_DIR / "fedsgd_iid_accuracy.png", dpi=300)
plt.show()

# Plot loss separately.
plt.figure(figsize=(10, 5))
plt.plot(
    df["round"],
    df["test_loss"],
    marker=".",
    markersize=4,
    label="FedSGD - IID",
)

plt.xlabel("Communication Round")
plt.ylabel("Test Loss")
plt.title("FedSGD on IID MNIST: Test Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_DIR / "fedsgd_iid_loss.png", dpi=300)
plt.show()

print(f"Best accuracy: {df['accuracy_percent'].max():.2f}%")
print(f"Best round: {df.loc[df['accuracy_percent'].idxmax(), 'round']}")
print(f"Final accuracy: {df['accuracy_percent'].iloc[-1]:.2f}%")