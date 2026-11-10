import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_result(name):
    path = Path("results") / f"{name}_val.csv"

    with open(path, newline="") as file:
        row = next(csv.DictReader(file))

    return (
        float(row["knn_accuracy"]),
        float(row["linear_accuracy"])
    )


def plot_accuracy(experiments, xlabel, filename):
    x = []
    knn = []
    linear = []

    for value, name in experiments:
        a, b = read_result(name)
        x.append(value)
        knn.append(a)
        linear.append(b)

    plt.figure(figsize=(6, 4))
    plt.plot(x, knn, "o-", label="kNN")
    plt.plot(x, linear, "s-", label="Linear probe")

    plt.xlabel(xlabel)
    plt.ylabel("Validation accuracy (%)")
    plt.xticks(x)
    plt.ylim(0, 60)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = Path("figures") / filename
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"\n{xlabel}")
    for i, value in enumerate(x):
        print(
            f"{value}: kNN={knn[i]:.2f}%, "
            f"Linear={linear[i]:.2f}%"
        )
    print("Saved:", path)


Path("figures").mkdir(exist_ok=True)

plot_accuracy(
    [
        (0.1, "temp_01_fp32"),
        (0.2, "baseline_fp32"),
        (0.5, "temp_05_fp32")
    ],
    "Temperature",
    "accuracy_vs_temperature.png"
)

plot_accuracy(
    [
        (16, "batch_16_fp32"),
        (32, "batch_32_fp32"),
        (64, "baseline_fp32")
    ],
    "Batch size",
    "accuracy_vs_batch_size.png"
)

base = read_result("baseline_fp32")
stress = read_result("false_negatives_fp32")

print("\nFalse-negative stress test")
print(f"kNN difference: {stress[0] - base[0]:+.2f} pp")
print(f"Linear difference: {stress[1] - base[1]:+.2f} pp")
