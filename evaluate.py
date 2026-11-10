import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import get_datasets
from model import SimCLR


@torch.no_grad()
def get_embeddings(encoder, dataset, device, batch_size=128):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda"
    )

    encoder.eval()
    features = []
    labels = []

    for images, targets in tqdm(loader, desc="Embeddings"):
        images = images.to(device)

        h = encoder(images).float()

        if not torch.isfinite(h).all():
            raise RuntimeError("Non-finite embeddings in FP32")

        features.append(h.cpu())
        labels.append(targets.cpu())

    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def evaluate_knn(x_train, y_train, x_eval, y_eval, device, k=20):
    bank = F.normalize(x_train.to(device), dim=1)
    queries = F.normalize(x_eval.to(device), dim=1)
    labels = y_train.to(device)

    correct = 0

    for query, target in zip(
        queries.split(256),
        y_eval.to(device).split(256)
    ):
        similarity = query @ bank.T
        indices = similarity.topk(k, dim=1).indices

        neighbors = labels[indices]
        votes = F.one_hot(neighbors, num_classes=10).sum(dim=1)

        predictions = votes.argmax(dim=1)
        correct += (predictions == target).sum().item()

    return 100.0 * correct / len(y_eval)


def evaluate_linear(x_train, y_train, x_eval, y_eval, device):
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True)
    std = std.clamp_min(1e-5)

    x_eval = ((x_eval - mean) / std).to(device)
    x_train = ((x_train - mean) / std).to(device)

    y_train = y_train.to(device)
    y_eval = y_eval.to(device)

    classifier = nn.Linear(x_train.shape[1], 10).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    for epoch in range(50):
        classifier.train()

        indices = torch.randperm(len(x_train), device=device)

        for batch in indices.split(512):
            logits = classifier(x_train[batch])
            loss = F.cross_entropy(logits, y_train[batch])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) in (1, 10, 20, 30, 40, 50):
            classifier.eval()

            with torch.no_grad():
                train_pred = classifier(x_train).argmax(dim=1)
                eval_pred = classifier(x_eval).argmax(dim=1)

                train_acc = (
                    (train_pred == y_train).float().mean().item() * 100
                )
                eval_acc = (
                    (eval_pred == y_eval).float().mean().item() * 100
                )

            print(
                f"Linear epoch {epoch + 1}: "
                f"train={train_acc:.2f}%, "
                f"val={eval_acc:.2f}%"
            )

    return eval_acc


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        f"checkpoints/{args.name}.pt",
        map_location="cpu",
        weights_only=False
    )

    config = checkpoint["config"]
    torch.manual_seed(config["seed"])

    model = SimCLR(config["projection_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    datasets = get_datasets(config)

    if args.probe_samples > len(datasets["train_eval"]):
        raise ValueError("Too many probe samples")

    train_dataset = Subset(
        datasets["train_eval"],
        range(args.probe_samples)
    )

    eval_dataset = datasets[args.split]

    print("Device:", device)
    print("Checkpoint:", args.name)
    print("Probe samples:", len(train_dataset))
    print("Evaluation samples:", len(eval_dataset))

    x_train, y_train = get_embeddings(
        model.encoder,
        train_dataset,
        device,
        args.batch_size
    )

    x_eval, y_eval = get_embeddings(
        model.encoder,
        eval_dataset,
        device,
        args.batch_size
    )

    knn_acc = evaluate_knn(
        x_train, y_train, x_eval, y_eval, device
    )

    linear_acc = evaluate_linear(
        x_train, y_train, x_eval, y_eval, device
    )

    print(f"kNN Accuracy: {knn_acc:.2f}%")
    print(f"Linear Probe Accuracy: {linear_acc:.2f}%")

    Path("results").mkdir(exist_ok=True)
    output = Path("results") / f"{args.name}_{args.split}.csv"

    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "experiment", "split", "temperature",
            "batch_size", "probe_samples",
            "knn_accuracy", "linear_accuracy"
        ])

        writer.writerow([
            args.name,
            args.split,
            config["temperature"],
            config["batch_size"],
            args.probe_samples,
            round(knn_acc, 4),
            round(linear_acc, 4)
        ])

    print("Results:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="baseline_small")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--probe-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)

    main(parser.parse_args())
