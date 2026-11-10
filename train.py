import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import get_datasets
from model import SimCLR
from loss import info_nce


def train(config, name, max_steps=None):
    seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    dataset = get_datasets(config)["train_ssl"]

    if "ssl_samples" in config:
        dataset = Subset(
            dataset,
            range(config["ssl_samples"])
        )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=config["num_workers"] > 0,
        generator=torch.Generator().manual_seed(seed)
    )

    model = SimCLR(config["projection_dim"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )


    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    history_path = Path("results") / f"{name}.csv"
    checkpoint_path = Path("checkpoints") / f"{name}.pt"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Experiment:", name)
    print("Temperature:", config["temperature"])
    print("Batch size:", config["batch_size"])
    print("Epochs:", config["epochs"])

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "steps", "seconds"])

        for epoch in range(config["epochs"]):
            model.train()

            total_loss = 0.0
            steps = 0
            start = time.perf_counter()

            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{config['epochs']}"
            )

            for x1, x2 in progress:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                pairs = config.get("false_negative_pairs", 0)

                if pairs:
                    if pairs < 0 or 2 * pairs > x1.size(0):
                        raise ValueError("Invalid false-negative pairs")

                    x1[-pairs:] = (
                        x1[:pairs].clone()
                        + 0.01 * torch.randn_like(x1[-pairs:])
                    )

                    x2[-pairs:] = (
                        x2[:pairs].clone()
                        + 0.01 * torch.randn_like(x2[-pairs:])
                    )

                images = torch.cat([x1, x2], dim=0)

                _, z = model(images)

                z1, z2 = z.chunk(2, dim=0)

                loss = info_nce(
                    z1,
                    z2,
                    config["temperature"]
                )

                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss")

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

                progress.set_postfix(loss=f"{loss.item():.4f}")

                if max_steps is not None and steps >= max_steps:
                    break

            mean_loss = total_loss / steps
            seconds = time.perf_counter() - start

            writer.writerow([
                epoch + 1,
                mean_loss,
                steps,
                round(seconds, 2)
            ])
            file.flush()

            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "config": config
            }, checkpoint_path)

            print(
                f"Epoch {epoch + 1}: "
                f"loss={mean_loss:.4f}, "
                f"time={seconds:.1f}s"
            )

    print("History:", history_path)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--false-negative-pairs", type=int, default=0)

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.epochs is not None:
        config["epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.temperature is not None:
        config["temperature"] = args.temperature

    config["false_negative_pairs"] = args.false_negative_pairs

    train(config, args.name, args.max_steps)
