import torch
import yaml

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


class ContrastiveDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.transform = get_augmentation()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]

        x1 = self.transform(image)
        x2 = self.transform(image)

        return x1, x2


def get_datasets(config):
    train_base = CIFAR10(
        root=config["data_dir"],
        train=True,
        download=True
    )

    eval_base = CIFAR10(
        root=config["data_dir"],
        train=True,
        download=True,
        transform=get_eval_transform()
    )

    test_base = CIFAR10(
        root=config["data_dir"],
        train=False,
        download=True,
        transform=get_eval_transform()
    )

    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(
        len(train_base), generator=generator
    ).tolist()

    n_train = config["train_size"]
    n_val = config["val_size"]

    if n_train + n_val != len(train_base):
        raise ValueError("train_size + val_size must equal 50000")

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    return {
        "train_ssl": ContrastiveDataset(train_base, train_indices),
        "train_eval": Subset(eval_base, train_indices),
        "val": Subset(eval_base, val_indices),
        "test": test_base
    }


def get_loaders(config):
    datasets = get_datasets(config)
    loaders = {}

    for name, dataset in datasets.items():
        is_train = name == "train_ssl"

        loaders[name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=is_train,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            generator=torch.Generator().manual_seed(config["seed"])
            if is_train else None
        )

    return loaders


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])

    loaders = get_loaders(config)

    for name, loader in loaders.items():
        print(name, len(loader.dataset))

    x1, x2 = next(iter(loaders["train_ssl"]))

    print("View 1:", x1.shape)
    print("View 2:", x2.shape)
    print("Views different:", not torch.equal(x1, x2))
    print("Mean absolute difference:", (x1 - x2).abs().mean().item())
