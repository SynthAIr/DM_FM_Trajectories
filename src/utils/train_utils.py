from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats._distn_infrastructure import rv_continuous
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset


def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_config(config, config_file):
    with open(config_file, "w") as f:
        yaml.dump(config, f)


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    print_dict(item, indent + 1)
                else:
                    print("  " * (indent + 1) + str(item))
        else:
            print("  " * indent + f"{key}: {value}")


def get_dataloaders(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    test_batch_size: Optional[int],
    num_workers: int = 5,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    val_size = int(train_size * val_ratio)
    train_size -= val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    if val_size > 0:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=(
                test_batch_size if test_batch_size is not None else len(val_dataset)
            ),
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = None

    if test_size > 0:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=(
                test_batch_size if test_batch_size is not None else len(val_dataset)
            ),
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def init_hidden(
    x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True
):
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(
            x.device
        )
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(x.device)


def build_weights(size: int, builder: rv_continuous, **kwargs) -> np.ndarray:
    w = np.array([builder.pdf(i / (size + 1), **kwargs) for i in range(1, size + 1)])
    return w