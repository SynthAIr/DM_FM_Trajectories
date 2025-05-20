from typing import Optional, Tuple
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset




def get_dataloaders(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    test_batch_size: Optional[int],
    num_workers: int = 5,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Splits the dataset into train, validation, and test sets and returns the respective DataLoaders.
    Adapted from https://github.com/SynthAIr/SynTraj
    Parameters
    ----------
    dataset
    train_ratio
    val_ratio
    batch_size
    test_batch_size
    num_workers

    Returns
    -------

    """
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
