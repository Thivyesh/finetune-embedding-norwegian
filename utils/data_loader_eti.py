"""
ETI (Elektronisk Tjenesteinformasjon) Data Loader

Loads Norwegian health and welfare information Q&A dataset from:
https://huggingface.co/datasets/thivy/eti-embedding-training-data-2048

Dataset structure:
- ~78.9K samples in Norwegian
- Columns: anchor, positive
- (anchor, positive) pairs — no hard negatives
- Longer passages (up to 2048 tokens)

This is a high-quality Norwegian dataset from official health information sources.
"""

from datasets import load_dataset, Dataset
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eti_data(
    dataset_name: str = "thivy/eti-embedding-training-data-2048",
    split_ratio: Tuple[float, float, float] = (0.98, 0.01, 0.01),
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ETI (Elektronisk Tjenesteinformasjon) dataset.

    The ETI dataset contains Norwegian Q&A pairs from official health and welfare
    information sources. It's used for training dense embeddings that
    can retrieve relevant health information given a query.

    The dataset has a single 'train' split with (anchor, positive) pairs.
    We split it manually into train/dev/test.

    Args:
        dataset_name: HuggingFace dataset identifier
        split_ratio: Train/dev/test split ratios (default: 98/1/1)

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
        Each sample: {'anchor': str, 'positive': str}
    """
    logger.info(f"Loading ETI dataset from {dataset_name}...")

    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load ETI dataset '{dataset_name}'. "
            f"Please check that the dataset name is correct and you have internet access.\n"
            f"Error: {e}"
        )

    # Verify required columns exist
    sample_split = dataset[list(dataset.keys())[0]]
    required_columns = ['anchor', 'positive']
    missing_columns = [col for col in required_columns if col not in sample_split.column_names]

    if missing_columns:
        raise ValueError(
            f"ETI dataset missing required columns: {missing_columns}. "
            f"Available columns: {sample_split.column_names}"
        )

    # Use existing splits if available, otherwise split manually
    if 'train' in dataset and 'dev' in dataset and 'test' in dataset:
        train_dataset = dataset['train']
        dev_dataset = dataset['dev']
        test_dataset = dataset['test']
        logger.info(
            f"✓ ETI loaded with existing splits: {len(train_dataset):,} train, "
            f"{len(dev_dataset):,} dev, {len(test_dataset):,} test"
        )
    elif 'train' in dataset:
        # Fallback: split the train set manually
        full = dataset['train']
        total = len(full)
        train_size = int(total * split_ratio[0])
        dev_size = int(total * split_ratio[1])

        train_dataset = full.select(range(train_size))
        dev_dataset = full.select(range(train_size, train_size + dev_size))
        test_dataset = full.select(range(train_size + dev_size, total))
        logger.info(
            f"✓ ETI split manually: {len(train_dataset):,} train, "
            f"{len(dev_dataset):,} dev, {len(test_dataset):,} test"
        )
    else:
        raise ValueError(
            f"ETI dataset has unexpected splits: {list(dataset.keys())}. "
            f"Expected at least a 'train' split."
        )

    return train_dataset, dev_dataset, test_dataset
