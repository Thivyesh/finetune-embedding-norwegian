"""
ETI Triplet Data Loader

Loads Norwegian health and welfare information triplet dataset from:
https://huggingface.co/datasets/thivy/eti-embedding-training-data-2048-triplets

Dataset structure:
- ~330K triplet samples in Norwegian
- Columns: anchor, positive, negative
- (anchor, positive, negative) triplets with hard negatives
- Longer passages (up to 2048 tokens)

This triplet format enables more effective contrastive learning with explicit hard negatives.
"""

from datasets import load_dataset, Dataset
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eti_triplet_data(
    dataset_name: str = "thivy/eti-embedding-training-data-2048-triplets",
    split_ratio: Tuple[float, float, float] = (0.98, 0.01, 0.01),
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ETI triplet dataset for contrastive learning.

    The ETI triplet dataset contains Norwegian (anchor, positive, negative) triplets
    from official health and welfare information sources. Hard negatives enable
    better contrastive learning compared to in-batch negatives alone.

    Args:
        dataset_name: HuggingFace dataset identifier
        split_ratio: Train/dev/test split ratios (default: 98/1/1)

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
        Each sample: {'anchor': str, 'positive': str, 'negative': str}
    """
    logger.info(f"Loading ETI triplet dataset from {dataset_name}...")

    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

    if "train" not in dataset:
        raise ValueError(f"Dataset {dataset_name} does not have a 'train' split")

    train_data = dataset["train"]
    logger.info(f"✓ Loaded {len(train_data):,} triplet samples")

    # Verify triplet format
    sample = train_data[0]
    required_columns = ['anchor', 'positive', 'negative']
    for col in required_columns:
        if col not in sample:
            raise ValueError(f"Missing required column '{col}' in dataset")
    
    logger.info(f"✓ Verified triplet format: {required_columns}")

    # Manual split into train/dev/test
    total_size = len(train_data)
    train_size = int(total_size * split_ratio[0])
    dev_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - dev_size

    # Shuffle with fixed seed for reproducibility
    shuffled_data = train_data.shuffle(seed=42)
    
    train_dataset = shuffled_data.select(range(train_size))
    dev_dataset = shuffled_data.select(range(train_size, train_size + dev_size))
    test_dataset = shuffled_data.select(range(train_size + dev_size, total_size))

    logger.info(f"✓ ETI triplet split manually: {len(train_dataset):,} train, {len(dev_dataset):,} dev, {len(test_dataset):,} test")

    return train_dataset, dev_dataset, test_dataset


def load_eti_triplet_data_for_training():
    """
    Convenience function to load ETI triplet data with standard splits.
    
    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    return load_eti_triplet_data()


if __name__ == "__main__":
    # Test the data loader
    train, dev, test = load_eti_triplet_data()
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train):,}")
    print(f"  Dev: {len(dev):,}")
    print(f"  Test: {len(test):,}")
    
    print(f"\nSample triplet:")
    sample = train[0]
    print(f"  Anchor: {sample['anchor'][:100]}...")
    print(f"  Positive: {sample['positive'][:100]}...")
    print(f"  Negative: {sample['negative'][:100]}...")