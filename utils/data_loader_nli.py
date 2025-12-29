"""
Data loading and preprocessing for embedding model training.

This module handles:
1. Loading the dataset from HuggingFace
2. Formatting it correctly for triplet training
3. Creating train/eval/test splits
"""

from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_triplet_dataset(
    dataset_name: str,
    train_split: str = "train",
    eval_split: str = "dev",
    test_split: str = "test",
    anchor_column: str = "anchor",
    positive_column: str = "positive",
    negative_column: str = "negative",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare triplet dataset for embedding training.

    WHAT THIS FUNCTION DOES:
    1. Downloads the dataset from HuggingFace (cached after first download)
    2. Selects the correct splits (train/dev/test)
    3. Optionally limits the dataset size for faster experimentation
    4. Ensures columns are in the correct order for the loss function

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "Fremtind/all-nli-norwegian")
        train_split: Name of training split in the dataset
        eval_split: Name of evaluation/validation split
        test_split: Name of test split
        anchor_column: Name of column containing anchor sentences
        positive_column: Name of column containing positive (similar) sentences
        negative_column: Name of column containing negative (dissimilar) sentences
        max_train_samples: Optional limit on training samples (for quick testing)
        max_eval_samples: Optional limit on eval samples (for quick testing)

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset)

    Example:
        >>> train_ds, eval_ds, test_ds = load_triplet_dataset(
        ...     "Fremtind/all-nli-norwegian",
        ...     max_train_samples=1000  # Use only 1000 samples for testing
        ... )
        >>> print(train_ds[0])
        {
            'anchor': 'Hundene leker i snøen.',
            'positive': 'Tre hunder leker med et leketøy i snøen.',
            'negative': 'Det er veldig varmt.'
        }
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset from HuggingFace
    # This downloads the data on first run, then uses cache
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}'. "
            f"Please check that the dataset name is correct and you have internet access.\n"
            f"Error: {e}"
        )

    # Extract splits
    if train_split not in dataset:
        raise ValueError(
            f"Train split '{train_split}' not found in dataset. "
            f"Available splits: {list(dataset.keys())}"
        )

    train_dataset = dataset[train_split]
    eval_dataset = dataset.get(eval_split, None)
    test_dataset = dataset.get(test_split, None)

    logger.info(f"✓ Loaded training split: {len(train_dataset):,} samples")
    if eval_dataset:
        logger.info(f"✓ Loaded evaluation split: {len(eval_dataset):,} samples")
    if test_dataset:
        logger.info(f"✓ Loaded test split: {len(test_dataset):,} samples")

    # Verify required columns exist
    required_columns = [anchor_column, positive_column, negative_column]
    missing_columns = [col for col in required_columns if col not in train_dataset.column_names]

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset.\n"
            f"Available columns: {train_dataset.column_names}\n"
            f"Please check your config file's column names."
        )

    # Limit dataset size if requested (useful for quick testing)
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        logger.info(f"Limiting training samples to {max_train_samples:,}")
        train_dataset = train_dataset.select(range(max_train_samples))

    if eval_dataset and max_eval_samples is not None and max_eval_samples < len(eval_dataset):
        logger.info(f"Limiting evaluation samples to {max_eval_samples:,}")
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Show example for verification
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE TRIPLET FROM DATASET:")
    logger.info("="*70)
    example = train_dataset[0]
    logger.info(f"Anchor:   {example[anchor_column]}")
    logger.info(f"Positive: {example[positive_column]}")
    logger.info(f"Negative: {example[negative_column]}")
    logger.info("="*70 + "\n")

    return train_dataset, eval_dataset, test_dataset


def inspect_dataset(dataset_name: str) -> None:
    """
    Inspect a dataset's structure before using it for training.

    This is helpful for understanding:
    - What splits are available
    - What columns exist
    - What the data looks like

    Args:
        dataset_name: HuggingFace dataset identifier

    Example:
        >>> inspect_dataset("Fremtind/all-nli-norwegian")
    """
    logger.info(f"\nInspecting dataset: {dataset_name}")
    logger.info("="*70)

    dataset = load_dataset(dataset_name)

    # Show splits
    logger.info(f"\nAvailable splits: {list(dataset.keys())}")

    # Show split sizes
    for split_name, split_data in dataset.items():
        logger.info(f"  {split_name}: {len(split_data):,} samples")

    # Show columns
    first_split = list(dataset.keys())[0]
    columns = dataset[first_split].column_names
    logger.info(f"\nColumns in '{first_split}' split: {columns}")

    # Show first example
    logger.info(f"\nFirst example from '{first_split}':")
    logger.info("-" * 70)
    example = dataset[first_split][0]
    for key, value in example.items():
        # Truncate long values for readability
        value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        logger.info(f"  {key}: {value_str}")

    logger.info("="*70)


if __name__ == "__main__":
    """
    Test the data loader.

    Run this file directly to inspect the Norwegian NLI dataset:
        python utils/data_loader.py
    """
    # Inspect the dataset structure
    inspect_dataset("Fremtind/all-nli-norwegian")

    # Load a small sample
    print("\nLoading sample of dataset...")
    train_ds, eval_ds, test_ds = load_triplet_dataset(
        dataset_name="Fremtind/all-nli-norwegian",
        max_train_samples=100,
        max_eval_samples=50
    )

    print(f"\n✓ Successfully loaded {len(train_ds)} training samples")
    print(f"✓ Successfully loaded {len(eval_ds)} evaluation samples")
    print(f"✓ Successfully loaded {len(test_ds)} test samples")
