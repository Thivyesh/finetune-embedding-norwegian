"""
Data loading and preprocessing for STS (Semantic Textual Similarity) training.

This module handles:
1. Loading the STS dataset from HuggingFace
2. Formatting it for similarity/regression training
3. Creating train/eval splits

STS Dataset Format:
- sentence1: First sentence
- sentence2: Second sentence
- score: Similarity score (0-5 scale)
"""

from datasets import load_dataset, Dataset
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sts_dataset(
    dataset_name: str = "tollefj/sts-concatenated-NOB",
    train_split: str = "train",
    test_split: str = "test",
    sentence1_column: str = "sentence1",
    sentence2_column: str = "sentence2",
    score_column: str = "score",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    create_dev_split: bool = True,
    dev_split_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Load and prepare STS dataset for embedding fine-tuning.

    STS (Semantic Textual Similarity) datasets provide:
    - Pairs of sentences (sentence1, sentence2)
    - Similarity scores (typically 0-5 scale)
    - Used for regression training (predicting similarity)

    DIFFERENCE FROM NLI (Triplet) DATA:
    - NLI: (anchor, positive, negative) - classification
    - STS: (sentence1, sentence2, score) - regression

    Args:
        dataset_name: HuggingFace dataset identifier
        train_split: Name of training split in the dataset
        test_split: Name of test split
        sentence1_column: Column name for first sentence
        sentence2_column: Column name for second sentence
        score_column: Column name for similarity score
        max_train_samples: Optional limit on training samples
        max_test_samples: Optional limit on test samples
        create_dev_split: Whether to create a dev split from training data
        dev_split_ratio: Ratio of training data to use for dev (if create_dev_split=True)

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
        Note: dev_dataset will be None if create_dev_split=False

    Example:
        >>> train_ds, dev_ds, test_ds = load_sts_dataset(
        ...     "tollefj/sts-concatenated-NOB",
        ...     max_train_samples=100
        ... )
        >>> print(train_ds[0])
        {
            'sentence1': 'Vivendi holdt døren åpen for ytterligere bud...',
            'sentence2': 'Vivendi holdt døren åpen for ytterligere bud i neste dag...',
            'score': 4.0
        }
    """
    logger.info(f"Loading STS dataset: {dataset_name}")

    # Load dataset from HuggingFace
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
    test_dataset = dataset.get(test_split, None)

    logger.info(f"✓ Loaded training split: {len(train_dataset):,} samples")
    if test_dataset:
        logger.info(f"✓ Loaded test split: {len(test_dataset):,} samples")

    # Verify required columns exist
    required_columns = [sentence1_column, sentence2_column, score_column]
    missing_columns = [col for col in required_columns if col not in train_dataset.column_names]

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset.\n"
            f"Available columns: {train_dataset.column_names}\n"
            f"Please check your config file's column names."
        )

    # Create dev split if requested
    dev_dataset = None
    if create_dev_split:
        logger.info(f"Creating dev split ({dev_split_ratio:.1%} of training data)")

        # Split training data
        split_dict = train_dataset.train_test_split(
            test_size=dev_split_ratio,
            seed=42
        )

        train_dataset = split_dict['train']
        dev_dataset = split_dict['test']

        logger.info(f"✓ Training split after dev extraction: {len(train_dataset):,} samples")
        logger.info(f"✓ Created dev split: {len(dev_dataset):,} samples")

    # Limit dataset size if requested
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        logger.info(f"Limiting training samples to {max_train_samples:,}")
        train_dataset = train_dataset.select(range(max_train_samples))

    if test_dataset and max_test_samples is not None and max_test_samples < len(test_dataset):
        logger.info(f"Limiting test samples to {max_test_samples:,}")
        test_dataset = test_dataset.select(range(max_test_samples))

    # Show example for verification
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE STS PAIR FROM DATASET:")
    logger.info("="*70)
    example = train_dataset[0]
    logger.info(f"Sentence 1: {example[sentence1_column]}")
    logger.info(f"Sentence 2: {example[sentence2_column]}")
    logger.info(f"Score:      {example[score_column]} (0-5 scale)")
    logger.info("="*70 + "\n")

    # Show score distribution
    logger.info("Score Distribution (first 1000 samples):")
    sample_size = min(1000, len(train_dataset))
    scores = [train_dataset[i][score_column] for i in range(sample_size)]

    score_ranges = {
        "0-1": sum(1 for s in scores if 0 <= s < 1),
        "1-2": sum(1 for s in scores if 1 <= s < 2),
        "2-3": sum(1 for s in scores if 2 <= s < 3),
        "3-4": sum(1 for s in scores if 3 <= s < 4),
        "4-5": sum(1 for s in scores if 4 <= s <= 5),
    }

    for range_name, count in score_ranges.items():
        pct = (count / sample_size) * 100
        logger.info(f"  {range_name}: {count:4d} ({pct:5.1f}%)")
    logger.info("")

    return train_dataset, dev_dataset, test_dataset


def convert_sts_to_pairs(
    dataset: Dataset,
    sentence1_column: str = "sentence1",
    sentence2_column: str = "sentence2",
) -> Dataset:
    """
    Convert STS dataset to simple sentence pairs format (without scores).

    This is useful if you want to use the STS dataset with MultipleNegativesRankingLoss
    instead of a regression loss. Treats all pairs as positive examples.

    Args:
        dataset: STS dataset with sentence1, sentence2, score columns
        sentence1_column: Name of first sentence column
        sentence2_column: Name of second sentence column

    Returns:
        Dataset with 'anchor' and 'positive' columns (suitable for triplet loss)

    Note:
        This discards the score information and treats all pairs as positives.
        For true STS training with scores, use CosineSimilarityLoss instead.
    """
    logger.info("Converting STS format to sentence pairs (discarding scores)")

    def map_to_pairs(example):
        return {
            'anchor': example[sentence1_column],
            'positive': example[sentence2_column],
        }

    pairs_dataset = dataset.map(
        map_to_pairs,
        remove_columns=dataset.column_names
    )

    logger.info(f"✓ Converted {len(pairs_dataset):,} STS examples to sentence pairs")

    return pairs_dataset


def inspect_sts_dataset(dataset_name: str = "tollefj/sts-concatenated-NOB") -> None:
    """
    Inspect an STS dataset's structure before using it for training.

    This is helpful for understanding:
    - What splits are available
    - What columns exist
    - Score distribution
    - What the data looks like

    Args:
        dataset_name: HuggingFace dataset identifier

    Example:
        >>> inspect_sts_dataset("tollefj/sts-concatenated-NOB")
    """
    logger.info(f"\nInspecting STS dataset: {dataset_name}")
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

    # Show first 3 examples
    logger.info(f"\nFirst 3 examples from '{first_split}':")
    logger.info("-" * 70)
    for i in range(min(3, len(dataset[first_split]))):
        example = dataset[first_split][i]
        logger.info(f"\nExample {i+1}:")
        for key, value in example.items():
            # Truncate long values for readability
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            logger.info(f"  {key}: {value_str}")

    # Score statistics if score column exists
    if 'score' in columns:
        scores = [dataset[first_split][i]['score'] for i in range(min(1000, len(dataset[first_split])))]
        logger.info(f"\nScore Statistics (first {len(scores)} samples):")
        logger.info(f"  Min:    {min(scores):.2f}")
        logger.info(f"  Max:    {max(scores):.2f}")
        logger.info(f"  Mean:   {sum(scores)/len(scores):.2f}")
        logger.info(f"  Median: {sorted(scores)[len(scores)//2]:.2f}")

    logger.info("="*70)


if __name__ == "__main__":
    """
    Test the STS data loader.

    Run this file directly to inspect the Norwegian STS dataset:
        python utils/data_loader_sts.py
    """
    # Inspect the dataset structure
    inspect_sts_dataset("tollefj/sts-concatenated-NOB")

    # Load a small sample
    print("\nLoading sample of dataset...")
    train_ds, dev_ds, test_ds = load_sts_dataset(
        dataset_name="tollefj/sts-concatenated-NOB",
        max_train_samples=100,
        create_dev_split=True
    )

    print(f"\n✓ Successfully loaded {len(train_ds)} training samples")
    if dev_ds:
        print(f"✓ Successfully loaded {len(dev_ds)} dev samples")
    if test_ds:
        print(f"✓ Successfully loaded {len(test_ds)} test samples")

    # Test conversion to pairs
    print("\nTesting conversion to pairs format...")
    pairs_ds = convert_sts_to_pairs(train_ds)
    print(f"✓ Converted to {len(pairs_ds)} sentence pairs")
    print(f"Example pair: {pairs_ds[0]}")
