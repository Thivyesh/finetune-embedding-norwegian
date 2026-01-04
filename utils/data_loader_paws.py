"""
Data loader for Norwegian PAWS-X (Paraphrase Adversaries from Word Scrambling).

This module loads the Norwegian PAWS-X dataset for paraphrase identification training.
PAWS-X contains pairs of sentences with binary labels indicating whether they are
paraphrases of each other.

Dataset: NbAiLab/norwegian-paws-x
Format: JSONL with sentence1, sentence2, label (0/1)
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Optional
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_paws_dataset(
    data_dir: str = "data/paws-x/x-final",
    language: str = "nb",  # 'nb' for Bokmål, 'nn' for Nynorsk
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load Norwegian PAWS-X dataset for paraphrase identification.

    Args:
        data_dir: Directory containing the extracted PAWS-X files
        language: 'nb' for Bokmål or 'nn' for Nynorsk
        max_samples: Optional limit on training samples

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    data_path = Path(data_dir) / language

    if not data_path.exists():
        raise FileNotFoundError(
            f"PAWS-X data not found at {data_path}. "
            f"Run download script first or check data_dir path."
        )

    # File paths
    train_file = data_path / "translated_train.json"
    dev_file = data_path / "translated_dev_2k.json"
    test_file = data_path / "translated_test_2k.json"

    # Load JSONL files
    def load_jsonl(file_path: Path) -> list:
        """Load JSONL file (one JSON object per line)."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    logger.info(f"Loading Norwegian PAWS-X ({language.upper()})...")

    # Load splits
    train_data = load_jsonl(train_file)
    dev_data = load_jsonl(dev_file)
    test_data = load_jsonl(test_file)

    # Apply max_samples limit if specified
    if max_samples and max_samples < len(train_data):
        logger.info(f"Limiting training data to {max_samples:,} samples")
        train_data = train_data[:max_samples]

    logger.info(f"✓ Loaded PAWS-X dataset:")
    logger.info(f"  Language: {language.upper()} ({'Bokmål' if language == 'nb' else 'Nynorsk'})")
    logger.info(f"  Train: {len(train_data):,} pairs")
    logger.info(f"  Dev: {len(dev_data):,} pairs")
    logger.info(f"  Test: {len(test_data):,} pairs")

    # Calculate statistics
    train_positive = sum(1 for x in train_data if x['label'] == 1)
    train_negative = len(train_data) - train_positive
    logger.info(f"  Train distribution: {train_positive:,} paraphrases, {train_negative:,} non-paraphrases")
    logger.info(f"  Balance: {train_positive/len(train_data)*100:.1f}% positive")

    # Convert to HuggingFace Dataset format
    def to_dataset(data: list) -> Dataset:
        """Convert list of dicts to HuggingFace Dataset."""
        return Dataset.from_dict({
            'sentence1': [x['sentence1'] for x in data],
            'sentence2': [x['sentence2'] for x in data],
            'label': [x['label'] for x in data],
        })

    train_dataset = to_dataset(train_data)
    dev_dataset = to_dataset(dev_data)
    test_dataset = to_dataset(test_data)

    # Show sample
    logger.info(f"\n📝 Sample paraphrase pair:")
    sample_idx = next(i for i, x in enumerate(train_data) if x['label'] == 1)
    sample = train_data[sample_idx]
    logger.info(f"  S1: {sample['sentence1'][:80]}...")
    logger.info(f"  S2: {sample['sentence2'][:80]}...")
    logger.info(f"  Label: {sample['label']} (paraphrase)")

    logger.info(f"\n📝 Sample non-paraphrase pair:")
    sample_idx = next(i for i, x in enumerate(train_data) if x['label'] == 0)
    sample = train_data[sample_idx]
    logger.info(f"  S1: {sample['sentence1'][:80]}...")
    logger.info(f"  S2: {sample['sentence2'][:80]}...")
    logger.info(f"  Label: {sample['label']} (not paraphrase)\n")

    return train_dataset, dev_dataset, test_dataset


def download_paws_data():
    """
    Download and extract Norwegian PAWS-X dataset.

    This function downloads the dataset from HuggingFace Hub and extracts it
    to the data/paws-x directory.
    """
    import tarfile
    from huggingface_hub import hf_hub_download

    logger.info("Downloading Norwegian PAWS-X dataset...")

    # Download tar.gz
    tar_path = hf_hub_download(
        repo_id='NbAiLab/norwegian-paws-x',
        filename='norwegian-x-final.tar.gz',
        repo_type='dataset',
        local_dir='data/paws-x'
    )

    logger.info(f"✓ Downloaded to: {tar_path}")
    logger.info("Extracting...")

    # Extract
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall('data/paws-x')

    logger.info("✓ Extracted to: data/paws-x/x-final/")
    logger.info("  Available languages: nb (Bokmål), nn (Nynorsk)")
    logger.info("  Files: train.json, dev_2k.json, test_2k.json")


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Download if needed
    data_path = Path("data/paws-x/x-final/nb")
    if not data_path.exists():
        logger.info("Data not found. Downloading...")
        download_paws_data()

    # Load and test
    train, dev, test = load_paws_dataset()

    logger.info(f"\n✓ Successfully loaded PAWS-X dataset")
    logger.info(f"  Train: {len(train):,} samples")
    logger.info(f"  Dev: {len(dev):,} samples")
    logger.info(f"  Test: {len(test):,} samples")
