"""
DDSC Data Loader: Nordic Embedding Training Data

Dataset: DDSC Nordic Embedding Training Data
- Norwegian: 242,620 samples (25%)
- Danish: 483,509 samples (50%)
- Swedish: 242,120 samples (25%)
- Total: 968,249 samples

Format: {query, positive, negative, instruction, task, language}
- ~40% have HARD NEGATIVES (retrieval & unit-triple tasks)
- ~60% use IN-BATCH NEGATIVES only (text-matching & classification tasks)
- Has task instructions
- LLM-generated high quality

Loss: MultipleNegativesRankingLoss (handles both with/without hard negatives)

Note: ALL samples are valuable! MultipleNegativesRankingLoss automatically:
- Uses provided hard negatives + in-batch negatives (when negative != None)
- Uses only in-batch negatives (when negative == None)
"""

from datasets import load_dataset, Dataset
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ddsc_data(
    languages: List[str] = ['norwegian', 'danish', 'swedish'],  # All Scandinavian
    tasks: Optional[List[str]] = None,  # None = all tasks
    split_ratio: Tuple[float, float, float] = (0.98, 0.01, 0.01),  # train/dev/test
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load DDSC Nordic Embedding Training Data (all tasks, mixed hard/in-batch negatives).

    Args:
        languages: List of languages to include ('norwegian', 'danish', 'swedish')
        tasks: List of tasks to include (None = all tasks)
               Options: 'retrieval', 'text-matching-short', 'text-matching-long',
                       'unit-triple', 'classification'
        split_ratio: Train/dev/test split (default: 98/1/1)

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
        Each sample: {
            'query': str,
            'positive': str,
            'negative': str,  ← HARD NEGATIVE!
            'instruction': str,
            'task': str,
            'language': str
        }
    """

    logger.info("="*70)
    logger.info("Loading DDSC Nordic Embedding Training Data (Group A)")
    logger.info("="*70)
    logger.info(f"Languages: {languages}")
    logger.info(f"Tasks: {tasks if tasks else 'all'}")

    # Load full dataset
    logger.info("\nLoading dataset...")
    ddsc = load_dataset('DDSC/nordic-embedding-training-data', split='train')
    logger.info(f"✓ Loaded {len(ddsc):,} total samples")

    # Filter by language
    if languages and len(languages) < 3:
        logger.info(f"\nFiltering for languages: {languages}...")
        ddsc = ddsc.filter(lambda x: x['language'] in languages)
        logger.info(f"✓ Filtered to {len(ddsc):,} samples")

    # Filter by task (optional)
    if tasks:
        logger.info(f"\nFiltering for tasks: {tasks}...")
        ddsc = ddsc.filter(lambda x: x['task'] in tasks)
        logger.info(f"✓ Filtered to {len(ddsc):,} samples")

    # Rename columns to match our format
    def format_ddsc(example):
        return {
            'query': example['query'],
            'positive': example['positive'],
            'negative': example['negative'],  # HARD NEGATIVE!
            'instruction': example.get('instruction', ''),
            'task': example.get('task', ''),
            'language': example.get('language', ''),
        }

    logger.info("\nFormatting dataset...")
    ddsc_formatted = ddsc.map(
        format_ddsc,
        remove_columns=['prompt', 'response']  # Remove LLM generation metadata
    )

    # Shuffle
    logger.info("Shuffling...")
    ddsc_formatted = ddsc_formatted.shuffle(seed=42)

    # Split into train/dev/test
    total_size = len(ddsc_formatted)
    train_size = int(total_size * split_ratio[0])
    dev_size = int(total_size * split_ratio[1])

    logger.info(f"\nSplitting dataset ({split_ratio[0]:.0%}/{split_ratio[1]:.0%}/{split_ratio[2]:.0%})...")

    train_dataset = ddsc_formatted.select(range(train_size))
    dev_dataset = ddsc_formatted.select(range(train_size, train_size + dev_size))
    test_dataset = ddsc_formatted.select(range(train_size + dev_size, total_size))

    # Statistics
    logger.info(f"\n{'='*70}")
    logger.info("GROUP A (HARD RETRIEVAL) - STAGE 3 DATA")
    logger.info(f"{'='*70}")
    logger.info(f"Train: {len(train_dataset):,} samples")
    logger.info(f"Dev:   {len(dev_dataset):,} samples")
    logger.info(f"Test:  {len(test_dataset):,} samples")

    # Language distribution
    logger.info(f"\nLanguage distribution (train):")
    lang_counts = {}
    for item in train_dataset:
        lang = item['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    for lang, count in sorted(lang_counts.items()):
        pct = count / len(train_dataset) * 100
        logger.info(f"  {lang.capitalize()}: {count:,} ({pct:.1f}%)")

    # Task distribution
    logger.info(f"\nTask distribution (train):")
    task_counts = {}
    for item in train_dataset:
        task = item['task']
        task_counts[task] = task_counts.get(task, 0) + 1

    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = count / len(train_dataset) * 100
        logger.info(f"  {task}: {count:,} ({pct:.1f}%)")

    # Show samples
    logger.info("\n" + "="*70)
    logger.info("SAMPLE DATA")
    logger.info("="*70)
    
    # Find sample with hard negative
    sample_with_neg = next((s for s in train_dataset if s['negative'] is not None), None)
    if sample_with_neg:
        logger.info("\nSample WITH hard negative:")
        logger.info(f"  Language: {sample_with_neg['language']}")
        logger.info(f"  Task: {sample_with_neg['task']}")
        logger.info(f"  Query: {sample_with_neg['query'][:80]}...")
        logger.info(f"  Positive: {sample_with_neg['positive'][:80]}...")
        logger.info(f"  Negative (HARD): {sample_with_neg['negative'][:80]}...")
    
    # Find sample without hard negative
    sample_without_neg = next((s for s in train_dataset if s['negative'] is None), None)
    if sample_without_neg:
        logger.info("\nSample WITHOUT hard negative (in-batch negatives only):")
        logger.info(f"  Language: {sample_without_neg['language']}")
        logger.info(f"  Task: {sample_without_neg['task']}")
        logger.info(f"  Query: {sample_without_neg['query'][:80]}...")
        logger.info(f"  Positive: {sample_without_neg['positive'][:80]}...")
        logger.info(f"  Negative: None (will use in-batch negatives)")

    logger.info("\n✅ Dataset contains BOTH types - MultipleNegativesRankingLoss handles both!")
    logger.info("   - ~40% with hard negatives (retrieval & unit-triple tasks)")
    logger.info("   - ~60% with in-batch negatives only (text-matching & classification)")
    logger.info("   - ALL 968k samples are valuable for training!")
    logger.info("\nThis is HARD curriculum learning (Stage 3)")

    return train_dataset, dev_dataset, test_dataset


def load_ddsc_norwegian_only() -> Tuple[Dataset, Dataset, Dataset]:
    """Convenience function to load Norwegian only."""
    return load_ddsc_data(languages=['norwegian'])


def load_ddsc_nordic() -> Tuple[Dataset, Dataset, Dataset]:
    """Convenience function to load all Nordic languages (recommended)."""
    return load_ddsc_data(languages=['norwegian', 'danish', 'swedish'])


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING DDSC DATA LOADER")
    print("="*70 + "\n")

    # Test 1: All Nordic languages (recommended)
    print("Test 1: All Nordic languages (NO + DA + SV)")
    print("-"*70)
    train, dev, test = load_ddsc_nordic()

    print("\n" + "="*70)
    print("SAMPLES WITH HARD NEGATIVES")
    print("="*70)

    for i in range(min(3, len(train))):
        sample = train[i]
        print(f"\nSample {i+1} ({sample['language']}, {sample['task']}):")
        print(f"  Query: {sample['query'][:80]}...")
        print(f"  Positive: {sample['positive'][:80]}...")
        if sample['negative'] is not None:
            print(f"  HARD Negative: {sample['negative'][:80]}...")
        else:
            print("  HARD Negative: None (⚠️  WARNING: Missing hard negative!)")
        print(f"  Instruction: {sample['instruction'][:60]}...")

    # Test 2: Norwegian only
    print("\n" + "="*70)
    print("\nTest 2: Norwegian only")
    print("-"*70)
    train_no, dev_no, test_no = load_ddsc_norwegian_only()

    print("\n✓ DDSC data loader test complete!")
