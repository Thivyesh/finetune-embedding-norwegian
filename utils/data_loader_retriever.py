"""
Retrieval QA Data Loader: Query → Document retrieval datasets only.

Loads ONLY datasets that match the ETI format: short query → long document passage.
Excludes paraphrase, NLI, classification, and short-matching tasks.

Sources included:
- NorQuAD (Norwegian Wikipedia QA — query → passage)
- ScandiQA (Scandinavian extractive QA — query → passage)
- Supervised-DA (Danish Wikipedia queries — query → passage)

Sources explicitly EXCLUDED:
- PAWS-X (sentence ↔ sentence paraphrase — not retrieval)
- NorOpenBookQA (question → short fact, not a full document)

Format: {query, positive} — No hard negatives (uses in-batch negatives)
"""

from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_retrieval_qa_data(
    use_norquad: bool = True,
    use_scandiqa: bool = True,
    use_supervised_da: bool = True,
    scandiqa_languages: List[str] = ['no', 'da', 'sv'],
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load query → document retrieval QA data only.

    All included datasets share the ETI format: short query → long passage.
    This excludes paraphrase, NLI, classification, and short-matching tasks.

    Args:
        use_norquad: Include NorQuAD (Norwegian Wikipedia QA, ~3.8K)
        use_scandiqa: Include ScandiQA (NO/DA/SV extractive QA, ~18.9K)
        use_supervised_da: Include Supervised-DA (Danish Wikipedia, ~93K)
        scandiqa_languages: Languages for ScandiQA subset

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
        Each sample: {'query': str, 'positive': str}
    """
    all_train = []
    all_dev = []
    all_test = []

    # ===================================================================
    # 1. NorQuAD (Norwegian Wikipedia QA) — query → passage
    # ===================================================================
    if use_norquad:
        logger.info("Loading NorQuAD...")
        norquad = load_dataset('ltg/norquad')

        def format_norquad(example):
            return {
                'query': example['question'],
                'positive': example['context'],
            }

        norquad_train = norquad['train'].map(
            format_norquad, remove_columns=norquad['train'].column_names
        )
        norquad_dev = norquad['validation'].map(
            format_norquad, remove_columns=norquad['validation'].column_names
        )
        norquad_test = norquad['test'].map(
            format_norquad, remove_columns=norquad['test'].column_names
        )

        all_train.append(norquad_train)
        all_dev.append(norquad_dev)
        all_test.append(norquad_test)

        logger.info(
            f"✓ NorQuAD: {len(norquad_train):,} train, "
            f"{len(norquad_dev):,} dev, {len(norquad_test):,} test"
        )

    # ===================================================================
    # 2. ScandiQA (NO + DA + SV extractive QA) — query → passage
    # ===================================================================
    if use_scandiqa:
        logger.info(f"Loading ScandiQA ({', '.join(scandiqa_languages)})...")

        scandiqa_train_datasets = []
        scandiqa_dev_datasets = []
        scandiqa_test_datasets = []

        for lang in scandiqa_languages:
            try:
                scandiqa = load_dataset(
                    'parquet',
                    data_files={
                        'train': f'https://huggingface.co/datasets/alexandrainst/scandi-qa/resolve/refs%2Fconvert%2Fparquet/{lang}/train/*.parquet',
                        'test': f'https://huggingface.co/datasets/alexandrainst/scandi-qa/resolve/refs%2Fconvert%2Fparquet/{lang}/test/*.parquet',
                    }
                )

                def format_scandiqa(example):
                    return {
                        'query': example['question'],
                        'positive': example['context'],
                    }

                scandiqa_train_full = scandiqa['train'].map(
                    format_scandiqa, remove_columns=scandiqa['train'].column_names
                )
                scandiqa_test_lang = scandiqa['test'].map(
                    format_scandiqa, remove_columns=scandiqa['test'].column_names
                )

                # Split train into train/dev (90/10)
                train_size = int(0.9 * len(scandiqa_train_full))
                scandiqa_train_lang = scandiqa_train_full.select(range(train_size))
                scandiqa_dev_lang = scandiqa_train_full.select(
                    range(train_size, len(scandiqa_train_full))
                )

                scandiqa_train_datasets.append(scandiqa_train_lang)
                scandiqa_dev_datasets.append(scandiqa_dev_lang)
                scandiqa_test_datasets.append(scandiqa_test_lang)

                logger.info(
                    f"  ✓ ScandiQA ({lang}): {len(scandiqa_train_lang):,} train, "
                    f"{len(scandiqa_dev_lang):,} dev, {len(scandiqa_test_lang):,} test"
                )
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load ScandiQA ({lang}): {e}")

        if scandiqa_train_datasets:
            all_train.append(concatenate_datasets(scandiqa_train_datasets))
            all_dev.append(concatenate_datasets(scandiqa_dev_datasets))
            all_test.append(concatenate_datasets(scandiqa_test_datasets))

    # ===================================================================
    # 3. Supervised-DA (Danish Wikipedia queries) — query → passage
    # ===================================================================
    if use_supervised_da:
        logger.info("Loading Supervised-DA (Danish)...")
        supervised = load_dataset('jealk/supervised-da', split='train')

        def format_supervised_da(example):
            return {
                'query': example['query'],
                'positive': example['pos'],
            }

        supervised_formatted = supervised.map(
            format_supervised_da, remove_columns=supervised.column_names
        )

        # Split into train/dev/test (80/10/10)
        total = len(supervised_formatted)
        train_size = int(0.8 * total)
        dev_size = int(0.1 * total)

        all_train.append(supervised_formatted.select(range(train_size)))
        all_dev.append(supervised_formatted.select(
            range(train_size, train_size + dev_size)
        ))
        all_test.append(supervised_formatted.select(
            range(train_size + dev_size, total)
        ))

        logger.info(f"✓ Supervised-DA: {train_size:,} train, {dev_size:,} dev, {total - train_size - dev_size:,} test")

    # ===================================================================
    # Combine
    # ===================================================================
    if not all_train:
        raise ValueError("No retrieval QA datasets loaded! Enable at least one source.")

    logger.info("\nCombining retrieval QA datasets...")
    train_dataset = concatenate_datasets(all_train) if len(all_train) > 1 else all_train[0]
    dev_dataset = concatenate_datasets(all_dev) if len(all_dev) > 1 else all_dev[0]
    test_dataset = concatenate_datasets(all_test) if len(all_test) > 1 else all_test[0]

    # Shuffle training data
    train_dataset = train_dataset.shuffle(seed=42)

    logger.info(f"\n{'='*70}")
    logger.info("RETRIEVAL QA DATA (query → document format only)")
    logger.info(f"{'='*70}")
    logger.info(f"Train: {len(train_dataset):,} query-positive pairs")
    logger.info(f"Dev:   {len(dev_dataset):,} query-positive pairs")
    logger.info(f"Test:  {len(test_dataset):,} query-positive pairs")
    logger.info(f"Format: {train_dataset.column_names}")
    logger.info(f"\nSample:")
    logger.info(f"  Query: {train_dataset[0]['query']}")
    logger.info(f"  Positive: {train_dataset[0]['positive'][:150]}...")
    logger.info(f"\n⚠️  No hard negatives — will use in-batch negatives")

    return train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL QA DATA LOADER")
    print("=" * 70 + "\n")

    train, dev, test = load_retrieval_qa_data()

    print(f"\nTrain: {len(train):,}")
    print(f"Dev:   {len(dev):,}")
    print(f"Test:  {len(test):,}")

    for i in range(min(3, len(train))):
        print(f"\nSample {i+1}:")
        print(f"  Query: {train[i]['query'][:100]}...")
        print(f"  Positive: {train[i]['positive'][:100]}...")

    print("\n✓ Retrieval QA data loader test complete!")
