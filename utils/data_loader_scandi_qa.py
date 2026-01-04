"""
Scandinavian QA Data Loader: Multi-source QA datasets

Combines:
- NorQuAD (3,808 Norwegian QA)
- NorOpenBookQA (2,886 Norwegian QA)
- ScandiQA (18,924 NO+DA+SV QA)
- Supervised-DA (93,200 Danish pairs)
- PAWS-X (49,401 Norwegian paraphrase, label=1 only)

Total: ~168k training pairs

Format: {query, positive} - No hard negatives (uses in-batch negatives)
Loss: MultipleNegativesRankingLoss with in-batch negatives only
"""

from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Tuple
import logging

# Import existing PAWS loader
try:
    from utils.data_loader_paws import load_paws_dataset
except ImportError:
    from data_loader_paws import load_paws_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_scandi_qa_data(
    use_norquad: bool = True,
    use_openbookqa: bool = True,
    use_scandiqa: bool = True,
    use_supervised_da: bool = True,
    use_paws: bool = True,
    scandiqa_languages: list = ['no', 'da', 'sv'],  # All Scandinavian languages
    paws_data_dir: str = "data/paws-x/x-final",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load multi-source Scandinavian QA training data (no hard negatives).

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
        Each sample: {'query': str, 'positive': str}
    """

    all_train = []
    all_dev = []
    all_test = []

    # ===================================================================
    # 1. NorQuAD (Norwegian Wikipedia QA)
    # ===================================================================
    if use_norquad:
        logger.info("Loading NorQuAD...")
        norquad = load_dataset('ltg/norquad')

        def format_norquad(example):
            return {
                'query': example['question'],
                'positive': example['context'],
            }

        norquad_train = norquad['train'].map(format_norquad, remove_columns=norquad['train'].column_names)
        norquad_dev = norquad['validation'].map(format_norquad, remove_columns=norquad['validation'].column_names)
        norquad_test = norquad['test'].map(format_norquad, remove_columns=norquad['test'].column_names)

        all_train.append(norquad_train)
        all_dev.append(norquad_dev)
        all_test.append(norquad_test)

        logger.info(f"✓ NorQuAD: {len(norquad_train):,} train, {len(norquad_dev):,} dev, {len(norquad_test):,} test")

    # ===================================================================
    # 2. NorOpenBookQA (Norwegian Science QA)
    # ===================================================================
    if use_openbookqa:
        logger.info("Loading NorOpenBookQA...")
        openbookqa = load_dataset('ltg/noropenbookqa', 'nb')

        def format_openbookqa(example):
            return {
                'query': example['question_stem'],
                'positive': example['fact'] if example['fact'] else example['question_stem'],
            }

        openbookqa_train = openbookqa['train'].map(format_openbookqa, remove_columns=openbookqa['train'].column_names)
        openbookqa_test = openbookqa['test'].map(format_openbookqa, remove_columns=openbookqa['test'].column_names)

        # Split test for dev/test
        test_size = len(openbookqa_test)
        dev_size = test_size // 2
        openbookqa_dev = openbookqa_test.select(range(dev_size))
        openbookqa_test = openbookqa_test.select(range(dev_size, test_size))

        all_train.append(openbookqa_train)
        all_dev.append(openbookqa_dev)
        all_test.append(openbookqa_test)

        logger.info(f"✓ NorOpenBookQA: {len(openbookqa_train):,} train, {len(openbookqa_dev):,} dev, {len(openbookqa_test):,} test")

    # ===================================================================
    # 3. ScandiQA (NO + DA + SV extractive QA) - HIGH QUALITY
    # ===================================================================
    if use_scandiqa:
        logger.info(f"Loading ScandiQA ({', '.join(scandiqa_languages)})...")
        
        scandiqa_train_datasets = []
        scandiqa_dev_datasets = []
        scandiqa_test_datasets = []
        
        for lang in scandiqa_languages:
            try:
                # Load directly from Parquet files in the convert branch
                scandiqa = load_dataset(
                    'parquet',
                    data_files={
                        'train': f'https://huggingface.co/datasets/alexandrainst/scandi-qa/resolve/refs%2Fconvert%2Fparquet/{lang}/train/*.parquet',
                        'test': f'https://huggingface.co/datasets/alexandrainst/scandi-qa/resolve/refs%2Fconvert%2Fparquet/{lang}/test/*.parquet',
                    }
                )
                
                def format_scandiqa(example):
                    # ScandiQA has: question, answer, context
                    # Use question as query and context as positive
                    return {
                        'query': example['question'],
                        'positive': example['context'],
                    }
                
                # Format datasets
                scandiqa_train_full = scandiqa['train'].map(format_scandiqa, remove_columns=scandiqa['train'].column_names)
                scandiqa_test_lang = scandiqa['test'].map(format_scandiqa, remove_columns=scandiqa['test'].column_names)
                
                # Split train into train/validation (90/10)
                train_size = int(0.9 * len(scandiqa_train_full))
                scandiqa_train_lang = scandiqa_train_full.select(range(train_size))
                scandiqa_dev_lang = scandiqa_train_full.select(range(train_size, len(scandiqa_train_full)))
                
                scandiqa_train_datasets.append(scandiqa_train_lang)
                scandiqa_dev_datasets.append(scandiqa_dev_lang)
                scandiqa_test_datasets.append(scandiqa_test_lang)
                
                logger.info(f"  ✓ ScandiQA ({lang}): {len(scandiqa_train_lang):,} train, {len(scandiqa_dev_lang):,} dev, {len(scandiqa_test_lang):,} test")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load ScandiQA ({lang}): {e}")
        
        # Combine all ScandiQA languages
        if scandiqa_train_datasets:
            all_train.append(concatenate_datasets(scandiqa_train_datasets))
            all_dev.append(concatenate_datasets(scandiqa_dev_datasets))
            all_test.append(concatenate_datasets(scandiqa_test_datasets))
            
            total_train = sum(len(ds) for ds in scandiqa_train_datasets)
            total_dev = sum(len(ds) for ds in scandiqa_dev_datasets)
            total_test = sum(len(ds) for ds in scandiqa_test_datasets)
            logger.info(f"✓ ScandiQA Total: {total_train:,} train, {total_dev:,} dev, {total_test:,} test")

    # ===================================================================
    # 4. Supervised-DA (Danish Wikipedia Queries) - SOTA uses this!
    # ===================================================================
    if use_supervised_da:
        logger.info("Loading Supervised-DA (Danish)...")
        supervised = load_dataset('jealk/supervised-da', split='train')

        def format_supervised_da(example):
            return {
                'query': example['query'],
                'positive': example['pos'],
            }

        supervised_train = supervised.map(format_supervised_da, remove_columns=supervised.column_names)

        # Split for dev/test (80/10/10)
        total = len(supervised_train)
        train_size = int(0.8 * total)
        dev_size = int(0.1 * total)

        supervised_train_split = supervised_train.select(range(train_size))
        supervised_dev = supervised_train.select(range(train_size, train_size + dev_size))
        supervised_test = supervised_train.select(range(train_size + dev_size, total))

        all_train.append(supervised_train_split)
        all_dev.append(supervised_dev)
        all_test.append(supervised_test)

        logger.info(f"✓ Supervised-DA: {len(supervised_train_split):,} train, {len(supervised_dev):,} dev, {len(supervised_test):,} test")

    # ===================================================================
    # 5. PAWS-X Norwegian (Paraphrase pairs, label=1 only)
    # ===================================================================
    if use_paws:
        try:
            # Use existing PAWS loader
            paws_train_raw, paws_dev_raw, paws_test_raw = load_paws_dataset(data_dir=paws_data_dir)

            # Convert to query-positive format (filter for paraphrases only)
            def format_paws(example):
                # Only keep paraphrases (label=1)
                if example['label'] == 1:
                    return {
                        'query': example['sentence1'],
                        'positive': example['sentence2'],
                    }
                return None

            paws_train = paws_train_raw.filter(lambda x: x['label'] == 1).map(
                format_paws,
                remove_columns=paws_train_raw.column_names
            )
            paws_dev = paws_dev_raw.filter(lambda x: x['label'] == 1).map(
                format_paws,
                remove_columns=paws_dev_raw.column_names
            )
            paws_test = paws_test_raw.filter(lambda x: x['label'] == 1).map(
                format_paws,
                remove_columns=paws_test_raw.column_names
            )

            all_train.append(paws_train)
            all_dev.append(paws_dev)
            all_test.append(paws_test)

            logger.info(f"✓ PAWS-X: {len(paws_train):,} train, {len(paws_dev):,} dev, {len(paws_test):,} test (paraphrases only)")
        except FileNotFoundError as e:
            logger.warning("⚠️  PAWS-X not found - skipping. Run data_loader_paws.py first to download.")
            logger.warning(f"    Error: {e}")

    # ===================================================================
    # Combine all datasets
    # ===================================================================
    if not all_train:
        raise ValueError("No datasets loaded! Check paths and enable at least one dataset.")

    logger.info("\nCombining all Group B datasets...")
    train_dataset = concatenate_datasets(all_train) if len(all_train) > 1 else all_train[0]
    dev_dataset = concatenate_datasets(all_dev) if len(all_dev) > 1 else all_dev[0]
    test_dataset = concatenate_datasets(all_test) if len(all_test) > 1 else all_test[0]

    # Shuffle training data
    train_dataset = train_dataset.shuffle(seed=42)

    logger.info(f"\n{'='*70}")
    logger.info("GROUP B (EASY RETRIEVAL) - STAGE 2 DATA")
    logger.info(f"{'='*70}")
    logger.info(f"Train: {len(train_dataset):,} query-positive pairs")
    logger.info(f"Dev:   {len(dev_dataset):,} query-positive pairs")
    logger.info(f"Test:  {len(test_dataset):,} query-positive pairs")
    logger.info(f"\nFormat: {train_dataset.column_names}")
    logger.info("\nSample:")
    logger.info(f"  Query: {train_dataset[0]['query']}")
    logger.info(f"  Positive: {train_dataset[0]['positive'][:150]}...")
    logger.info("\n⚠️  NO HARD NEGATIVES - Will use in-batch negatives")
    logger.info("This is EASY curriculum learning (Stage 2)")

    return train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING SCANDINAVIAN QA DATA LOADER")
    print("="*70 + "\n")

    train, dev, test = load_scandi_qa_data(
        use_norquad=True,
        use_openbookqa=True,
        use_scandiqa=False,  # Skip for now (deprecated loader)
        use_supervised_da=True,
        use_paws=True,
    )

    print("\n" + "="*70)
    print("SAMPLES FROM DIFFERENT SOURCES")
    print("="*70)

    for i in range(min(5, len(train))):
        print(f"\nSample {i+1}:")
        print(f"  Query: {train[i]['query'][:100]}...")
        print(f"  Positive: {train[i]['positive'][:100]}...")

    print("\n✓ Scandinavian QA data loader test complete!")
