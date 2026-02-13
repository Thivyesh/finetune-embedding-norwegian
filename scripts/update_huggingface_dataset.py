#!/usr/bin/env python3
"""
Add ETI dataset to existing HuggingFace dataset.

This script loads an existing Scandinavian embedding training dataset from HuggingFace,
adds the ETI (Norwegian health/welfare) data, and pushes it back.

Usage:
    # Dry run - see what would be added
    python scripts/add_eti_to_dataset.py
    
    # Push to HuggingFace
    python scripts/add_eti_to_dataset.py --push
    
    # Save locally
    python scripts/add_eti_to_dataset.py --save-local data/combined_with_eti
    
    # Custom repo
    python scripts/add_eti_to_dataset.py --repo-id your-username/your-dataset --push
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_eti() -> dict[str, Dataset]:
    """
    Load ETI (Norwegian health/welfare) dataset - both simple and advanced.
    
    Returns:
        Dictionary with 'train', 'dev', and 'test' HuggingFace Dataset splits
    """
    logger.info("Loading ETI dataset (smpl + adv)...")
    
    # Load both types
    all_splits = {"train": [], "dev": [], "test": []}
    
    for dataset_type in ["smpl", "adv"]:
        train_file = Path(f"data/processed/eti_train_{dataset_type}.json")
        test_file = Path(f"data/processed/eti_test_{dataset_type}.json")
        
        if not train_file.exists():
            logger.warning(f"  ⚠ ETI train {dataset_type} not found at {train_file} – skipping")
            continue
        if not test_file.exists():
            logger.warning(f"  ⚠ ETI test {dataset_type} not found at {test_file} – skipping")
            continue
        
        # Load data (LlamaIndex format: dict with queries, corpus, relevant_docs)
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Convert from LlamaIndex format to HuggingFace format
        def convert_to_hf_format(data, dataset_type):
            """Convert LlamaIndex format to list of (anchor, positive) pairs."""
            pairs = []
            
            queries = data.get('queries', {})
            corpus = data.get('corpus', {})
            relevant_docs = data.get('relevant_docs', {})
            
            for query_id, query_text in queries.items():
                # Get the relevant document(s) for this query
                doc_ids = relevant_docs.get(query_id, [])
                
                for doc_id in doc_ids:
                    if doc_id in corpus:
                        pairs.append({
                            "anchor": query_text,
                            "positive": corpus[doc_id],
                            "negative": None,
                            "source": f"eti-{dataset_type}",
                            "language": "no",
                            "task_type": "qa",
                        })
            
            return pairs
        
        train_pairs = convert_to_hf_format(train_data, dataset_type)
        test_pairs = convert_to_hf_format(test_data, dataset_type)
        
        if not train_pairs:
            logger.warning(f"  ⚠ No training pairs found in {train_file}")
            continue
        
        if not test_pairs:
            logger.warning(f"  ⚠ No test pairs found in {test_file}")
            continue
        
        # Convert to HuggingFace Dataset
        train_ds = Dataset.from_dict({
            "anchor": [p["anchor"] for p in train_pairs],
            "positive": [p["positive"] for p in train_pairs],
            "negative": [p["negative"] for p in train_pairs],
            "source": [p["source"] for p in train_pairs],
            "language": [p["language"] for p in train_pairs],
            "task_type": [p["task_type"] for p in train_pairs],
        })
        
        test_ds = Dataset.from_dict({
            "anchor": [p["anchor"] for p in test_pairs],
            "positive": [p["positive"] for p in test_pairs],
            "negative": [p["negative"] for p in test_pairs],
            "source": [p["source"] for p in test_pairs],
            "language": [p["language"] for p in test_pairs],
            "task_type": [p["task_type"] for p in test_pairs],
        })
        
        # Split test into dev/test (50/50)
        mid = len(test_ds) // 2
        
        all_splits["train"].append(train_ds)
        all_splits["dev"].append(test_ds.select(range(mid)))
        all_splits["test"].append(test_ds.select(range(mid, len(test_ds))))
        
        logger.info(f"  ✓ ETI {dataset_type} train: {len(train_ds):,} pairs")
        logger.info(f"      Queries: {len(train_data['queries'])}, Docs: {len(train_data['corpus'])}")
    
    # Concatenate all splits
    splits = {}
    for split_name, datasets in all_splits.items():
        if datasets:
            splits[split_name] = concatenate_datasets(datasets)
            logger.info(f"  ✓ ETI combined {split_name}: {len(splits[split_name]):,}")
    
    return splits


def print_summary(dd: DatasetDict) -> None:
    """Print a detailed summary of the dataset."""
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    for split_name, ds in dd.items():
        print(f"\n{'─' * 40}")
        print(f"Split: {split_name} — {len(ds):,} samples")
        print(f"{'─' * 40}")
        
        # Source distribution
        from collections import Counter
        sources = Counter(ds["source"])
        print("  By source:")
        for src, cnt in sources.most_common():
            print(f"    {src:40s} {cnt:>10,}")
        
        # Language distribution
        langs = Counter(ds["language"])
        print("  By language:")
        for lang, cnt in langs.most_common():
            print(f"    {lang:40s} {cnt:>10,}")
        
        # Negative availability
        has_neg = sum(1 for n in ds["negative"] if n is not None)
        no_neg = len(ds) - has_neg
        print(f"  With hard negatives:    {has_neg:>10,} ({has_neg/len(ds)*100:.1f}%)")
        print(f"  In-batch negatives only:{no_neg:>10,} ({no_neg/len(ds)*100:.1f}%)")
    
    total = sum(len(ds) for ds in dd.values())
    print(f"\n{'=' * 80}")
    print(f"TOTAL across all splits: {total:,}")
    print(f"{'=' * 80}\n")


def add_eti_to_existing_dataset(
   repo_id: str,
    push: bool = False,
    private: bool = False,
    save_local: str = None,
    create_pr: bool = False  # ← Nytt parameter
) -> DatasetDict:
    """
    Load existing dataset from HuggingFace, add ETI data, and optionally push back.
    
    Args:
        repo_id: HuggingFace repo ID to load from and push to
        push: Whether to push the updated dataset back to HuggingFace
        private: Whether to make the repo private
        save_local: Optional path to save dataset locally
        create_pr: Whether to create a Pull Request instead of pushing directly
    
    Returns:
        Updated DatasetDict with ETI data added
    """
    logger.info(f"Loading existing dataset from {repo_id}...")
    
    # Load existing dataset from HuggingFace
    try:
        existing_dd = load_dataset(repo_id)
    except Exception as e:
        logger.error(f"Failed to load dataset from {repo_id}: {e}")
        raise
    
    logger.info(f"  ✓ Loaded existing dataset")
    logger.info(f"    Train: {len(existing_dd['train']):,}")
    logger.info(f"    Dev: {len(existing_dd['dev']):,}")
    logger.info(f"    Test: {len(existing_dd['test']):,}")
    
    # Load ETI data
    eti_splits = load_eti()
    
    if not eti_splits:
        logger.error("No ETI data loaded. Check that files exist in data/eti/")
        raise ValueError("No ETI data found")
    
    # Combine with existing data
    logger.info("\nCombining existing dataset with ETI...")
    
    combined = {}
    for split_name in ["train", "dev", "test"]:
        datasets_to_combine = []
        
        # Add existing data
        if split_name in existing_dd:
            datasets_to_combine.append(existing_dd[split_name])
            logger.info(f"  Existing {split_name}: {len(existing_dd[split_name]):,}")
        
        # Add ETI data
        if split_name in eti_splits:
            datasets_to_combine.append(eti_splits[split_name])
            logger.info(f"  ETI {split_name}: {len(eti_splits[split_name]):,}")
        
        if datasets_to_combine:
            combined[split_name] = concatenate_datasets(datasets_to_combine)
            if split_name == "train":
                combined[split_name] = combined[split_name].shuffle(seed=42)
            logger.info(f"  → Combined {split_name}: {len(combined[split_name]):,}")
    
    updated_dd = DatasetDict(combined)
    
    # Print summary
    print_summary(updated_dd)
    
    # Save locally if requested
    if save_local:
        logger.info(f"Saving to {save_local}...")
        Path(save_local).parent.mkdir(parents=True, exist_ok=True)
        updated_dd.save_to_disk(save_local)
        logger.info(f"✓ Saved to {save_local}")
    
    # Push if requested
    if push:
        if create_pr:
            logger.info(f"Creating Pull Request for {repo_id}...")
        else:
            logger.info(f"Pushing updated dataset to {repo_id}...")
        
        updated_dd.push_to_hub(
            repo_id, 
            private=private,
            create_pr=create_pr  # ← Legg til dette
        )
        
        if create_pr:
            logger.info(f"✓ Pull Request created for https://huggingface.co/datasets/{repo_id}")
        else:
            logger.info(f"✓ Pushed to https://huggingface.co/datasets/{repo_id}")
    else:
        logger.info("\nDry run complete. Use --push to upload to HuggingFace.")
    
    return updated_dd


def main():
    parser = argparse.ArgumentParser(
        description="Add ETI dataset to existing HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - see what would be added
  python scripts/update_huggingface_dataset.py
  
  # Create Pull Request (when you don't have write access)
  python scripts/update_huggingface_dataset.py --push --create-pr
  
  # Save locally first
  python scripts/update_huggingface_dataset.py --save-local data/combined_with_eti
  
  # Push directly (requires write access)
  python scripts/update_huggingface_dataset.py --push
        """
    )
    
    parser.add_argument(
        "--repo-id",
        default="thivy/scandinavian-embedding-training-data",
        help="HuggingFace repo ID to load from and push to"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push updated dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a Pull Request instead of pushing directly (use when you don't have write access)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repo private"
    )
    parser.add_argument(
        "--save-local",
        type=str,
        default=None,
        help="Save dataset locally to this path (Arrow format)"
    )
    
    args = parser.parse_args()
    
    try:
        add_eti_to_existing_dataset(
            repo_id=args.repo_id,
            push=args.push,
            private=args.private,
            save_local=args.save_local,
            create_pr=args.create_pr
        )
    except Exception as e:
        logger.error(f"Failed to add ETI to dataset: {e}")
        raise


if __name__ == "__main__":
    main()