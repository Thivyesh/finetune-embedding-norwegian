"""
Create a combined HuggingFace dataset from all training sources.

Merges:
  1. NLI (Fremtind/all-nli-norwegian)         ~569k triplets
  2. NorQuAD (ltg/norquad)                     ~3.8k pairs
  3. NorOpenBookQA (ltg/noropenbookqa)         ~2.9k pairs
  4. ScandiQA (alexandrainst/scandi-qa)        ~17k pairs (NO+DA+SV)
  5. Supervised-DA (jealk/supervised-da)        ~93k pairs
  6. PAWS-X Norwegian (local JSONL)            ~49k pairs (label=1 only)
  7. DDSC (DDSC/nordic-embedding-training-data) ~968k (NO+DA+SV, mixed negatives)

Unified schema:
  - anchor:    str         (the query / anchor sentence)
  - positive:  str         (the positive / similar sentence)
  - negative:  str | None  (hard negative, if available)
  - source:    str         (dataset origin)
  - language:  str         (no, da, sv)
  - task_type: str         (nli, qa, retrieval, paraphrase, etc.)

Usage:
  uv run python scripts/create_combined_dataset.py
  uv run python scripts/create_combined_dataset.py --push
  uv run python scripts/create_combined_dataset.py --push --repo-id thivy/scandinavian-embedding-training-data
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


# ─────────────────────────────────────────────────────────────────────
# 1. NLI
# ─────────────────────────────────────────────────────────────────────
def load_nli() -> dict[str, Dataset]:
    """Load Fremtind/all-nli-norwegian and normalise columns."""
    logger.info("Loading NLI (Fremtind/all-nli-norwegian)...")
    ds = load_dataset("Fremtind/all-nli-norwegian")

    def fmt(example):
        return {
            "anchor": example["anchor"],
            "positive": example["positive"],
            "negative": example["negative"],
            "source": "fremtind-all-nli-norwegian",
            "language": "no",
            "task_type": "nli",
        }

    splits = {}
    for split_name, hf_split in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        if hf_split in ds:
            splits[split_name] = ds[hf_split].map(fmt, remove_columns=ds[hf_split].column_names)
            logger.info(f"  ✓ NLI {split_name}: {len(splits[split_name]):,}")
    return splits


# ─────────────────────────────────────────────────────────────────────
# 2. NorQuAD
# ─────────────────────────────────────────────────────────────────────
def load_norquad() -> dict[str, Dataset]:
    logger.info("Loading NorQuAD (ltg/norquad)...")
    ds = load_dataset("ltg/norquad")

    def fmt(example):
        return {
            "anchor": example["question"],
            "positive": example["context"],
            "negative": None,
            "source": "ltg-norquad",
            "language": "no",
            "task_type": "qa",
        }

    splits = {}
    for split_name, hf_split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        if hf_split in ds:
            splits[split_name] = ds[hf_split].map(fmt, remove_columns=ds[hf_split].column_names)
            logger.info(f"  ✓ NorQuAD {split_name}: {len(splits[split_name]):,}")
    return splits


# ─────────────────────────────────────────────────────────────────────
# 3. NorOpenBookQA
# ─────────────────────────────────────────────────────────────────────
def load_noropenbookqa() -> dict[str, Dataset]:
    logger.info("Loading NorOpenBookQA (ltg/noropenbookqa)...")
    ds = load_dataset("ltg/noropenbookqa", "nb")

    def fmt(example):
        return {
            "anchor": example["question_stem"],
            "positive": example["fact"] if example["fact"] else example["question_stem"],
            "negative": None,
            "source": "ltg-noropenbookqa",
            "language": "no",
            "task_type": "qa",
        }

    splits = {}
    # Train
    if "train" in ds:
        splits["train"] = ds["train"].map(fmt, remove_columns=ds["train"].column_names)
        logger.info(f"  ✓ NorOpenBookQA train: {len(splits['train']):,}")

    # Split test 50/50 for dev/test
    if "test" in ds:
        test_ds = ds["test"].map(fmt, remove_columns=ds["test"].column_names)
        mid = len(test_ds) // 2
        splits["dev"] = test_ds.select(range(mid))
        splits["test"] = test_ds.select(range(mid, len(test_ds)))
        logger.info(f"  ✓ NorOpenBookQA dev: {len(splits['dev']):,}")
        logger.info(f"  ✓ NorOpenBookQA test: {len(splits['test']):,}")
    return splits


# ─────────────────────────────────────────────────────────────────────
# 4. ScandiQA (NO + DA + SV)
# ─────────────────────────────────────────────────────────────────────
def load_scandiqa(languages: list[str] | None = None) -> dict[str, Dataset]:
    languages = languages or ["no", "da", "sv"]
    logger.info(f"Loading ScandiQA ({', '.join(languages)})...")

    all_train, all_dev, all_test = [], [], []

    for lang in languages:
        try:
            ds = load_dataset(
                "parquet",
                data_files={
                    "train": f"https://huggingface.co/datasets/alexandrainst/scandi-qa/resolve/refs%2Fconvert%2Fparquet/{lang}/train/*.parquet",
                    "test": f"https://huggingface.co/datasets/alexandrainst/scandi-qa/resolve/refs%2Fconvert%2Fparquet/{lang}/test/*.parquet",
                },
            )

            def fmt(example, _lang=lang):
                return {
                    "anchor": example["question"],
                    "positive": example["context"],
                    "negative": None,
                    "source": "alexandrainst-scandi-qa",
                    "language": _lang,
                    "task_type": "qa",
                }

            train_full = ds["train"].map(fmt, remove_columns=ds["train"].column_names)
            test_ds = ds["test"].map(fmt, remove_columns=ds["test"].column_names)

            # Split train 90/10
            split_idx = int(0.9 * len(train_full))
            all_train.append(train_full.select(range(split_idx)))
            all_dev.append(train_full.select(range(split_idx, len(train_full))))
            all_test.append(test_ds)
            logger.info(f"  ✓ ScandiQA ({lang}): {split_idx:,} train")
        except Exception as e:
            logger.warning(f"  ⚠ ScandiQA ({lang}) failed: {e}")

    splits: dict[str, Dataset] = {}
    if all_train:
        splits["train"] = concatenate_datasets(all_train)
        splits["dev"] = concatenate_datasets(all_dev)
        splits["test"] = concatenate_datasets(all_test)
    return splits


# ─────────────────────────────────────────────────────────────────────
# 5. Supervised-DA
# ─────────────────────────────────────────────────────────────────────
def load_supervised_da() -> dict[str, Dataset]:
    logger.info("Loading Supervised-DA (jealk/supervised-da)...")
    raw = load_dataset("jealk/supervised-da", split="train")

    def fmt(example):
        return {
            "anchor": example["query"],
            "positive": example["pos"],
            "negative": None,
            "source": "jealk-supervised-da",
            "language": "da",
            "task_type": "qa",
        }

    formatted = raw.map(fmt, remove_columns=raw.column_names)
    total = len(formatted)
    train_end = int(0.8 * total)
    dev_end = int(0.9 * total)

    splits = {
        "train": formatted.select(range(train_end)),
        "dev": formatted.select(range(train_end, dev_end)),
        "test": formatted.select(range(dev_end, total)),
    }
    for k, v in splits.items():
        logger.info(f"  ✓ Supervised-DA {k}: {len(v):,}")
    return splits


# ─────────────────────────────────────────────────────────────────────
# 6. PAWS-X Norwegian
# ─────────────────────────────────────────────────────────────────────
def load_paws(data_dir: str = "data/paws-x/x-final") -> dict[str, Dataset]:
    logger.info("Loading PAWS-X Norwegian...")
    data_path = Path(data_dir) / "nb"

    if not data_path.exists():
        logger.warning(f"  ⚠ PAWS-X data not found at {data_path} – skipping")
        return {}

    def read_jsonl(path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    files = {
        "train": data_path / "translated_train.json",
        "dev": data_path / "translated_dev_2k.json",
        "test": data_path / "translated_test_2k.json",
    }

    splits: dict[str, Dataset] = {}
    for split_name, fpath in files.items():
        if not fpath.exists():
            logger.warning(f"  ⚠ {fpath} not found – skipping {split_name}")
            continue
        rows = read_jsonl(fpath)
        # Only keep paraphrases (label=1)
        positives = [r for r in rows if r["label"] == 1]
        splits[split_name] = Dataset.from_dict(
            {
                "anchor": [r["sentence1"] for r in positives],
                "positive": [r["sentence2"] for r in positives],
                "negative": [None] * len(positives),
                "source": ["paws-x-norwegian"] * len(positives),
                "language": ["no"] * len(positives),
                "task_type": ["paraphrase"] * len(positives),
            }
        )
        logger.info(f"  ✓ PAWS-X {split_name}: {len(splits[split_name]):,} (label=1 only)")
    return splits


# ─────────────────────────────────────────────────────────────────────
# 7. DDSC Nordic Embedding Training Data
# ─────────────────────────────────────────────────────────────────────
DDSC_LANG_MAP = {"norwegian": "no", "danish": "da", "swedish": "sv"}


def load_ddsc(
    languages: list[str] | None = None,
    split_ratio: tuple[float, float, float] = (0.98, 0.01, 0.01),
) -> dict[str, Dataset]:
    languages = languages or ["norwegian", "danish", "swedish"]
    logger.info(f"Loading DDSC ({', '.join(languages)})...")
    raw = load_dataset("DDSC/nordic-embedding-training-data", split="train")

    # Filter languages
    if len(languages) < 3:
        raw = raw.filter(lambda x: x["language"] in languages)

    def fmt(example):
        return {
            "anchor": example["query"],
            "positive": example["positive"],
            "negative": example["negative"],  # may be None
            "source": "ddsc-nordic-embedding",
            "language": DDSC_LANG_MAP.get(example["language"], example["language"]),
            "task_type": example.get("task", "retrieval"),
        }

    formatted = raw.map(fmt, remove_columns=[c for c in raw.column_names if c not in []])
    # Remove any leftover original columns
    keep_cols = {"anchor", "positive", "negative", "source", "language", "task_type"}
    drop_cols = [c for c in formatted.column_names if c not in keep_cols]
    if drop_cols:
        formatted = formatted.remove_columns(drop_cols)

    formatted = formatted.shuffle(seed=42)
    total = len(formatted)
    train_end = int(total * split_ratio[0])
    dev_end = train_end + int(total * split_ratio[1])

    splits = {
        "train": formatted.select(range(train_end)),
        "dev": formatted.select(range(train_end, dev_end)),
        "test": formatted.select(range(dev_end, total)),
    }
    for k, v in splits.items():
        logger.info(f"  ✓ DDSC {k}: {len(v):,}")
    return splits

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
        train_file = Path(f"data/eti/eti_train_{dataset_type}.json")
        test_file = Path(f"data/eti/eti_test_{dataset_type}.json")
        
        if not train_file.exists():
            logger.warning(f"  ETI train {dataset_type} not found at {train_file} – skipping")
            continue
        if not test_file.exists():
            logger.warning(f"  ETI test {dataset_type} not found at {test_file} – skipping")
            continue
        
        # Load data
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Convert to HuggingFace Dataset
        train_ds = Dataset.from_dict({
            "anchor": [item["anchor"] for item in train_data],
            "positive": [item["context"] for item in train_data],
            "negative": [None] * len(train_data),
            "source": [f"eti-{dataset_type}"] * len(train_data),
            "language": ["no"] * len(train_data),
            "task_type": ["qa"] * len(train_data),
        })
        
        test_ds = Dataset.from_dict({
            "anchor": [item["anchor"] for item in test_data],
            "positive": [item["context"] for item in test_data],
            "negative": [None] * len(test_data),
            "source": [f"eti-{dataset_type}"] * len(test_data),
            "language": ["no"] * len(test_data),
            "task_type": ["qa"] * len(test_data),
        })
        
        # Split test into dev/test (50/50)
        mid = len(test_ds) // 2
        
        all_splits["train"].append(train_ds)
        all_splits["dev"].append(test_ds.select(range(mid)))
        all_splits["test"].append(test_ds.select(range(mid, len(test_ds))))
        
        logger.info(f"  ETI {dataset_type} train: {len(train_ds):,}")
    
    # Concatenate all splits
    splits = {}
    for split_name, datasets in all_splits.items():
        if datasets:
            splits[split_name] = concatenate_datasets(datasets)
            logger.info(f"  ETI combined {split_name}: {len(splits[split_name]):,}")
    
    return splits


# ─────────────────────────────────────────────────────────────────────
# Combine everything
# ─────────────────────────────────────────────────────────────────────
def combine_all() -> DatasetDict:
    """Load all sources, normalise, and concatenate into a single DatasetDict."""

    loaders = [
        ("NLI", load_nli),
        ("NorQuAD", load_norquad),
        ("NorOpenBookQA", load_noropenbookqa),
        ("ScandiQA", load_scandiqa),
        ("Supervised-DA", load_supervised_da),
        ("PAWS-X", load_paws),
        ("DDSC", load_ddsc),
    ]

    per_split: dict[str, list[Dataset]] = {"train": [], "dev": [], "test": []}

    for name, loader in loaders:
        try:
            splits = loader()
            for split_name in ("train", "dev", "test"):
                if split_name in splits:
                    per_split[split_name].append(splits[split_name])
        except Exception as e:
            logger.error(f"✗ Failed to load {name}: {e}")

    combined = {}
    for split_name, datasets in per_split.items():
        if datasets:
            merged = concatenate_datasets(datasets)
            if split_name == "train":
                merged = merged.shuffle(seed=42)
            combined[split_name] = merged
            logger.info(f"Combined {split_name}: {len(merged):,} samples")

    return DatasetDict(combined)


def print_summary(dd: DatasetDict) -> None:
    """Print a detailed summary of the combined dataset."""
    print("\n" + "=" * 80)
    print("COMBINED DATASET SUMMARY")
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


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Create combined Scandinavian embedding training dataset")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        default="thivy/scandinavian-embedding-training-data",
        help="HuggingFace repo ID (default: thivy/scandinavian-embedding-training-data)",
    )
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    parser.add_argument("--save-local", type=str, default=None, help="Save to local directory (Arrow format)")
    args = parser.parse_args()

    dd = combine_all()
    print_summary(dd)

    if args.save_local:
        logger.info(f"Saving to {args.save_local}...")
        dd.save_to_disk(args.save_local)
        logger.info("✓ Saved locally")

    if args.push:
        logger.info(f"Pushing to HuggingFace Hub: {args.repo_id} ...")
        dd.push_to_hub(
            args.repo_id,
            private=args.private,
        )
        logger.info(f"✓ Pushed to https://huggingface.co/datasets/{args.repo_id}")
    elif not args.save_local:
        logger.info("Dry run complete. Use --push to upload or --save-local to save.")


if __name__ == "__main__":
    main()
