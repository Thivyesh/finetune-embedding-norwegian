"""
Pre-training verification script.

This script checks that everything is set up correctly before training:
1. Config file is valid
2. All dependencies are installed
3. Dataset can be loaded
4. Model can be loaded
5. Device is configured correctly

Run this before starting training to catch issues early!

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.read_config import load_config, validate_config
from utils.data_loader import inspect_dataset, load_triplet_dataset
from utils.trainer import EmbeddingTrainer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_imports():
    """Verify all required dependencies are installed."""
    logger.info("="*70)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*70)

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("yaml", "PyYAML"),
    ]

    all_installed = True
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - NOT INSTALLED")
            all_installed = False

    if not all_installed:
        logger.error("\n❌ Missing dependencies. Run: uv sync")
        return False

    logger.info("\n✓ All dependencies installed!\n")
    return True


def check_config():
    """Verify config file exists and is valid."""
    logger.info("="*70)
    logger.info("CHECKING CONFIGURATION")
    logger.info("="*70)

    config_path = "configs/training_config.yaml"

    if not Path(config_path).exists():
        logger.error(f"✗ Config file not found: {config_path}")
        return False

    try:
        config = load_config(config_path)
        validate_config(config)
        logger.info(f"✓ Config loaded and validated successfully")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Dataset: {config.dataset.name}")
        logger.info(f"  Output: {config.training.output_dir}\n")
        return True, config
    except Exception as e:
        logger.error(f"✗ Config validation failed: {e}")
        return False, None


def check_dataset(config):
    """Verify dataset can be loaded."""
    logger.info("="*70)
    logger.info("CHECKING DATASET")
    logger.info("="*70)

    try:
        # First inspect the dataset
        inspect_dataset(config.dataset.name)

        # Try loading a small sample
        logger.info("\nTrying to load a small sample...")
        train_ds, eval_ds, test_ds = load_triplet_dataset(
            dataset_name=config.dataset.name,
            train_split=config.dataset.train_split,
            eval_split=config.dataset.eval_split,
            test_split=config.dataset.test_split,
            anchor_column=config.dataset.anchor_column,
            positive_column=config.dataset.positive_column,
            negative_column=config.dataset.negative_column,
            max_train_samples=10,
            max_eval_samples=10,
        )

        logger.info(f"\n✓ Dataset loaded successfully!")
        logger.info(f"  Sample train size: {len(train_ds)}")
        logger.info(f"  Sample eval size: {len(eval_ds)}\n")
        return True
    except Exception as e:
        logger.error(f"\n✗ Dataset loading failed: {e}")
        return False


def check_model(config):
    """Verify model can be loaded."""
    logger.info("="*70)
    logger.info("CHECKING MODEL")
    logger.info("="*70)

    try:
        trainer = EmbeddingTrainer(config)
        logger.info(f"Device detected: {trainer.device}")

        logger.info(f"\nLoading model: {config.model.name}")
        logger.info("(This may take a few minutes on first run...)")

        model = trainer.load_model()

        logger.info(f"\n✓ Model loaded successfully!")
        logger.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        logger.info(f"  Max sequence length: {model.max_seq_length}\n")

        # Test encoding
        logger.info("Testing model encoding...")
        test_sentence = "Dette er en test."
        embedding = model.encode(test_sentence)
        logger.info(f"✓ Successfully encoded test sentence")
        logger.info(f"  Embedding shape: {embedding.shape}\n")

        return True
    except Exception as e:
        logger.error(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_output_directory(config):
    """Verify output directory can be created."""
    logger.info("="*70)
    logger.info("CHECKING OUTPUT DIRECTORY")
    logger.info("="*70)

    output_dir = Path(config.training.output_dir)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Output directory ready: {output_dir.absolute()}\n")
        return True
    except Exception as e:
        logger.error(f"✗ Cannot create output directory: {e}\n")
        return False


def main():
    """Run all verification checks."""
    logger.info("\n" + "="*70)
    logger.info("PRE-TRAINING VERIFICATION")
    logger.info("="*70 + "\n")

    checks = []

    # Check 1: Dependencies
    checks.append(("Dependencies", check_imports()))

    # Check 2: Config
    config_result = check_config()
    if isinstance(config_result, tuple):
        success, config = config_result
        checks.append(("Configuration", success))
    else:
        checks.append(("Configuration", False))
        config = None

    if config:
        # Check 3: Dataset
        checks.append(("Dataset", check_dataset(config)))

        # Check 4: Model
        checks.append(("Model", check_model(config)))

        # Check 5: Output directory
        checks.append(("Output Directory", check_output_directory(config)))

    # Summary
    logger.info("="*70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*70)

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
        if not passed:
            all_passed = False

    logger.info("="*70)

    if all_passed:
        logger.info("\n🎉 ALL CHECKS PASSED!")
        logger.info("\nYou're ready to start training! Run:")
        logger.info("  python main.py --quick-test    # Quick test first")
        logger.info("  python main.py                 # Full training\n")
        return 0
    else:
        logger.error("\n❌ SOME CHECKS FAILED")
        logger.error("\nPlease fix the issues above before training.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
