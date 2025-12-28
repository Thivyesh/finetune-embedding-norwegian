"""
Main training script for Norwegian embedding model fine-tuning.

This is the entry point for training. It orchestrates:
1. Loading configuration
2. Loading data
3. Training the model
4. Saving the results

Usage:
    python main.py                                    # Use default config
    python main.py --config configs/custom.yaml       # Use custom config
    python main.py --quick-test                       # Quick test with small dataset
"""

import argparse
import logging
from pathlib import Path

from utils.read_config import load_config, validate_config
from utils.data_loader import load_triplet_dataset
from utils.trainer import EmbeddingTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Norwegian embedding models on triplet data"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file (default: configs/training_config.yaml)"
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with limited samples (overrides config limits)"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    return parser.parse_args()


def main():
    """
    Main training pipeline.

    WORKFLOW:
    1. Parse arguments and load config
    2. Validate configuration
    3. Load and prepare datasets
    4. Initialize trainer
    5. Train model
    6. Save results
    """
    # Banner
    logger.info("\n" + "="*70)
    logger.info("NORWEGIAN EMBEDDING MODEL FINE-TUNING")
    logger.info("="*70 + "\n")

    # Parse arguments
    args = parse_args()

    # Load and validate configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)

    # Override for quick testing
    if args.quick_test:
        logger.info("\n⚡ QUICK TEST MODE: Using limited samples")
        config.dataset.max_train_samples = 1000
        config.dataset.max_eval_samples = 200
        config.training.num_train_epochs = 1
        config.training.eval_steps = 50
        config.training.save_steps = 50
        config.training.logging_steps = 10

    # Load datasets
    logger.info("\n" + "-"*70)
    logger.info("LOADING DATASETS")
    logger.info("-"*70)

    train_dataset, eval_dataset, test_dataset = load_triplet_dataset(
        dataset_name=config.dataset.name,
        train_split=config.dataset.train_split,
        eval_split=config.dataset.eval_split,
        test_split=config.dataset.test_split,
        anchor_column=config.dataset.anchor_column,
        positive_column=config.dataset.positive_column,
        negative_column=config.dataset.negative_column,
        max_train_samples=config.dataset.max_train_samples,
        max_eval_samples=config.dataset.max_eval_samples,
    )

    # Initialize trainer
    logger.info("\n" + "-"*70)
    logger.info("INITIALIZING TRAINER")
    logger.info("-"*70)

    trainer = EmbeddingTrainer(config)

    # Load model and setup training
    trainer.load_model()
    trainer.setup_loss_function()

    # Train!
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Save final model
    logger.info("\n" + "-"*70)
    logger.info("SAVING FINAL MODEL")
    logger.info("-"*70)

    trainer.save_model()

    # Push to hub if configured
    if hasattr(config, 'advanced') and config.advanced.push_to_hub:
        logger.info("\n" + "-"*70)
        logger.info("PUSHING TO HUGGINGFACE HUB")
        logger.info("-"*70)
        trainer.push_to_hub()

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✓ ALL DONE!")
    logger.info("="*70)
    logger.info(f"\nYour fine-tuned model is saved at:")
    logger.info(f"  {Path(config.training.output_dir).absolute()}")
    logger.info(f"\nTo use the model:")
    logger.info(f"  from sentence_transformers import SentenceTransformer")
    logger.info(f"  model = SentenceTransformer('{config.training.output_dir}')")
    logger.info(f"  embeddings = model.encode(['Din tekst her', 'Mer tekst'])")
    logger.info("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Training interrupted by user (Ctrl+C)")
        logger.info("Partial results may be saved in checkpoints.")
    except Exception as e:
        logger.error(f"\n\n❌ Training failed with error:")
        logger.error(f"  {type(e).__name__}: {e}")
        raise
