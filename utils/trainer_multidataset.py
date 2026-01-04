"""
Multi-dataset trainer for Norwegian embedding models.

Combines multiple datasets (NLI, Scandi QA QA, DDSC DDSC) simultaneously
using ROUND_ROBIN or PROPORTIONAL sampling to prevent catastrophic forgetting.

Based on: https://huggingface.co/blog/train-sentence-transformers
"""

import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments,
    MultiDatasetBatchSamplers,
)
from sentence_transformers import SentenceTransformerTrainer
from transformers import TrainerCallback

from utils.read_config import Config
from utils.data_loader_nli import load_triplet_dataset
from utils.data_loader_scandi_qa import load_scandi_qa_data  # Multi-source QA datasets
from utils.data_loader_ddsc import load_ddsc_data  # DDSC retrieval data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AverageLossCallback(TrainerCallback):
    """
    Callback to compute average loss across all datasets.
    
    This creates an 'eval_avg_loss' metric that averages the individual
    dataset losses: eval_nli_loss, eval_group-b-qa_loss, eval_ddsc_loss
    """
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """After logging, compute average loss across all datasets if eval metrics are present."""
        if logs is None:
            return
        
        # Only process if we have evaluation metrics (check for any eval_*_loss)
        has_eval_metrics = any(key.startswith('eval_') and key.endswith('_loss') for key in logs)
        if not has_eval_metrics:
            return
        
        # Extract individual dataset losses
        dataset_losses = []
        loss_keys = ['eval_nli_loss', 'eval_group-b-qa_loss', 'eval_ddsc_loss']
        
        for key in loss_keys:
            if key in logs:
                dataset_losses.append(logs[key])
        
        # Compute average if we have multiple losses (not just one dataset)
        if len(dataset_losses) > 1:
            avg_loss = sum(dataset_losses) / len(dataset_losses)
            logs['eval_avg_loss'] = avg_loss
            
            # Log for visibility
            logger.info(f"  Average loss across {len(dataset_losses)} datasets: {avg_loss:.4f}")


class MultiDatasetTrainer:
    """Trainer for multi-dataset Norwegian embedding model training."""

    def __init__(self, config: Config):
        """
        Initialize multi-dataset trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")


    def train(self):
        """Run multi-dataset training."""
        logger.info("\n" + "="*70)
        logger.info("MULTI-DATASET TRAINING - EXPERIMENT 1")
        logger.info("="*70)
        logger.info("Strategy: ROUND_ROBIN sampling")
        logger.info("Datasets: NLI + Scandi QA (QA) + DDSC (DDSC)")
        logger.info("="*70 + "\n")

        # Load all datasets (train + eval)
        logger.info("Loading NLI dataset (train + eval)...")
        nli_train, nli_eval, nli_test = load_triplet_dataset(
            dataset_name="Fremtind/all-nli-norwegian",
            train_split="train",
            eval_split="dev",
            test_split="test",
            anchor_column="anchor",
            positive_column="positive",
            negative_column="negative",
            max_train_samples=None,
            max_eval_samples=1000,  # Limit eval for speed
        )

        logger.info("Loading Scandi QA dataset (train + eval)...")
        # Get Scandi QA config if available, otherwise use defaults
        scandi_qa_config = getattr(self.config.datasets, 'group_b_qa', None)
        use_norquad = getattr(scandi_qa_config, 'use_norquad', True) if scandi_qa_config else True
        use_openbookqa = getattr(scandi_qa_config, 'use_openbookqa', True) if scandi_qa_config else True
        use_scandiqa = getattr(scandi_qa_config, 'use_scandiqa', False) if scandi_qa_config else False
        use_supervised_da = getattr(scandi_qa_config, 'use_supervised_da', True) if scandi_qa_config else True
        use_paws = getattr(scandi_qa_config, 'use_paws', False) if scandi_qa_config else False
        
        scandi_qa_train, scandi_qa_eval, scandi_qa_test = load_scandi_qa_data(
            use_norquad=use_norquad,
            use_openbookqa=use_openbookqa,
            use_scandiqa=use_scandiqa,
            use_supervised_da=use_supervised_da,
            use_paws=use_paws,
        )

        logger.info("Loading DDSC dataset (train + eval)...")
        # Get DDSC config if available, otherwise use defaults
        ddsc_config = getattr(self.config.datasets, 'group_a_ddsc', None)
        if ddsc_config:
            language_filter = getattr(ddsc_config, 'language_filter', ['norwegian'])
            # Ensure it's a list
            if isinstance(language_filter, str):
                language_filter = [language_filter]
        else:
            language_filter = ['norwegian']
        
        logger.info(f"  Languages: {', '.join(language_filter)}")
        ddsc_train, ddsc_eval, ddsc_test = load_ddsc_data(
            languages=language_filter,
            tasks=None,  # All tasks
            split_ratio=(0.98, 0.01, 0.01),
        )

        # Filter None values from all datasets (defensive check)
        logger.info("Filtering None values from all datasets...")

        # NLI: filter None in anchor, positive, or negative
        def has_valid_triplet(example):
            return (example.get('anchor') is not None and
                    example.get('positive') is not None and
                    example.get('negative') is not None)

        nli_train_size = len(nli_train)
        nli_train = nli_train.filter(has_valid_triplet)
        nli_filtered = nli_train_size - len(nli_train)
        if nli_filtered > 0:
            logger.info(f"  NLI train: Filtered {nli_filtered:,} samples with None values")

        nli_eval_size = len(nli_eval)
        nli_eval = nli_eval.filter(has_valid_triplet)
        nli_eval_filtered = nli_eval_size - len(nli_eval)
        if nli_eval_filtered > 0:
            logger.info(f"  NLI eval: Filtered {nli_eval_filtered:,} samples with None values")

        # Scandi QA: filter None in query or positive
        def has_valid_pair(example):
            return (example.get('query') is not None and
                    example.get('positive') is not None)

        scandi_qa_train_size = len(scandi_qa_train)
        scandi_qa_train = scandi_qa_train.filter(has_valid_pair)
        scandi_qa_filtered = scandi_qa_train_size - len(scandi_qa_train)
        if scandi_qa_filtered > 0:
            logger.info(f"  Scandi QA train: Filtered {scandi_qa_filtered:,} samples with None values")

        scandi_qa_eval_size = len(scandi_qa_eval)
        scandi_qa_eval = scandi_qa_eval.filter(has_valid_pair)
        scandi_qa_eval_filtered = scandi_qa_eval_size - len(scandi_qa_eval)
        if scandi_qa_eval_filtered > 0:
            logger.info(f"  Scandi QA eval: Filtered {scandi_qa_eval_filtered:,} samples with None values")

        # DDSC: filter None in query, positive, or negative (same as GroupATrainer line 227)
        def has_valid_triplet_with_hard_neg(example):
            return (example.get('query') is not None and
                    example.get('positive') is not None and
                    example.get('negative') is not None)

        # DDSC: Keep ALL samples (MultipleNegativesRankingLoss handles both with/without negatives)
        # No filtering needed - samples without negatives use in-batch negatives only
        logger.info(f"  DDSC: Keeping all {len(ddsc_train):,} train samples")
        logger.info(f"  DDSC: Samples with hard negatives: {sum(1 for x in ddsc_train if x.get('negative') is not None):,}")
        logger.info(f"  DDSC: Samples without negatives (in-batch only): {sum(1 for x in ddsc_train if x.get('negative') is None):,}")

        logger.info(f"  DDSC: Keeping all {len(ddsc_eval):,} eval samples")

        # Remove extra columns from DDSC (but keep all samples including those without negatives)
        ddsc_train_formatted = ddsc_train.remove_columns(['instruction', 'task', 'language'])
        ddsc_eval_formatted = ddsc_eval.remove_columns(['instruction', 'task', 'language'])

        # Filter out None values for model card widget (model card generation can't handle None)
        # But we keep them for training! The loss function handles None properly.
        def has_no_none_values(example):
            """Check if sample has any None values (for model card only)."""
            return all(v is not None for v in example.values())

        logger.info("\nFiltering datasets for model card widget (removing None values)...")
        nli_train_clean = nli_train.filter(has_no_none_values)
        scandi_qa_train_clean = scandi_qa_train.filter(has_no_none_values)
        ddsc_train_clean = ddsc_train_formatted.filter(has_no_none_values)
        
        logger.info(f"  NLI: {len(nli_train):,} → {len(nli_train_clean):,} (kept {len(nli_train_clean)/len(nli_train)*100:.1f}%)")
        logger.info(f"  Scandi QA: {len(scandi_qa_train):,} → {len(scandi_qa_train_clean):,} (kept {len(scandi_qa_train_clean)/len(scandi_qa_train)*100:.1f}%)")
        logger.info(f"  DDSC: {len(ddsc_train_formatted):,} → {len(ddsc_train_clean):,} (kept {len(ddsc_train_clean)/len(ddsc_train_formatted)*100:.1f}%)")

        # Create dictionary of training datasets (CLEANED for model card)
        train_datasets = {
            'nli': nli_train_clean,
            'group-b-qa': scandi_qa_train_clean,
            'ddsc': ddsc_train_clean,
        }

        # Create dictionary of evaluation datasets with MATCHING KEYS (also filtered)
        nli_eval_clean = nli_eval.filter(has_no_none_values)
        scandi_qa_eval_clean = scandi_qa_eval.filter(has_no_none_values)
        ddsc_eval_clean = ddsc_eval_formatted.filter(has_no_none_values)
        
        eval_datasets = {
            'nli': nli_eval_clean.shuffle(seed=42).select(range(min(1000, len(nli_eval_clean)))),
            'group-b-qa': scandi_qa_eval_clean.shuffle(seed=42).select(range(min(500, len(scandi_qa_eval_clean)))),
            'ddsc': ddsc_eval_clean.shuffle(seed=42).select(range(min(500, len(ddsc_eval_clean)))),
        }

        # Print dataset sizes
        logger.info("\n" + "="*70)
        logger.info("DATASET SUMMARY")
        logger.info("="*70)
        for name, dataset in train_datasets.items():
            logger.info(f"{name}: {len(dataset):,} samples")
        total_samples = sum(len(ds) for ds in train_datasets.values())
        logger.info(f"Total: {total_samples:,} samples")
        logger.info("="*70 + "\n")

        # Initialize model
        logger.info("Loading base model...")
        model = SentenceTransformer(
            self.config.model.base_model,
            device=self.device,
            trust_remote_code=True,
        )
        model.max_seq_length = self.config.model.max_seq_length
        logger.info(f"✓ Model loaded: {self.config.model.base_model}")
        logger.info(f"  Max sequence length: {model.max_seq_length}")
        logger.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Enable gradient checkpointing for memory efficiency
        # Trades ~20% compute for 30-50% memory reduction
        if hasattr(model[0].auto_model, 'gradient_checkpointing_enable'):
            model[0].auto_model.gradient_checkpointing_enable()
            logger.info("  ✓ Gradient checkpointing enabled (saves 30-50% memory)")
        logger.info("")

        # Create losses for each dataset
        train_losses = {
            'nli': losses.MultipleNegativesRankingLoss(model),
            'group-b-qa': losses.MultipleNegativesRankingLoss(model),
            'ddsc': losses.MultipleNegativesRankingLoss(model),
        }

        # Evaluation will use the eval_datasets dict

        # Set up training arguments
        output_dir = Path(self.config.experiment_tracking.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map sampling strategy string to enum
        sampler_map = {
            "ROUND_ROBIN": MultiDatasetBatchSamplers.ROUND_ROBIN,
            "PROPORTIONAL": MultiDatasetBatchSamplers.PROPORTIONAL,
        }
        sampler = sampler_map.get(
            self.config.training.multi_dataset_batch_sampler,
            MultiDatasetBatchSamplers.ROUND_ROBIN
        )

        # Hub configuration
        push_to_hub = False
        hub_model_id = None
        hub_strategy = "checkpoint"
        hub_token = None
        hub_private_repo = False

        if hasattr(self.config, 'advanced'):
            push_to_hub = getattr(self.config.advanced, 'push_to_hub', False)
            hub_model_id = getattr(self.config.advanced, 'hub_model_id', None)
            hub_strategy = getattr(self.config.advanced, 'hub_strategy', 'checkpoint')
            hub_token = getattr(self.config.advanced, 'hub_token', None)
            hub_private_repo = getattr(self.config.advanced, 'hub_private_repo', False)

        logger.info("\n" + "="*70)
        logger.info("TRAINING CONFIGURATION")
        logger.info("="*70)
        logger.info(f"Batch sampler: {self.config.training.multi_dataset_batch_sampler}")
        logger.info(f"Per-device batch size: {self.config.training.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.training.per_device_train_batch_size * self.config.training.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        logger.info(f"Epochs: {self.config.training.num_train_epochs}")
        logger.info(f"Early stopping patience: {self.config.training.early_stopping_patience}")
        if push_to_hub:
            logger.info(f"Push to Hub: {hub_model_id} (strategy: {hub_strategy})")
        logger.info("="*70 + "\n")

        args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            run_name=self.config.experiment_tracking.run_name,

            # Multi-dataset configuration
            multi_dataset_batch_sampler=sampler,

            # Batch size
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,

            # Training duration
            num_train_epochs=self.config.training.num_train_epochs,

            # Learning rate
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler.type,

            # Regularization
            weight_decay=self.config.training.weight_decay,

            # Mixed precision
            bf16=self.config.training.bf16,

            # Evaluation
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,

            # Early stopping
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,

            # Logging
            logging_dir=self.config.experiment_tracking.logging_dir,
            logging_steps=self.config.training.logging_steps,
            logging_first_step=self.config.training.logging_first_step,
            report_to=self.config.experiment_tracking.report_to,

            # HuggingFace Hub
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_strategy=hub_strategy,
            hub_token=hub_token,
            hub_private_repo=hub_private_repo,
        )

        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_datasets,  # Dict of training datasets
            eval_dataset=eval_datasets,     # Dict of eval datasets (matching keys)
            loss=train_losses,              # Dict of losses (matching keys)
            callbacks=[AverageLossCallback()],  # Add custom callback for average loss
        )

        # Check for existing checkpoints to resume from
        resume_from_checkpoint = None
        checkpoints = sorted([d for d in output_dir.glob("checkpoint-*") if d.is_dir()])
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_from_checkpoint = str(latest_checkpoint)
            logger.info(f"\n{'='*70}")
            logger.info(f"RESUMING FROM CHECKPOINT: {latest_checkpoint.name}")
            logger.info(f"{'='*70}\n")
        else:
            logger.info("\n" + "="*70)
            logger.info("STARTING TRAINING FROM SCRATCH")
            logger.info("="*70 + "\n")

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        final_model_path = output_dir / "final"
        model.save_pretrained(str(final_model_path))
        logger.info(f"\n✓ Training complete! Model saved to: {final_model_path}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total samples trained: {total_samples:,}")
        logger.info(f"Model saved to: {final_model_path}")
        logger.info(f"Checkpoints saved to: {output_dir}")
        logger.info("="*70 + "\n")

        logger.info("Next step: Run MTEB evaluation to compare with Stage 2 QA model")
        logger.info("Command: uv run python scripts/evaluate_mteb.py --model models/norbert4-base-multidataset-exp1/final\n")


def main():
    """Main entry point for multi-dataset training."""
    import sys
    from utils.read_config import load_config

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/training_config_exp1_multidataset_roundrobin.yaml"

    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    trainer = MultiDatasetTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
