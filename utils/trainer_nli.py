"""
NLI Trainer for Norwegian embedding models.

Specialized trainer for Natural Language Inference (NLI) datasets with triplet format.
"""

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset
from pathlib import Path
import logging
import torch
from transformers import EarlyStoppingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLITrainer:
    """
    Trainer specialized for NLI triplet datasets.

    Handles:
    - MultipleNegativesRankingLoss
    - TripletEvaluator (anchor, positive, negative)
    - Cosine accuracy metrics
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.loss = None
        self.trainer = None
        self.device = self._setup_device()

    def _setup_device(self) -> str:
        """Determine which device to use for training."""
        if hasattr(self.config, 'compute') and hasattr(self.config.compute, 'device'):
            device = self.config.compute.device
            if device == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    logger.info("✓ Using Apple Silicon GPU (MPS)")
                    return "mps"
                else:
                    return "cpu"
            return device

        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("✓ Using Apple Silicon GPU (MPS)")
            return "mps"
        return "cpu"

    def load_model(self) -> SentenceTransformer:
        """Load the pre-trained model."""
        logger.info(f"Loading base model: {self.config.model.name}")

        self.model = SentenceTransformer(
            self.config.model.name,
            device=self.device,
            trust_remote_code=True
        )

        if hasattr(self.config.model, 'max_seq_length'):
            self.model.max_seq_length = self.config.model.max_seq_length
            logger.info(f"✓ Set max sequence length to {self.config.model.max_seq_length}")

        logger.info("✓ Model loaded successfully!")
        logger.info(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        return self.model

    def setup_loss_function(self):
        """Set up MultipleNegativesRankingLoss for NLI triplet data."""
        if self.model is None:
            raise ValueError("Model must be loaded before setting up loss function")

        logger.info("Setting up loss function: MultipleNegativesRankingLoss")
        self.loss = MultipleNegativesRankingLoss(self.model)

        logger.info("✓ MultipleNegativesRankingLoss configured")
        logger.info("   This loss pulls similar sentences closer in embedding space")
        logger.info("   and pushes dissimilar sentences farther apart")

        return self.loss

    def setup_training_arguments(self) -> SentenceTransformerTrainingArguments:
        """Configure all training hyperparameters for NLI."""
        logger.info("Configuring training arguments...")

        # Learning rate scheduler
        lr_scheduler_type = "linear"
        lr_scheduler_kwargs = {}
        if hasattr(self.config, 'lr_scheduler') and self.config.lr_scheduler is not None:
            lr_scheduler_type = self.config.lr_scheduler.type
            if lr_scheduler_type == "cosine_with_restarts":
                lr_scheduler_kwargs = {
                    "num_cycles": self.config.lr_scheduler.num_cycles if hasattr(self.config.lr_scheduler, 'num_cycles') else 1
                }
            logger.info(f"Using LR scheduler: {lr_scheduler_type}")
        else:
            logger.info(f"Using default LR scheduler: {lr_scheduler_type}")

        # Experiment tracking
        report_to = []
        if hasattr(self.config, 'experiment_tracking'):
            report_to = self.config.experiment_tracking.report_to if hasattr(self.config.experiment_tracking, 'report_to') else []

        # Run name
        run_name = None
        if hasattr(self.config, 'experiment_tracking') and hasattr(self.config.experiment_tracking, 'mlflow_run_name'):
            run_name = self.config.experiment_tracking.mlflow_run_name
        if run_name is None:
            model_name = self.config.model.name.split('/')[-1]
            run_name = f"{model_name}-{self.config.training.num_train_epochs}ep-bs{self.config.training.per_device_train_batch_size}"

        # Hub config
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

        args = SentenceTransformerTrainingArguments(
            output_dir=self.config.training.output_dir,
            run_name=run_name,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs if lr_scheduler_kwargs else None,
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=self.config.training.eval_steps if hasattr(self.config.training, 'eval_steps') else None,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps if hasattr(self.config.training, 'save_steps') else None,
            save_total_limit=self.config.training.save_total_limit if hasattr(self.config.training, 'save_total_limit') else None,
            load_best_model_at_end=self.config.training.load_best_model_at_end if hasattr(self.config.training, 'load_best_model_at_end') else True,
            logging_steps=self.config.training.logging_steps,
            logging_dir=self.config.training.logging_dir if hasattr(self.config.training, 'logging_dir') else None,
            report_to=report_to,
            fp16=self.config.training.fp16 if hasattr(self.config.training, 'fp16') else False,
            bf16=self.config.training.bf16 if hasattr(self.config.training, 'bf16') else False,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps if hasattr(self.config.training, 'gradient_accumulation_steps') else 1,
            dataloader_num_workers=0 if self.device == "mps" else (
                self.config.training.dataloader_num_workers if hasattr(self.config.training, 'dataloader_num_workers') else 0
            ),
            seed=self.config.training.seed if hasattr(self.config.training, 'seed') else 42,
            metric_for_best_model=self.config.evaluation.metric_for_best_model if hasattr(self.config, 'evaluation') else None,
            greater_is_better=self.config.evaluation.greater_is_better if hasattr(self.config, 'evaluation') else True,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_strategy=hub_strategy,
            hub_token=hub_token,
            hub_private_repo=hub_private_repo,
        )

        logger.info("✓ Training arguments configured")
        return args

    def setup_evaluator(self, eval_dataset: Dataset) -> TripletEvaluator:
        """Set up TripletEvaluator for NLI data."""
        if eval_dataset is None:
            logger.warning("⚠ No evaluation dataset provided, skipping evaluator setup")
            return None

        logger.info("Setting up triplet evaluator...")

        anchors = eval_dataset[self.config.dataset.anchor_column]
        positives = eval_dataset[self.config.dataset.positive_column]
        negatives = eval_dataset[self.config.dataset.negative_column]

        evaluator = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name="eval"
        )

        logger.info(f"✓ Evaluator configured with {len(anchors):,} triplets")
        return evaluator

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None, resume_from_checkpoint: str = None):
        """Run the complete NLI training pipeline."""
        logger.info("\n" + "="*70)
        logger.info("STARTING NLI TRAINING")
        logger.info("="*70)

        # Setup MLflow if enabled
        if hasattr(self.config, 'experiment_tracking') and self.config.experiment_tracking.use_mlflow:
            try:
                import mlflow
                import os

                tracking_uri = (
                    self.config.experiment_tracking.mlflow_tracking_uri
                    if hasattr(self.config.experiment_tracking, 'mlflow_tracking_uri') and self.config.experiment_tracking.mlflow_tracking_uri
                    else "file:./mlruns"
                )
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.config.experiment_tracking.mlflow_experiment_name)

                if hasattr(self.config.experiment_tracking, 'mlflow_tags'):
                    import json
                    tags = self.config.experiment_tracking.mlflow_tags
                    if hasattr(tags, '__dict__'):
                        tags = vars(tags)
                    os.environ['MLFLOW_TAGS'] = json.dumps(tags)

                logger.info(f"✓ MLflow experiment tracking enabled: {self.config.experiment_tracking.mlflow_experiment_name}")
            except Exception as e:
                logger.warning(f"⚠ Failed to setup MLflow: {e}")

        # Ensure everything is set up
        if self.model is None:
            self.load_model()
        if self.loss is None:
            self.setup_loss_function()

        # Setup training args and evaluator
        args = self.setup_training_arguments()
        evaluator = self.setup_evaluator(eval_dataset) if eval_dataset else None

        # Setup callbacks
        from utils.embedding_metrics_callback import EmbeddingMetricsCallback

        custom_callbacks = [
            EmbeddingMetricsCallback(log_frequency=100),
        ]

        # Add early stopping if configured
        if hasattr(self.config.training, 'early_stopping_patience'):
            early_stopping_patience = self.config.training.early_stopping_patience
            early_stopping_threshold = getattr(self.config.training, 'early_stopping_threshold', 0.0)

            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
            custom_callbacks.append(early_stopping)

            logger.info(f"✓ Early stopping enabled:")
            logger.info(f"  - Patience: {early_stopping_patience} evaluations")
            logger.info(f"  - Threshold: {early_stopping_threshold}")

        # Create trainer
        logger.info("\nInitializing trainer...")

        self.trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=self.loss,
            evaluator=evaluator,
            callbacks=custom_callbacks,
        )

        logger.info("✓ Trainer initialized")
        logger.info(f"\nTraining configuration:")
        logger.info(f"  Training samples: {len(train_dataset):,}")
        if eval_dataset:
            logger.info(f"  Evaluation samples: {len(eval_dataset):,}")
        logger.info(f"  Batch size: {self.config.training.per_device_train_batch_size}")
        logger.info(f"  Epochs: {self.config.training.num_train_epochs}")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")

        # Train
        logger.info("\n" + "="*70)
        logger.info("BEGINNING TRAINING LOOP")
        logger.info("="*70)

        checkpoint = resume_from_checkpoint
        if checkpoint is None and hasattr(self.config.advanced, 'resume_from_checkpoint'):
            checkpoint = self.config.advanced.resume_from_checkpoint

        try:
            self.trainer.train(resume_from_checkpoint=checkpoint)
            logger.info("\n✓ Training completed successfully!")
        except Exception as e:
            logger.error(f"\n❌ Training failed with error: {e}")
            raise

    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        output_path = Path(self.config.training.output_dir)
        self.model.save(str(output_path))
        logger.info(f"✓ Model saved to: {output_path}")

    def push_to_hub(self):
        """Push model to HuggingFace Hub."""
        if not hasattr(self.config, 'advanced') or not self.config.advanced.push_to_hub:
            logger.info("Push to hub not enabled")
            return

        model_id = self.config.advanced.hub_model_id
        token = self.config.advanced.hub_token if hasattr(self.config.advanced, 'hub_token') else None

        logger.info(f"\nPushing model to HuggingFace Hub: {model_id}")
        try:
            try:
                self.model.push_to_hub(model_id, token=token, exist_ok=True)
                logger.info("✓ Model pushed to Hub successfully!")
                return
            except TypeError:
                pass

            self.model.push_to_hub(model_id, token=token)
            logger.info("✓ Model pushed to Hub successfully!")

        except Exception as e:
            if "already created this model repo" in str(e) or "409" in str(e):
                logger.warning(f"Repository {model_id} already exists, updating instead...")
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=self.config.training.output_dir,
                    repo_id=model_id,
                    repo_type="model",
                    token=token,
                    ignore_patterns=["checkpoint-*", "logs/*"],
                )
                logger.info("✓ Model updated on Hub successfully!")
            else:
                logger.error(f"Failed to push to hub: {e}")
                raise
