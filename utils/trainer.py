"""
Training pipeline for Norwegian embedding models.

This module orchestrates the entire training process:
1. Load the pre-trained model
2. Set up the loss function
3. Configure training arguments
4. Set up evaluation
5. Train the model
"""

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset
from pathlib import Path
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """
    Wrapper class for training embedding models.

    This class makes training cleaner by organizing all the components
    in one place with clear methods.
    """

    def __init__(self, config):
        """
        Initialize trainer with configuration.

        Args:
            config: Config object from read_config.py
        """
        self.config = config
        self.model = None
        self.loss = None
        self.trainer = None
        self.device = self._setup_device()

    def _setup_device(self) -> str:
        """
        Determine which device to use for training.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        device_config = self.config.compute.device

        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("✓ Using Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                logger.info("⚠ Using CPU (training will be slow)")
        else:
            device = device_config
            logger.info(f"✓ Using specified device: {device}")

        return device

    def load_model(self) -> SentenceTransformer:
        """
        Load the base model for fine-tuning.

        WHAT HAPPENS HERE:
        - Downloads the pre-trained model from HuggingFace (cached after first download)
        - The model is already trained on Norwegian text, we're just adapting it
        - Sets the maximum sequence length

        Returns:
            Loaded SentenceTransformer model
        """
        logger.info(f"Loading base model: {self.config.model.name}")

        try:
            self.model = SentenceTransformer(
                self.config.model.name,
                device=self.device,
                trust_remote_code=True  # Required for some Norwegian models like NorBERT
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load model '{self.config.model.name}'. "
                f"Please check that the model name is correct and you have internet access.\n"
                f"Error: {e}"
            )

        # Set maximum sequence length
        if hasattr(self.config.model, 'max_seq_length'):
            self.model.max_seq_length = self.config.model.max_seq_length
            logger.info(f"✓ Set max sequence length to {self.config.model.max_seq_length}")

        logger.info(f"✓ Model loaded successfully!")
        logger.info(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        return self.model

    def setup_loss_function(self) -> MultipleNegativesRankingLoss:
        """
        Set up the loss function for training.

        WHAT IS MultipleNegativesRankingLoss?
        - It's a contrastive learning loss function
        - For each (anchor, positive) pair in the batch, it:
          1. Pulls the anchor and positive closer together
          2. Pushes the anchor away from all negatives
        - It's very efficient because it uses other samples in the batch as extra negatives

        HOW IT WORKS:
        Batch: [(A1, P1, N1), (A2, P2, N2), (A3, P3, N3)]
        For A1:
          - Pull closer to P1 (its positive)
          - Push away from N1, N2, N3 (negatives from all samples)

        This is why larger batch sizes often work better!

        Returns:
            Configured loss function
        """
        logger.info("Setting up loss function: MultipleNegativesRankingLoss")

        if self.model is None:
            raise ValueError("Model must be loaded before setting up loss function")

        self.loss = MultipleNegativesRankingLoss(self.model)

        logger.info("✓ Loss function configured")
        logger.info("   This loss will make similar sentences closer in embedding space")
        logger.info("   and dissimilar sentences farther apart.")

        return self.loss

    def setup_training_arguments(self) -> SentenceTransformerTrainingArguments:
        """
        Configure all training hyperparameters.

        These arguments control:
        - How long to train (epochs)
        - How fast to learn (learning rate)
        - How often to save/evaluate
        - Performance optimizations

        Returns:
            Training arguments object
        """
        logger.info("Configuring training arguments...")

        # Learning rate scheduler configuration
        lr_scheduler_type = "linear"  # Default
        lr_scheduler_kwargs = {}
        if hasattr(self.config, 'lr_scheduler'):
            lr_scheduler_type = self.config.lr_scheduler.type
            if lr_scheduler_type == "cosine_with_restarts":
                lr_scheduler_kwargs = {
                    "num_cycles": self.config.lr_scheduler.num_cycles if hasattr(self.config.lr_scheduler, 'num_cycles') else 1
                }
            logger.info(f"Using LR scheduler: {lr_scheduler_type}")

        # Experiment tracking configuration
        report_to = []
        if hasattr(self.config, 'experiment_tracking'):
            report_to = self.config.experiment_tracking.report_to if hasattr(self.config.experiment_tracking, 'report_to') else []
            logger.info(f"Reporting to: {report_to}")

        # MLflow configuration
        run_name = None
        if hasattr(self.config, 'experiment_tracking') and hasattr(self.config.experiment_tracking, 'mlflow_run_name'):
            run_name = self.config.experiment_tracking.mlflow_run_name
        if run_name is None:
            # Auto-generate run name from model
            model_name = self.config.model.name.split('/')[-1]
            run_name = f"{model_name}-{self.config.training.num_train_epochs}ep-bs{self.config.training.per_device_train_batch_size}"

        # HuggingFace Hub configuration
        push_to_hub = False
        hub_model_id = None
        hub_strategy = "end"
        hub_token = None
        hub_private_repo = False

        if hasattr(self.config, 'advanced'):
            push_to_hub = self.config.advanced.push_to_hub if hasattr(self.config.advanced, 'push_to_hub') else False
            hub_model_id = self.config.advanced.hub_model_id if hasattr(self.config.advanced, 'hub_model_id') else None
            hub_strategy = self.config.advanced.hub_strategy if hasattr(self.config.advanced, 'hub_strategy') else "end"
            hub_token = self.config.advanced.hub_token if hasattr(self.config.advanced, 'hub_token') else None
            hub_private_repo = self.config.advanced.hub_private_repo if hasattr(self.config.advanced, 'hub_private_repo') else False

            if push_to_hub:
                logger.info(f"HuggingFace Hub push enabled: {hub_model_id}")
                logger.info(f"  Push strategy: {hub_strategy}")

        args = SentenceTransformerTrainingArguments(
            # Output
            output_dir=self.config.training.output_dir,
            run_name=run_name,  # For MLflow/W&B

            # Training duration
            num_train_epochs=self.config.training.num_train_epochs,

            # Batch sizes
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,

            # Learning rate & optimization
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,

            # Learning rate scheduler
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs if lr_scheduler_kwargs else None,

            # Evaluation
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=self.config.training.eval_steps if hasattr(self.config.training, 'eval_steps') else None,

            # Saving
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps if hasattr(self.config.training, 'save_steps') else None,
            save_total_limit=self.config.training.save_total_limit if hasattr(self.config.training, 'save_total_limit') else None,
            load_best_model_at_end=self.config.training.load_best_model_at_end if hasattr(self.config.training, 'load_best_model_at_end') else True,

            # Logging
            logging_steps=self.config.training.logging_steps,
            logging_dir=self.config.training.logging_dir if hasattr(self.config.training, 'logging_dir') else None,
            report_to=report_to,  # MLflow, TensorBoard, etc.

            # Performance
            fp16=self.config.training.fp16 if hasattr(self.config.training, 'fp16') else False,
            bf16=self.config.training.bf16 if hasattr(self.config.training, 'bf16') else False,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps if hasattr(self.config.training, 'gradient_accumulation_steps') else 1,
            # IMPORTANT: MPS (Apple Silicon) doesn't support multiprocessing data loaders
            # Set to 0 for MPS to avoid "_share_filename_: only available on CPU" error
            dataloader_num_workers=0 if self.device == "mps" else (
                self.config.training.dataloader_num_workers if hasattr(self.config.training, 'dataloader_num_workers') else 0
            ),

            # Reproducibility
            seed=self.config.training.seed if hasattr(self.config.training, 'seed') else 42,

            # Evaluation metric
            metric_for_best_model=self.config.evaluation.metric_for_best_model if hasattr(self.config, 'evaluation') else None,
            greater_is_better=self.config.evaluation.greater_is_better if hasattr(self.config, 'evaluation') else True,

            # HuggingFace Hub integration
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_strategy=hub_strategy,
            hub_token=hub_token,
            hub_private_repo=hub_private_repo,
        )

        logger.info("✓ Training arguments configured")
        return args

    def setup_evaluator(self, eval_dataset: Dataset) -> TripletEvaluator:
        """
        Set up evaluator to measure model performance during training.

        WHAT DOES THE EVALUATOR DO?
        - Takes triplets from the evaluation set
        - Computes embeddings for anchor, positive, and negative
        - Checks if distance(anchor, positive) < distance(anchor, negative)
        - Reports accuracy: how often the model gets this right

        This helps us track if the model is actually learning!

        Args:
            eval_dataset: Evaluation dataset with triplets

        Returns:
            Configured evaluator
        """
        if eval_dataset is None:
            logger.warning("⚠ No evaluation dataset provided, skipping evaluator setup")
            return None

        logger.info("Setting up triplet evaluator...")

        # Extract columns based on config
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

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        resume_from_checkpoint: str = None
    ):
        """
        Run the complete training pipeline.

        TRAINING PROCESS:
        1. Model processes batches of triplets
        2. Computes embeddings for anchor, positive, negative
        3. Loss function calculates how far we are from the goal
        4. Backpropagation updates model weights to reduce loss
        5. Repeat for all batches (= 1 epoch)
        6. Repeat for multiple epochs
        7. Periodically evaluate and save checkpoints

        Args:
            train_dataset: Training data
            eval_dataset: Evaluation data (optional but recommended)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)

        # Setup MLflow if enabled
        mlflow_active = False
        if hasattr(self.config, 'experiment_tracking') and self.config.experiment_tracking.use_mlflow:
            try:
                import mlflow
                import os

                # Set tracking URI
                tracking_uri = (
                    self.config.experiment_tracking.mlflow_tracking_uri
                    if hasattr(self.config.experiment_tracking, 'mlflow_tracking_uri') and self.config.experiment_tracking.mlflow_tracking_uri
                    else "file:./mlruns"
                )
                mlflow.set_tracking_uri(tracking_uri)

                # Set experiment
                experiment_name = self.config.experiment_tracking.mlflow_experiment_name
                mlflow.set_experiment(experiment_name)

                # Set tags if provided
                if hasattr(self.config.experiment_tracking, 'mlflow_tags'):
                    import json
                    tags = self.config.experiment_tracking.mlflow_tags
                    if hasattr(tags, '__dict__'):
                        tags = vars(tags)
                    # HuggingFace MLflow callback expects JSON string
                    os.environ['MLFLOW_TAGS'] = json.dumps(tags)

                mlflow_active = True
                logger.info(f"✓ MLflow experiment tracking enabled: {experiment_name}")
                logger.info(f"  Tracking URI: {mlflow.get_tracking_uri()}")
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

        # Setup custom callbacks for enhanced metrics logging
        from utils.embedding_metrics_callback import EmbeddingMetricsCallback

        custom_callbacks = [
            EmbeddingMetricsCallback(log_frequency=100),  # Log every 100 steps
            # Note: Similarity distribution analysis removed - too slow during training
            # Run as post-training analysis instead
        ]

        # Create trainer
        logger.info("\nInitializing trainer...")
        logger.info("✓ Custom metrics callbacks enabled:")
        logger.info("  - Gradient norms and weight updates")
        logger.info("  - Embedding space statistics")
        logger.info("  - Model health indicators")

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
        logger.info(f"  Evaluation samples: {len(eval_dataset):,}" if eval_dataset else "  Evaluation samples: None")
        logger.info(f"  Batch size: {args.per_device_train_batch_size}")
        logger.info(f"  Epochs: {args.num_train_epochs}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Output directory: {args.output_dir}")

        # Start training!
        logger.info("\n" + "="*70)
        logger.info("BEGINNING TRAINING LOOP")
        logger.info("="*70 + "\n")

        # Resume from checkpoint if specified
        checkpoint = resume_from_checkpoint or (
            self.config.advanced.resume_from_checkpoint
            if hasattr(self.config, 'advanced') and hasattr(self.config.advanced, 'resume_from_checkpoint')
            else None
        )

        try:
            self.trainer.train(resume_from_checkpoint=checkpoint)
        except KeyboardInterrupt:
            logger.warning("\n⚠ Training interrupted by user")
            logger.info("Saving checkpoint before exit...")
            self.save_model(f"{args.output_dir}/interrupted-checkpoint")
            raise
        except Exception as e:
            logger.error(f"\n❌ Training failed with error: {e}")
            raise

        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)

    def save_model(self, output_path: str = None):
        """
        Save the trained model.

        Args:
            output_path: Where to save (defaults to config output_dir)
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = output_path or self.config.training.output_dir
        logger.info(f"\nSaving model to: {save_path}")

        self.model.save(save_path)
        logger.info("✓ Model saved successfully!")

        # Also save config for reproducibility
        config_save_path = Path(save_path) / "training_config.yaml"
        import yaml
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        logger.info(f"✓ Config saved to: {config_save_path}")

    def push_to_hub(self, model_id: str = None, token: str = None):
        """
        Push trained model to HuggingFace Hub.

        Args:
            model_id: Hub model ID (e.g., "username/model-name")
            token: HuggingFace API token
        """
        if self.model is None:
            raise ValueError("No model to push")

        model_id = model_id or (
            self.config.advanced.hub_model_id
            if hasattr(self.config, 'advanced')
            else None
        )
        token = token or (
            self.config.advanced.hub_token
            if hasattr(self.config, 'advanced')
            else None
        )

        if not model_id:
            raise ValueError("model_id must be provided")

        logger.info(f"\nPushing model to HuggingFace Hub: {model_id}")
        try:
            # First try with exist_ok parameter (newer versions of sentence-transformers)
            try:
                self.model.push_to_hub(model_id, token=token, exist_ok=True)
                logger.info("✓ Model pushed to Hub successfully!")
                return
            except TypeError:
                # exist_ok not supported, try without it
                pass

            # Try regular push
            self.model.push_to_hub(model_id, token=token)
            logger.info("✓ Model pushed to Hub successfully!")

        except Exception as e:
            # If the error is about repo already existing, upload directly
            if "already created this model repo" in str(e) or "409" in str(e):
                logger.warning(f"Repository {model_id} already exists, updating instead...")
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=self.config.training.output_dir,
                    repo_id=model_id,
                    repo_type="model",
                    token=token,
                    ignore_patterns=["checkpoint-*", "logs/*"],  # Don't upload checkpoints/logs
                )
                logger.info("✓ Model updated on Hub successfully!")
            else:
                logger.error(f"Failed to push to hub: {e}")
                raise
