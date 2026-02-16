#!/usr/bin/env python3
"""
Train EmbeddingGemma-300M on ETI triplet data for Norwegian health information retrieval.

This script uses triplet loss with hard negatives for more effective contrastive learning
compared to the previous MultipleNegativesRankingLoss approach.

Key Features:
- TripletLoss with hard negatives 
- Single dataset (ETI triplets only)
- Conservative learning rate for stability
- Comprehensive evaluation (Triplet + Information Retrieval)
"""

import argparse
import logging
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# SentenceTransformers imports
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from datasets import concatenate_datasets

# Project imports
from utils.read_config import load_config
from utils.data_loader_eti_triplets import load_eti_triplet_data


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def setup_hardware(config):
    """Setup hardware optimizations."""
    logger = logging.getLogger(__name__)
    
    # Device detection
    device = getattr(config.hardware, 'device', 'auto')
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # CUDA optimizations
    if device == 'cuda':
        # Enable TF32 for faster training on Ampere+ GPUs
        if getattr(config.hardware, 'tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ TF32 enabled for faster training")
        
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        logger.info("✓ cuDNN benchmark enabled")
        
        # GPU memory info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return device


def load_triplet_data(config):
    """
    Load ETI triplet dataset.
    
    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset) as HF Dataset objects
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*70)
    logger.info("LOADING TRIPLET DATASET")  
    logger.info("="*70)
    
    if not getattr(config.dataset, 'use_eti_triplets', True):
        raise ValueError("ETI triplets must be enabled for triplet training!")
    
    # Load ETI triplet data - returns HF Dataset objects directly
    logger.info("Loading ETI triplet dataset...")
    eti_train, eti_dev, eti_test = load_eti_triplet_data()
    
    # Log sample triplets
    logger.info(f"Sample triplets from training set:")
    for i in range(min(3, len(eti_train))):
        item = eti_train[i]
        logger.info(f"  Example {i+1}:")
        logger.info(f"    Anchor: {item['anchor'][:100]}...")
        logger.info(f"    Positive: {item['positive'][:100]}...")
        logger.info(f"    Negative: {item['negative'][:100]}...")
    
    logger.info(f"\n✓ Total triplet samples: {len(eti_train):,} train, {len(eti_dev):,} dev, {len(eti_test):,} test")
    logger.info(f"✓ Columns: {eti_train.column_names}")
    
    # Return Dataset objects directly (NOT InputExample lists)
    return eti_train, eti_dev, eti_test


def create_evaluators(config, dev_dataset, test_dataset):
    """Create evaluation functions for triplet training."""
    logger = logging.getLogger(__name__)
    
    evaluators = []
    
    # 1. Triplet Evaluator (primary for triplet loss)
    logger.info("Setting up TripletEvaluator...")
    
    # Sample dev set for speed (max 1000)
    eval_sample_size = min(1000, len(dev_dataset))
    dev_sample = dev_dataset.shuffle(seed=42).select(range(eval_sample_size))
    
    triplet_evaluator = TripletEvaluator(
        anchors=[item['anchor'] for item in dev_sample],
        positives=[item['positive'] for item in dev_sample],
        negatives=[item['negative'] for item in dev_sample],
        name="dev_triplet",
        show_progress_bar=True,
    )
    evaluators.append(triplet_evaluator)
    
    # 2. Information Retrieval Evaluator (for comparison with previous results)
    if getattr(config.evaluation, 'include_ir_evaluation', True):
        logger.info("Setting up InformationRetrievalEvaluator...")
        
        # Use the same dev_sample for IR evaluation
        queries = {}
        corpus = {}
        relevant = {}
        
        for idx, item in enumerate(dev_sample):
            qid = f"q{idx}"
            did = f"d{idx}"
            queries[qid] = item['anchor']
            corpus[did] = item['positive']
            relevant[qid] = {did}
        
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant,
            name="dev_ir",
            show_progress_bar=True,
        )
        evaluators.append(ir_evaluator)
    
    logger.info(f"✓ Created {len(evaluators)} evaluators")
    return evaluators, test_dataset


def train_triplet_model(config_path: str, test_mode: bool = False, test_samples: int = 50):
    """Train the triplet model."""
    logger = setup_logging()
    
    logger.info("\n" + "="*70)
    logger.info("EMBEDDINGGEMMA-300M TRIPLET FINE-TUNING")
    if test_mode:
        logger.info(f"*** TEST MODE: Using only {test_samples} samples ***")
    logger.info("="*70)
    logger.info(f"Config: {config_path}")
    logger.info(f"Start time: {datetime.now()}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup MLflow tracking from config (before training starts)
    import os
    if hasattr(config, 'experiment_tracking'):
        mlflow_uri = getattr(config.experiment_tracking, 'mlflow_tracking_uri', 'file:./mlruns')
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        logger.info(f"Set MLflow tracking URI from config: {mlflow_uri}")
    elif 'azureml' in os.environ.get('MLFLOW_TRACKING_URI', '').lower():
        os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'
        logger.info("Switched to local MLflow tracking: file:./mlruns")
    
    # Setup hardware
    device = setup_hardware(config)
    
    # Load model
    logger.info(f"\nLoading base model: {config.model.model_name}")
    model = SentenceTransformer(config.model.model_name, device=device)
    logger.info(f"✓ Model loaded: {model}")
    
    # Load triplet data (returns HF Dataset objects)
    train_dataset, dev_dataset, test_dataset = load_triplet_data(config)
    
    # In test mode, reduce dataset sizes
    if test_mode:
        train_dataset = train_dataset.select(range(min(test_samples, len(train_dataset))))
        dev_dataset = dev_dataset.select(range(min(test_samples, len(dev_dataset))))
        test_dataset = test_dataset.select(range(min(test_samples, len(test_dataset))))
        logger.info(f"Test mode: train={len(train_dataset)}, dev={len(dev_dataset)}, test={len(test_dataset)}")
    
    # Create loss function
    logger.info("\nSetting up TripletLoss...")
    triplet_margin = float(getattr(config.training, 'triplet_margin', 0.5))
    distance_metric_name = getattr(config.training, 'distance_metric', 'cosine')
    
    # Map string to TripletDistanceMetric function
    if distance_metric_name == 'cosine':
        distance_metric = TripletDistanceMetric.COSINE
    elif distance_metric_name == 'euclidean':
        distance_metric = TripletDistanceMetric.EUCLIDEAN
    else:
        distance_metric = TripletDistanceMetric.COSINE
        logger.warning(f"Unknown distance metric '{distance_metric_name}', defaulting to cosine")
    
    loss_function = TripletLoss(
        model=model,
        distance_metric=distance_metric,
        triplet_margin=triplet_margin
    )
    logger.info(f"✓ TripletLoss: margin={triplet_margin}, distance={distance_metric_name}")
    
    # Create evaluators
    evaluators, test_dataset = create_evaluators(config, dev_dataset, test_dataset)
    
    # Setup training arguments
    logger.info("\nConfiguring training arguments...")
    
    args = SentenceTransformerTrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=float(config.training.num_train_epochs),
        per_device_train_batch_size=int(config.training.per_device_train_batch_size),
        per_device_eval_batch_size=int(getattr(config.training, 'per_device_eval_batch_size', config.training.per_device_train_batch_size * 2)),
        learning_rate=float(config.training.learning_rate),
        warmup_ratio=float(getattr(config.training, 'warmup_ratio', 0.1)),
        weight_decay=float(getattr(config.training, 'weight_decay', 0.01)),
        lr_scheduler_type=getattr(config.training, 'lr_scheduler_type', 'linear'),
        
        # Evaluation
        eval_strategy=getattr(config.training, 'eval_strategy', 'steps'),
        eval_steps=getattr(config.training, 'eval_steps', 500),
        
        # Saving
        save_strategy=getattr(config.training, 'save_strategy', 'steps'),
        save_steps=getattr(config.training, 'save_steps', 500),
        save_total_limit=getattr(config.training, 'save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model=getattr(config.training, 'metric_for_best_model', 'eval_dev_triplet_cosine_accuracy'),
        greater_is_better=True,
        
        # Logging
        logging_steps=getattr(config.training, 'logging_steps', 50),
        logging_dir=getattr(config.training, 'logging_dir', None),
        report_to=getattr(config.experiment_tracking, 'report_to', []) if hasattr(config, 'experiment_tracking') else [],
        
        # Performance
        fp16=getattr(config.training, 'fp16', False),
        bf16=getattr(config.training, 'bf16', False),
        gradient_accumulation_steps=getattr(config.training, 'gradient_accumulation_steps', 1),
        gradient_checkpointing=getattr(config.hardware, 'gradient_checkpointing', False) if hasattr(config, 'hardware') else False,
        dataloader_num_workers=0 if device == "mps" else 0,  # MPS doesn't support multiprocessing
        
        # Batch sampling
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        
        # Reproducibility
        seed=42,
    )
    
    # Create trainer
    logger.info("\nInitializing SentenceTransformerTrainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=loss_function,
        evaluator=evaluators[0] if len(evaluators) == 1 else evaluators,
    )
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRIPLET TRAINING")
    logger.info("="*70)
    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(dev_dataset):,}")
    logger.info(f"Epochs: {config.training.num_train_epochs}")
    logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Loss: TripletLoss (margin={triplet_margin})")
    
    # Train the model
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config.training.output_dir, "final")
    logger.info(f"\nSaving final model to: {final_model_path}")
    model.save(final_model_path)
    logger.info("✓ Model saved successfully!")
    
    # Test evaluation
    if getattr(config.evaluation, 'run_test_evaluation', True):
        logger.info("\n" + "="*70)
        logger.info("TEST EVALUATION")
        logger.info("="*70)
        
        # Sample test set for evaluation
        test_sample_size = min(1000, len(test_dataset))
        test_sample = test_dataset.shuffle(seed=42).select(range(test_sample_size))
        
        # Triplet test evaluation
        logger.info("Evaluating on test set with TripletEvaluator...")
        test_triplet_evaluator = TripletEvaluator(
            anchors=[item['anchor'] for item in test_sample],
            positives=[item['positive'] for item in test_sample], 
            negatives=[item['negative'] for item in test_sample],
            name="test_triplet",
            show_progress_bar=True,
        )
        
        test_triplet_results = test_triplet_evaluator(model)
        logger.info(f"✓ Test triplet evaluation complete!")
        logger.info(f"Test Triplet Accuracy: {test_triplet_results.get('test_triplet_cosine_accuracy', 'N/A')}")
        
        # IR test evaluation for comparison
        if getattr(config.evaluation, 'include_ir_evaluation', True):
            logger.info("Evaluating on test set with InformationRetrievalEvaluator...")
            
            test_queries = {}
            test_corpus = {}
            test_relevant = {}
            
            for idx, item in enumerate(test_sample):
                qid = f"q{idx}"
                did = f"d{idx}"
                test_queries[qid] = item['anchor']
                test_corpus[did] = item['positive']
                test_relevant[qid] = {did}
            
            test_ir_evaluator = InformationRetrievalEvaluator(
                queries=test_queries,
                corpus=test_corpus,
                relevant_docs=test_relevant,
                name="test_ir",
                show_progress_bar=True,
            )
            
            test_ir_results = test_ir_evaluator(model)
            logger.info(f"✓ Test IR evaluation complete!")
            logger.info(f"Test NDCG@10: {test_ir_results.get('test_ir_cosine_ndcg@10', 'N/A')}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"✓ Trained on {len(train_dataset):,} triplet examples")
    logger.info(f"✓ TripletLoss with {distance_metric} distance and margin {triplet_margin}")  
    logger.info(f"✓ Model saved to: {final_model_path}")
    logger.info("\nNext steps:")
    logger.info("1. Compare results with previous pairs-based training")
    logger.info("2. Test on your RAG pipeline")
    logger.info("3. Consider hyperparameter tuning if results are promising")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train EmbeddingGemma-300M with triplet loss")
    parser.add_argument(
        "--config", 
        default="configs/training_config_triplet_embeddinggemma.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with reduced samples"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=50,
        help="Number of samples to use in test mode (default: 50)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    train_triplet_model(args.config, test_mode=args.test, test_samples=args.test_samples)


if __name__ == "__main__":
    main()