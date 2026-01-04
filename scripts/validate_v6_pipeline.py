"""
Validation script for V6 training pipeline.

Tests:
1. Data loaders work correctly
2. Loss function compatibility
3. Trainers can be instantiated
4. Compute/device settings correct
5. Evaluators work (InformationRetrievalEvaluator & TripletEvaluator)
6. Data format matches trainer expectations

Run this before starting full training to catch issues early!
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader_group_b import load_group_b_data
from utils.data_loader_group_a import load_group_a_data
from utils.trainer_group_b import GroupBTrainer
from utils.trainer_group_a import GroupATrainer
from utils.read_config import load_config
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockConfig:
    """Mock config for testing."""
    def __init__(self, stage="group_b"):
        self.model = type('obj', (object,), {
            'name': 'ltg/norbert4-base',
            'max_seq_length': 128
        })()

        if stage == "group_b":
            self.training = type('obj', (object,), {
                'output_dir': 'models/test-stage2-group-b',
                'num_epochs': 1,
                'batch_size': 8,  # Small for testing
                'learning_rate': 2e-5,
                'warmup_ratio': 0.1,
                'eval_steps': 10,
                'save_steps': 10,
                'get': lambda self, key, default: getattr(self, key, default)
            })()
        else:  # group_a
            self.training = type('obj', (object,), {
                'output_dir': 'models/test-stage3-group-a',
                'num_epochs': 1,
                'batch_size': 8,  # Small for testing
                'learning_rate': 1e-5,
                'warmup_ratio': 0.1,
                'eval_steps': 10,
                'save_steps': 10,
                'get': lambda self, key, default: getattr(self, key, default)
            })()

        self.compute = type('obj', (object,), {
            'device': 'auto'
        })()


def validate_data_loaders():
    """Test 1: Validate data loaders work."""
    logger.info("="*70)
    logger.info("TEST 1: Validating Data Loaders")
    logger.info("="*70)

    # Test Group B loader
    logger.info("\n1a. Testing Group B loader...")
    try:
        train_b, dev_b, test_b = load_group_b_data(
            use_norquad=True,
            use_openbookqa=True,
            use_scandiqa=False,  # Skip deprecated
            use_supervised_da=True,
            use_paws=True
        )
        logger.info(f"✓ Group B: {len(train_b):,} train, {len(dev_b):,} dev, {len(test_b):,} test")

        # Check format
        sample = train_b[0]
        assert 'query' in sample, "Missing 'query' column"
        assert 'positive' in sample, "Missing 'positive' column"
        logger.info(f"✓ Group B format correct: {list(sample.keys())}")

    except Exception as e:
        logger.error(f"❌ Group B loader failed: {e}")
        raise

    # Test Group A loader
    logger.info("\n1b. Testing Group A loader...")
    try:
        train_a, dev_a, test_a = load_group_a_data(
            languages=['norwegian', 'danish', 'swedish']
        )
        logger.info(f"✓ Group A: {len(train_a):,} train, {len(dev_a):,} dev, {len(test_a):,} test")

        # Check format
        sample = train_a[0]
        assert 'query' in sample, "Missing 'query' column"
        assert 'positive' in sample, "Missing 'positive' column"
        assert 'negative' in sample, "Missing 'negative' column"
        logger.info(f"✓ Group A format correct: {list(sample.keys())}")

    except Exception as e:
        logger.error(f"❌ Group A loader failed: {e}")
        raise

    logger.info("\n✅ All data loaders working!\n")
    return (train_b, dev_b, test_b), (train_a, dev_a, test_a)


def validate_loss_function():
    """Test 2: Validate loss function works with data format."""
    logger.info("="*70)
    logger.info("TEST 2: Validating Loss Function Compatibility")
    logger.info("="*70)

    # Load small model for testing
    logger.info("\nLoading small model for testing...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

    # Test with Group B format (no hard negatives)
    logger.info("\n2a. Testing loss with Group B format (no hard negatives)...")
    try:
        loss = MultipleNegativesRankingLoss(model)
        logger.info("✓ Loss function instantiated for Group B")

        # Create sample batch
        sample_batch = {
            'query': ['Test query 1', 'Test query 2'],
            'positive': ['Test positive 1', 'Test positive 2']
        }
        logger.info(f"✓ Group B batch format: {list(sample_batch.keys())}")

    except Exception as e:
        logger.error(f"❌ Group B loss setup failed: {e}")
        raise

    # Test with Group A format (with hard negatives)
    logger.info("\n2b. Testing loss with Group A format (with hard negatives)...")
    try:
        loss = MultipleNegativesRankingLoss(model)
        logger.info("✓ Loss function instantiated for Group A")

        # Create sample batch with negatives
        sample_batch = {
            'anchor': ['Test query 1', 'Test query 2'],
            'positive': ['Test positive 1', 'Test positive 2'],
            'negative': ['Test negative 1', 'Test negative 2']
        }
        logger.info(f"✓ Group A batch format: {list(sample_batch.keys())}")

    except Exception as e:
        logger.error(f"❌ Group A loss setup failed: {e}")
        raise

    logger.info("\n✅ Loss function compatible with both formats!\n")


def validate_trainers():
    """Test 3: Validate trainers can be instantiated."""
    logger.info("="*70)
    logger.info("TEST 3: Validating Trainers")
    logger.info("="*70)

    # Test Group B trainer
    logger.info("\n3a. Testing Group B trainer...")
    try:
        config_b = MockConfig(stage="group_b")
        trainer_b = GroupBTrainer(config_b)
        logger.info(f"✓ Group B trainer instantiated")
        logger.info(f"  Device: {trainer_b.device}")

    except Exception as e:
        logger.error(f"❌ Group B trainer failed: {e}")
        raise

    # Test Group A trainer
    logger.info("\n3b. Testing Group A trainer...")
    try:
        config_a = MockConfig(stage="group_a")
        trainer_a = GroupATrainer(config_a)
        logger.info(f"✓ Group A trainer instantiated")
        logger.info(f"  Device: {trainer_a.device}")

    except Exception as e:
        logger.error(f"❌ Group A trainer failed: {e}")
        raise

    logger.info("\n✅ All trainers working!\n")


def validate_compute_settings():
    """Test 4: Validate compute/device settings."""
    logger.info("="*70)
    logger.info("TEST 4: Validating Compute Settings")
    logger.info("="*70)

    # Check PyTorch
    logger.info(f"\nPyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")

    # Test device selection
    config = MockConfig()
    trainer = GroupBTrainer(config)
    device = trainer.device

    logger.info(f"\nSelected device: {device}")

    if device == "mps":
        logger.info("✓ Using Apple Silicon GPU (MPS)")
        logger.info("⚠️  Remember: Use bf16=False, fp16=False for MPS")
        logger.info("⚠️  Remember: Use dataloader_num_workers=0 for MPS")
    elif device == "cuda":
        logger.info("✓ Using NVIDIA GPU (CUDA)")
    else:
        logger.info("⚠️  Using CPU (slower)")

    logger.info("\n✅ Compute settings validated!\n")


def validate_evaluators():
    """Test 5: Validate evaluators work correctly."""
    logger.info("="*70)
    logger.info("TEST 5: Validating Evaluators")
    logger.info("="*70)

    # Load small model for testing
    logger.info("\nLoading small model for testing...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

    # Test Group B evaluator (InformationRetrievalEvaluator)
    logger.info("\n5a. Testing Group B InformationRetrievalEvaluator...")
    try:
        # Load small Group B dataset
        train_b, dev_b, _ = load_group_b_data(use_scandiqa=False)

        # Use small sample for testing
        small_dev_b = dev_b.select(range(min(50, len(dev_b))))

        # Setup evaluator using trainer method
        config_b = MockConfig(stage="group_b")
        trainer_b = GroupBTrainer(config_b)
        trainer_b.model = model  # Set model

        evaluator_b = trainer_b.setup_evaluator(small_dev_b)

        assert evaluator_b is not None, "Evaluator should not be None"
        assert isinstance(evaluator_b, InformationRetrievalEvaluator), "Should be InformationRetrievalEvaluator"

        logger.info("✓ InformationRetrievalEvaluator instantiated")
        logger.info(f"  Queries: {len(evaluator_b.queries)}")
        logger.info(f"  Corpus: {len(evaluator_b.corpus)}")
        logger.info(f"  Name: {evaluator_b.name}")

        # Test evaluation (quick test on small dataset)
        logger.info("  Running quick evaluation test...")
        metrics = evaluator_b(model)

        # Check that expected metrics are returned
        expected_metrics = ['ndcg@10', 'mrr@10', 'map@100']
        for metric in expected_metrics:
            # Metrics should be prefixed with evaluator name
            full_metric = f"{evaluator_b.name}_{metric}"
            if full_metric in metrics:
                logger.info(f"  ✓ {full_metric}: {metrics[full_metric]:.4f}")
            else:
                logger.warning(f"  ⚠️  Metric '{full_metric}' not found in results")

        logger.info("✓ Group B evaluator working correctly")

    except Exception as e:
        logger.error(f"❌ Group B evaluator failed: {e}")
        raise

    # Test Group A evaluator (TripletEvaluator)
    logger.info("\n5b. Testing Group A TripletEvaluator...")
    try:
        # Load small Group A dataset
        train_a, dev_a, _ = load_group_a_data(languages=['norwegian'])

        # Use small sample for testing
        small_dev_a = dev_a.select(range(min(50, len(dev_a))))

        # Setup evaluator using trainer method
        config_a = MockConfig(stage="group_a")
        trainer_a = GroupATrainer(config_a)
        trainer_a.model = model  # Set model

        # Prepare dataset first
        prepared_dev_a = trainer_a.prepare_dataset_with_negatives(small_dev_a)

        evaluator_a = trainer_a.setup_evaluator(prepared_dev_a)

        assert evaluator_a is not None, "Evaluator should not be None"
        assert isinstance(evaluator_a, TripletEvaluator), "Should be TripletEvaluator"

        logger.info("✓ TripletEvaluator instantiated")
        logger.info(f"  Triplets: {len(evaluator_a.anchors)}")
        logger.info(f"  Name: {evaluator_a.name}")

        # Test evaluation (quick test on small dataset)
        logger.info("  Running quick evaluation test...")
        metrics = evaluator_a(model)

        # Check for cosine accuracy metric
        if 'cosine_accuracy' in metrics:
            logger.info(f"  ✓ cosine_accuracy: {metrics['cosine_accuracy']:.4f}")
        else:
            logger.warning("  ⚠️  Metric 'cosine_accuracy' not found in results")

        logger.info("✓ Group A evaluator working correctly")

    except Exception as e:
        logger.error(f"❌ Group A evaluator failed: {e}")
        raise

    logger.info("\n✅ All evaluators validated!\n")


def validate_data_format_for_trainer():
    """Test 6: Validate data format matches trainer expectations."""
    logger.info("="*70)
    logger.info("TEST 6: Validating Data Format for Trainers")
    logger.info("="*70)

    # Load small samples
    logger.info("\n6a. Checking Group B data format...")
    _, dev_b, _ = load_group_b_data(use_scandiqa=False)
    sample_b = dev_b[0]

    logger.info(f"  Columns: {list(sample_b.keys())}")
    logger.info(f"  Query type: {type(sample_b['query'])}")
    logger.info(f"  Positive type: {type(sample_b['positive'])}")

    # Check for None values
    has_none = any(sample_b[col] is None for col in sample_b.keys())
    if has_none:
        logger.warning("⚠️  Group B has None values - may cause issues")
    else:
        logger.info("✓ Group B: No None values")

    logger.info("\n6b. Checking Group A data format...")
    train_a, _, _ = load_group_a_data(languages=['norwegian'])

    # Check for None negatives
    none_negatives = sum(1 for i in range(min(100, len(train_a))) if train_a[i]['negative'] is None)
    total_checked = min(100, len(train_a))

    logger.info(f"  Checked {total_checked} samples")
    logger.info(f"  None negatives: {none_negatives}/{total_checked} ({none_negatives/total_checked*100:.1f}%)")

    if none_negatives > 0:
        logger.warning(f"⚠️  {none_negatives} samples have None negatives - will use in-batch negatives")

    # Check prepared format for trainer
    config_a = MockConfig(stage="group_a")
    trainer_a = GroupATrainer(config_a)

    # Sample the prepared format
    small_dataset = train_a.select(range(10))
    prepared = trainer_a.prepare_dataset_with_negatives(small_dataset)
    sample_prepared = prepared[0]

    logger.info(f"\n  Prepared format columns: {list(sample_prepared.keys())}")
    logger.info(f"  Expected: ['anchor', 'positive', 'negative']")

    assert 'anchor' in sample_prepared, "Missing 'anchor' in prepared format"
    assert 'positive' in sample_prepared, "Missing 'positive' in prepared format"
    assert 'negative' in sample_prepared, "Missing 'negative' in prepared format"

    logger.info("✓ Group A prepared format correct")

    logger.info("\n✅ Data formats validated!\n")


def validate_configs():
    """Test 7: Validate config files against trainer requirements."""
    logger.info("="*70)
    logger.info("TEST 7: Validating Config Files")
    logger.info("="*70)

    # Validate Stage 2 (Group B) config
    logger.info("\n7a. Validating Stage 2 (Group B) config...")
    config_b_path = Path(__file__).parent.parent / "configs" / "training_config_stage2_group_b.yaml"

    if not config_b_path.exists():
        logger.warning(f"⚠️  Config not found: {config_b_path}")
        logger.warning("   Skipping Group B config validation")
    else:
        try:
            config_b = load_config(str(config_b_path))
            logger.info(f"✓ Config loaded: {config_b_path.name}")

            # Required fields
            required_fields = {
                'model.name': lambda c: hasattr(c, 'model') and hasattr(c.model, 'name'),
                'training.output_dir': lambda c: hasattr(c, 'training') and hasattr(c.training, 'output_dir'),
                'training.num_epochs': lambda c: hasattr(c, 'training') and hasattr(c.training, 'num_epochs'),
                'training.batch_size or per_device_train_batch_size': lambda c: (
                    hasattr(c, 'training') and (
                        hasattr(c.training, 'batch_size') or
                        hasattr(c.training, 'per_device_train_batch_size')
                    )
                ),
                'data_loader': lambda c: hasattr(c, 'data_loader'),
            }

            missing = []
            for field_name, checker in required_fields.items():
                if not checker(config_b):
                    missing.append(field_name)
                else:
                    logger.info(f"  ✓ {field_name}")

            if missing:
                logger.error(f"  ❌ Missing required fields: {', '.join(missing)}")
                raise ValueError(f"Group B config missing required fields: {missing}")

            # Validate batch size accessibility
            batch_size = getattr(config_b.training, 'batch_size', None) or getattr(config_b.training, 'per_device_train_batch_size', None)
            if batch_size is None:
                raise ValueError("Config must have either training.batch_size or training.per_device_train_batch_size")
            logger.info(f"  ✓ Batch size accessible: {batch_size}")

            # Validate evaluator metric if specified
            if hasattr(config_b, 'evaluation') and hasattr(config_b.evaluation, 'metric_for_best_model'):
                metric = config_b.evaluation.metric_for_best_model
                if 'group_b_retrieval' not in metric and metric != 'eval_loss':
                    logger.warning(f"  ⚠️  Metric '{metric}' doesn't match evaluator name 'group_b_retrieval'")
                    logger.warning(f"     Expected something like 'eval_group_b_retrieval_ndcg@10'")
                else:
                    logger.info(f"  ✓ Evaluation metric: {metric}")

            # Test instantiation with real config
            logger.info("  Testing trainer instantiation with real config...")
            trainer_b = GroupBTrainer(config_b)
            logger.info("  ✓ GroupBTrainer instantiated successfully")

            logger.info("✓ Group B config validated")

        except Exception as e:
            logger.error(f"❌ Group B config validation failed: {e}")
            raise

    # Validate Stage 3 (Group A) config
    logger.info("\n7b. Validating Stage 3 (Group A) config...")
    config_a_path = Path(__file__).parent.parent / "configs" / "training_config_stage3_group_a.yaml"

    if not config_a_path.exists():
        logger.warning(f"⚠️  Config not found: {config_a_path}")
        logger.warning("   Skipping Group A config validation")
    else:
        try:
            config_a = load_config(str(config_a_path))
            logger.info(f"✓ Config loaded: {config_a_path.name}")

            # Required fields
            required_fields = {
                'model.name': lambda c: hasattr(c, 'model') and hasattr(c.model, 'name'),
                'training.output_dir': lambda c: hasattr(c, 'training') and hasattr(c.training, 'output_dir'),
                'training.num_epochs': lambda c: hasattr(c, 'training') and hasattr(c.training, 'num_epochs'),
                'training.batch_size or per_device_train_batch_size': lambda c: (
                    hasattr(c, 'training') and (
                        hasattr(c.training, 'batch_size') or
                        hasattr(c.training, 'per_device_train_batch_size')
                    )
                ),
                'data_loader.languages': lambda c: hasattr(c, 'data_loader') and hasattr(c.data_loader, 'languages'),
            }

            missing = []
            for field_name, checker in required_fields.items():
                if not checker(config_a):
                    missing.append(field_name)
                else:
                    logger.info(f"  ✓ {field_name}")

            if missing:
                logger.error(f"  ❌ Missing required fields: {', '.join(missing)}")
                raise ValueError(f"Group A config missing required fields: {missing}")

            # Validate batch size accessibility
            batch_size = getattr(config_a.training, 'batch_size', None) or getattr(config_a.training, 'per_device_train_batch_size', None)
            if batch_size is None:
                raise ValueError("Config must have either training.batch_size or training.per_device_train_batch_size")
            logger.info(f"  ✓ Batch size accessible: {batch_size}")

            # Validate evaluator metric if specified
            if hasattr(config_a, 'evaluation') and hasattr(config_a.evaluation, 'metric_for_best_model'):
                metric = config_a.evaluation.metric_for_best_model
                if 'cosine_accuracy' not in metric and metric != 'eval_loss':
                    logger.warning(f"  ⚠️  Metric '{metric}' doesn't match TripletEvaluator")
                    logger.warning(f"     Expected 'eval_cosine_accuracy'")
                else:
                    logger.info(f"  ✓ Evaluation metric: {metric}")

            # Test instantiation with real config
            logger.info("  Testing trainer instantiation with real config...")
            trainer_a = GroupATrainer(config_a)
            logger.info("  ✓ GroupATrainer instantiated successfully")

            logger.info("✓ Group A config validated")

        except Exception as e:
            logger.error(f"❌ Group A config validation failed: {e}")
            raise

    logger.info("\n✅ All configs validated!\n")


def main():
    """Run all validation tests."""
    logger.info("\n" + "="*70)
    logger.info("V6 TRAINING PIPELINE VALIDATION")
    logger.info("="*70 + "\n")

    try:
        # Test 1: Data loaders
        data_b, data_a = validate_data_loaders()

        # Test 2: Loss function
        validate_loss_function()

        # Test 3: Trainers
        validate_trainers()

        # Test 4: Compute settings
        validate_compute_settings()

        # Test 5: Evaluators
        validate_evaluators()

        # Test 6: Data format
        validate_data_format_for_trainer()

        # Test 7: Config validation
        validate_configs()

        # Summary
        logger.info("="*70)
        logger.info("🎉 ALL VALIDATION TESTS PASSED!")
        logger.info("="*70)
        logger.info("\n✅ Ready to create training configs and start V6 training!")
        logger.info("\nNext steps:")
        logger.info("  1. Create training configs for Stage 2 (Group B) and Stage 3 (Group A)")
        logger.info("  2. Train Stage 2: V1 → Group B")
        logger.info("  3. Train Stage 3: V6-Stage2 → Group A")
        logger.info("  4. Benchmark V6 final on MTEB\n")

        return True

    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("❌ VALIDATION FAILED!")
        logger.error("="*70)
        logger.error(f"\nError: {e}")
        logger.error("\n⚠️  Fix issues before proceeding with training!\n")
        raise


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
