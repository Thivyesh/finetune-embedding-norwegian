"""
Evaluate Norwegian embedding models on MTEB (Scandinavian Embedding Benchmark).

This script evaluates Norwegian models on the official Scandinavian Embedding Benchmark (SEB)
which is now integrated into MTEB as "MTEB(Scandinavian, v1)".

Official benchmark: https://kennethenevoldsen.com/scandinavian-embedding-benchmark/

Usage:
    # Install MTEB first
    uv add mteb

    # Evaluate a model with default config
    uv run python scripts/evaluate_mteb.py --model models/norbert4-base-multidataset-exp1/final

    # Use custom config
    uv run python scripts/evaluate_mteb.py --model models/my-model --config configs/my_eval_config.yaml

    # Results will be comparable to models on the SEB leaderboard
"""

import argparse
import logging
from pathlib import Path
import json
import yaml

try:
    import mteb
    from mteb import MTEB
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install mteb sentence-transformers")
    exit(1)

# Logger will be configured after loading config
logger = logging.getLogger(__name__)


def load_config(config_path: str = None):
    """
    Load evaluation configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default configs/evaluation_config.yaml
    
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "evaluation_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.warning("Using default configuration")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def get_default_config():
    """Get default configuration if config file is not found."""
    return {
        'benchmark': {
            'name': 'MTEB(Scandinavian, v1)',
            'languages': ['nob', 'nno'],
            'task_types': None,
            'exclude_tasks': []
        },
        'model': {
            'trust_remote_code': True,
            'device': None,
            'model_kwargs': None
        },
        'evaluation': {
            'output_dir': 'results/mteb',
            'verbosity': 2,
            'batch_size': None,
            'show_detailed_results': True,
            'show_overall_score': True
        },
        'cache': {
            'cache_dir': '~/.cache/mteb',
            'download_public_results': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'save_to_file': False,
            'log_file': 'logs/mteb_evaluation.log'
        }
    }


def configure_logging(config):
    """Configure logging based on config."""
    log_config = config.get('logging', {})
    
    # Set level
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Configure handlers
    handlers = [logging.StreamHandler()]
    
    if log_config.get('save_to_file', False):
        log_file = Path(log_config.get('log_file', 'logs/mteb_evaluation.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
        handlers=handlers,
        force=True  # Reconfigure if already configured
    )


def get_norwegian_benchmark_tasks(config):
    """
    Get Norwegian tasks from the official Scandinavian Embedding Benchmark.
    
    Args:
        config: Configuration dictionary
    
    Returns all tasks from MTEB(Scandinavian, v1) that include Norwegian (nob/nno).
    This ensures results are directly comparable to the SEB leaderboard.
    
    Tasks included (13 total):
    - Classification (6): NoRecClassification, NorwegianParliamentClassification, 
                          NordicLangClassification, ScalaClassification,
                          MassiveIntentClassification, MassiveScenarioClassification
    - Retrieval (2): NorQuadRetrieval, SNLRetrieval
    - Clustering (4): SNLHierarchicalClusteringS2S, SNLHierarchicalClusteringP2P,
                      VGHierarchicalClusteringS2S, VGHierarchicalClusteringP2P
    - BitextMining (1): NorwegianCourtsBitextMining
    """
    benchmark_config = config.get('benchmark', {})
    
    # Get official Scandinavian benchmark
    benchmark_name = benchmark_config.get('name', 'MTEB(Scandinavian, v1)')
    benchmark = mteb.get_benchmark(benchmark_name)
    
    # Get language filter
    languages = benchmark_config.get('languages', ['nob', 'nno'])
    
    # Filter for tasks with Norwegian language support
    norwegian_tasks = [
        task for task in benchmark.tasks 
        if any(lang in languages for lang in task.metadata.languages)
    ]
    
    # Filter by task type if specified
    task_types = benchmark_config.get('task_types')
    if task_types:
        norwegian_tasks = [
            task for task in norwegian_tasks
            if task.metadata.type in task_types
        ]
    
    # Exclude specific tasks if specified
    exclude_tasks = benchmark_config.get('exclude_tasks', [])
    if exclude_tasks:
        norwegian_tasks = [
            task for task in norwegian_tasks
            if task.metadata.name not in exclude_tasks
        ]
    
    logger.info(f"Loaded {len(norwegian_tasks)} Norwegian tasks from {benchmark_name}")
    logger.info("Tasks by category:")
    
    # Group by type for logging
    from collections import defaultdict
    by_type = defaultdict(list)
    for task in norwegian_tasks:
        by_type[task.metadata.type].append(task.metadata.name)
    
    for task_type in sorted(by_type.keys()):
        logger.info(f"  {task_type}: {len(by_type[task_type])} tasks")
    
    return norwegian_tasks


def evaluate_model(model_path: str, config: dict, tasks=None):
    """
    Evaluate a model on Norwegian MTEB tasks.

    Args:
        model_path: Path to sentence transformer model
        config: Configuration dictionary
        tasks: List of task objects (None = use benchmark tasks from config)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"MTEB EVALUATION: {model_path}")
    logger.info(f"{'='*70}\n")

    # Get configuration sections
    model_config = config.get('model', {})
    eval_config = config.get('evaluation', {})
    cache_config = config.get('cache', {})

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model_kwargs = {
        'trust_remote_code': model_config.get('trust_remote_code', True)
    }
    
    # Add device if specified
    if model_config.get('device'):
        model_kwargs['device'] = model_config['device']
    
    # Add additional model kwargs if specified
    if model_config.get('model_kwargs'):
        model_kwargs.update(model_config['model_kwargs'])
    
    model = SentenceTransformer(model_path, **model_kwargs)
    logger.info("✓ Model loaded")
    logger.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Get tasks to run
    if tasks is None:
        task_objects = get_norwegian_benchmark_tasks(config)
        logger.info(f"Running Norwegian benchmark: {len(task_objects)} tasks")
    else:
        # If task objects provided, use them directly
        task_objects = tasks
        logger.info(f"Running specified tasks: {len(task_objects)} tasks")

    # Create output directory
    output_dir = eval_config.get('output_dir', 'results/mteb')
    # Use full path to avoid collisions when multiple models have same directory name (e.g., "final")
    model_path_obj = Path(model_path)
    if model_path_obj.name == "final" and model_path_obj.parent.name:
        # If the model is in a "final" directory, use parent name to differentiate
        # e.g., models/norbert4-base-multidataset-exp1/final -> norbert4-base-multidataset-exp1
        output_name = model_path_obj.parent.name
    else:
        output_name = model_path_obj.name
    
    output_path = Path(output_dir) / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_path}")

    # Run evaluation
    logger.info("\nStarting MTEB evaluation...")

    # Create MTEB evaluation object with task objects (not task names)
    evaluation = MTEB(tasks=task_objects if isinstance(task_objects, list) else [task_objects])

    # Get encode kwargs if specified
    encode_kwargs = {}
    if eval_config.get('batch_size'):
        encode_kwargs['batch_size'] = eval_config['batch_size']

    try:
        results = evaluation.run(
            model,
            output_folder=str(output_path),
            verbosity=eval_config.get('verbosity', 2),
            encode_kwargs=encode_kwargs if encode_kwargs else None
        )

        logger.info("\n" + "="*70)
        logger.info("EVALUATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {output_path}")

        # Print summary if enabled
        if eval_config.get('show_detailed_results', True):
            logger.info("\n" + "-"*70)
            logger.info("RESULTS SUMMARY")
            logger.info("-"*70)

        # Collect scores for overall average
        all_scores = []
        task_scores = {}

        # Handle both list and dict results (MTEB API changes)
        if isinstance(results, list):
            # New MTEB API returns list of results
            for task_result in results:
                if hasattr(task_result, 'task_name'):
                    task_name = task_result.task_name
                    if eval_config.get('show_detailed_results', True):
                        logger.info(f"\n{task_name}:")
                    if hasattr(task_result, 'scores') and isinstance(task_result.scores, dict):
                        task_main_scores = []
                        for split, metrics in task_result.scores.items():
                            if eval_config.get('show_detailed_results', True):
                                logger.info(f"  {split}:")
                            if isinstance(metrics, dict):
                                # Get main metric for the split
                                main_metric = None
                                if 'main_score' in metrics:
                                    main_metric = metrics['main_score']
                                elif 'ndcg_at_10' in metrics:
                                    main_metric = metrics['ndcg_at_10']
                                elif 'v_measure' in metrics:
                                    main_metric = metrics['v_measure']
                                elif 'accuracy' in metrics:
                                    main_metric = metrics['accuracy']
                                elif 'f1' in metrics:
                                    main_metric = metrics['f1']
                                
                                if main_metric is not None:
                                    task_main_scores.append(main_metric)
                                
                                if eval_config.get('show_detailed_results', True):
                                    for metric, value in metrics.items():
                                        if isinstance(value, (float, int)):
                                            logger.info(f"    {metric}: {value:.4f}")
                        
                        # Average score for this task across splits
                        if task_main_scores:
                            avg_score = sum(task_main_scores) / len(task_main_scores)
                            task_scores[task_name] = avg_score
                            all_scores.append(avg_score)
                            
        elif isinstance(results, dict):
            # Old MTEB API returns dict
            for task_name, task_results in results.items():
                if eval_config.get('show_detailed_results', True):
                    logger.info(f"\n{task_name}:")
                if isinstance(task_results, dict):
                    for metric, value in task_results.items():
                        if isinstance(value, float):
                            if eval_config.get('show_detailed_results', True):
                                logger.info(f"  {metric}: {value:.4f}")
                            all_scores.append(value)

        # Print overall score if enabled
        if all_scores and eval_config.get('show_overall_score', True):
            overall_score = sum(all_scores) / len(all_scores)
            logger.info("\n" + "="*70)
            logger.info("OVERALL SCORE (Average across all tasks)")
            logger.info("="*70)
            logger.info(f"  {overall_score:.4f}")
            logger.info("\n  This score is comparable to the Scandinavian Embedding Benchmark")
            logger.info("  leaderboard: https://kennethenevoldsen.com/scandinavian-embedding-benchmark/")
            logger.info("="*70)

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error("This might happen if:")
        logger.error("  1. Task names are incorrect (check MTEB documentation)")
        logger.error("  2. Model is incompatible")
        logger.error("  3. Network issues downloading datasets")
        raise


def compare_models(model1_path: str, model2_path: str, output_dir: str = "results/mteb"):
    """
    Compare two models on Norwegian MTEB tasks.

    Args:
        model1_path: Path to first model (e.g., V1 NLI-only)
        model2_path: Path to second model (e.g., V1+STS)
        output_dir: Directory with saved results
    """
    logger.info(f"\n{'='*70}")
    logger.info("COMPARING MODELS")
    logger.info(f"{'='*70}\n")

    model1_name = Path(model1_path).name
    model2_name = Path(model2_path).name

    logger.info(f"Model 1 (baseline): {model1_name}")
    logger.info(f"Model 2 (improved): {model2_name}")

    # Load results
    results1_path = Path(output_dir) / model1_name
    results2_path = Path(output_dir) / model2_name

    if not results1_path.exists():
        logger.error(f"Results not found for model 1: {results1_path}")
        logger.info("Run evaluation first with: --model {model1_path}")
        return

    if not results2_path.exists():
        logger.error(f"Results not found for model 2: {results2_path}")
        logger.info("Run evaluation first with: --model {model2_path}")
        return

    # Compare results
    logger.info("\n" + "-"*70)
    logger.info("COMPARISON RESULTS")
    logger.info("-"*70)

    # Find all result files
    result_files1 = list(results1_path.glob("*.json"))
    result_files2 = list(results2_path.glob("*.json"))

    logger.info(f"\nModel 1 results: {len(result_files1)} tasks")
    logger.info(f"Model 2 results: {len(result_files2)} tasks")

    # Load and compare each task
    for file1 in result_files1:
        task_name = file1.stem
        file2 = results2_path / file1.name

        if not file2.exists():
            logger.warning(f"Task {task_name}: Only in model 1")
            continue

        with open(file1) as f:
            data1 = json.load(f)
        with open(file2) as f:
            data2 = json.load(f)

        logger.info(f"\n{task_name}:")

        # Extract main metrics (task-specific)
        if "test" in data1:
            metrics1 = data1["test"]
            metrics2 = data2["test"]

            for metric_name in metrics1.keys():
                if isinstance(metrics1[metric_name], (int, float)):
                    val1 = metrics1[metric_name]
                    val2 = metrics2.get(metric_name, None)

                    if val2 is not None:
                        diff = val2 - val1
                        pct = (diff / val1 * 100) if val1 != 0 else 0
                        symbol = "✓" if diff > 0 else "✗" if diff < 0 else "="

                        logger.info(f"  {metric_name}:")
                        logger.info(f"    Model 1: {val1:.4f}")
                        logger.info(f"    Model 2: {val2:.4f}")
                        logger.info(f"    Diff: {diff:+.4f} ({pct:+.2f}%) {symbol}")

    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHTS")
    logger.info("="*70)
    logger.info("\nExpected outcomes:")
    logger.info("  ✓ STS tasks: Model 2 (V1+STS) should improve")
    logger.info("    → Fine-tuning on STS should boost similarity scoring")
    logger.info("  ≈ Retrieval/Classification: Similar performance")
    logger.info("    → Low LR and early stopping preserve NLI knowledge")
    logger.info("  ✗ If Model 2 worse on NLI-related tasks:")
    logger.info("    → Indicates catastrophic forgetting (shouldn't happen)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Norwegian embedding models on MTEB (Scandinavian Embedding Benchmark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest: just set model_path in configs/evaluation_config.yaml and run:
  python scripts/evaluate_mteb.py
  
  # Override model from command line:
  python scripts/evaluate_mteb.py --model models/my-model
  
  # Use custom config file:
  python scripts/evaluate_mteb.py --config configs/custom_eval.yaml
  
  # Override output directory:
  python scripts/evaluate_mteb.py --output-dir results/custom
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model to evaluate (overrides config)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/evaluation_config.yaml)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config)"
    )

    parser.add_argument(
        "--verbosity",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="MTEB verbosity level (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Configure logging based on config
    configure_logging(config)

    # Get model path from args or config
    model_path = args.model or config.get('model_path')
    
    if not model_path:
        logger.error("No model specified!")
        logger.error("Either:")
        logger.error("  1. Set 'model_path' in configs/evaluation_config.yaml, OR")
        logger.error("  2. Use --model argument")
        parser.print_help()
        return

    # Override config with command-line arguments
    if args.output_dir:
        config['evaluation']['output_dir'] = args.output_dir
    
    if args.verbosity is not None:
        config['evaluation']['verbosity'] = args.verbosity

    # Evaluate model
    logger.info("Using configuration:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Benchmark: {config['benchmark']['name']}")
    logger.info(f"  Languages: {', '.join(config['benchmark']['languages'])}")
    logger.info(f"  Output directory: {config['evaluation']['output_dir']}")
    
    evaluate_model(model_path, config)


if __name__ == "__main__":
    main()
