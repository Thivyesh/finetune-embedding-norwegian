"""
Evaluate Norwegian embedding models on MTEB (Scandinavian Embedding Benchmark).

This script compares V1 (NLI-only) vs V4 (NLI→STS fine-tuned) on Norwegian tasks.

Usage:
    # Install MTEB first
    uv add mteb

    # Evaluate V1 model (NLI-only)
    uv run python scripts/evaluate_mteb.py --model models/norbert4-base-nli-norwegian

    # Evaluate V4 model (STS fine-tuned, best checkpoint)
    uv run python scripts/evaluate_mteb.py --model models/norbert4-base-nli-sts-norwegian-v4/checkpoint-90

    # Compare V1 vs V4
    uv run python scripts/evaluate_mteb.py --compare \
        --model1 models/norbert4-base-nli-norwegian \
        --model2 models/norbert4-base-nli-sts-norwegian-v4/checkpoint-90
"""

import argparse
import logging
from pathlib import Path
import json

try:
    from mteb import MTEB
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install mteb sentence-transformers")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Norwegian tasks from MTEB/SEB (Scandinavian Embedding Benchmark)
# Based on actual MTEB task list (verified December 2025)
# 30 Norwegian tasks total - using key representative tasks below
NORWEGIAN_TASKS = {
    "classification": [
        # Text classification tasks
        "NoRecClassification.v2",  # Norwegian Reviews Classification
        "NorwegianParliamentClassification.v2",  # Parliament speeches classification
        "NordicLangClassification",  # Nordic language identification
        "ScandiSentClassification",  # Scandinavian sentiment
        "ScalaClassification",  # Scala classification
    ],
    "retrieval": [
        # Question-answer retrieval
        "NorQuadRetrieval",  # Norwegian Wikipedia QA
        "SNLRetrieval",  # Store Norske Leksikon retrieval
        "WebFAQRetrieval",  # Web FAQ retrieval
    ],
    "clustering": [
        # Document clustering by topic
        "SNLHierarchicalClusteringS2S",  # SNL sentence clustering
        "SNLHierarchicalClusteringP2P",  # SNL paragraph clustering
        "VGHierarchicalClusteringS2S",  # VG News sentence clustering
        "VGHierarchicalClusteringP2P",  # VG News paragraph clustering
    ],
    "bitext_mining": [
        # Cross-lingual text alignment
        "NorwegianCourtsBitextMining",  # Norwegian courts bitext
        "Tatoeba",  # Tatoeba sentence pairs (includes Norwegian)
    ],
}


def get_all_norwegian_tasks():
    """Get all Norwegian task names from MTEB."""
    all_tasks = []
    for category, tasks in NORWEGIAN_TASKS.items():
        all_tasks.extend(tasks)
    return all_tasks


def evaluate_model(model_path: str, output_dir: str = "results/mteb", tasks=None):
    """
    Evaluate a model on Norwegian MTEB tasks.

    Args:
        model_path: Path to sentence transformer model
        output_dir: Directory to save results
        tasks: List of task names (None = all Norwegian tasks)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"MTEB EVALUATION: {model_path}")
    logger.info(f"{'='*70}\n")

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)
    logger.info(f"✓ Model loaded")
    logger.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Get tasks to run
    if tasks is None:
        tasks = get_all_norwegian_tasks()
        logger.info(f"Running all Norwegian tasks: {len(tasks)} tasks")
    else:
        logger.info(f"Running specified tasks: {tasks}")

    # Create output directory
    output_path = Path(output_dir) / Path(model_path).name
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_path}")

    # Run evaluation
    logger.info("\nStarting MTEB evaluation...")
    import mteb

    # Get task objects (new MTEB API)
    task_objects = mteb.get_tasks(tasks=tasks, languages=["nob", "nno"])
    evaluation = MTEB(tasks=task_objects)

    try:
        results = evaluation.run(
            model,
            output_folder=str(output_path),
            verbosity=2
        )

        logger.info("\n" + "="*70)
        logger.info("EVALUATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {output_path}")

        # Print summary
        logger.info("\n" + "-"*70)
        logger.info("RESULTS SUMMARY")
        logger.info("-"*70)

        for task_name, task_results in results.items():
            logger.info(f"\n{task_name}:")
            if isinstance(task_results, dict):
                for metric, value in task_results.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")

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
        description="Evaluate Norwegian embedding models on MTEB"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model to evaluate"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two models"
    )

    parser.add_argument(
        "--model1",
        type=str,
        help="Path to first model (for comparison)"
    )

    parser.add_argument(
        "--model2",
        type=str,
        help="Path to second model (for comparison)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mteb",
        help="Directory to save results (default: results/mteb)"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (default: all Norwegian tasks)"
    )

    args = parser.parse_args()

    if args.compare:
        if not args.model1 or not args.model2:
            logger.error("Comparison mode requires --model1 and --model2")
            return

        # Evaluate both models if results don't exist
        for model_path in [args.model1, args.model2]:
            results_path = Path(args.output_dir) / Path(model_path).name
            if not results_path.exists():
                logger.info(f"Evaluating {model_path}...")
                evaluate_model(model_path, args.output_dir, args.tasks)

        # Compare
        compare_models(args.model1, args.model2, args.output_dir)

    elif args.model:
        # Single model evaluation
        evaluate_model(args.model, args.output_dir, args.tasks)

    else:
        logger.error("Specify either --model or --compare mode")
        parser.print_help()


if __name__ == "__main__":
    main()
