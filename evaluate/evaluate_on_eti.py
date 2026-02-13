#!/usr/bin/env python3
"""
Evaluate embedding model on ETI test dataset.

Usage:
    python scripts/evaluate_on_eti.py
    python scripts/evaluate_on_eti.py --model intfloat/multilingual-e5-small --dataset-type smpl
    python scripts/evaluate_on_eti.py --model tuva/finetuned-model --dataset-type adv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_eti_test(dataset_type: str = "smpl") -> Dict:
    """
    Load ETI test dataset in format for InformationRetrievalEvaluator.
    
    Args:
        dataset_type: Either 'smpl', 'adv', or 'both'
    
    Returns:
        Dict with queries, corpus, and relevant_docs
    """
    logger.info(f"Loading ETI test dataset ({dataset_type})...")
    
    all_queries = {}
    all_corpus = {}
    all_relevant_docs = {}
    
    types_to_load = ["smpl", "adv"] if dataset_type == "both" else [dataset_type]
    
    for dtype in types_to_load:
        test_file = Path(f"data/processed/eti_test_{dtype}.json")
        
        if not test_file.exists():
            logger.warning(f"  ⚠ ETI test {dtype} not found at {test_file} – skipping")
            continue
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Merge data
        queries = data.get('queries', {})
        corpus = data.get('corpus', {})
        relevant_docs = data.get('relevant_docs', {})
        
        # Add prefix to avoid ID collisions when combining
        prefix = f"{dtype}_"
        
        for qid, query in queries.items():
            all_queries[prefix + qid] = query
        
        for doc_id, doc in corpus.items():
            all_corpus[prefix + doc_id] = doc
        
        for qid, doc_ids in relevant_docs.items():
            all_relevant_docs[prefix + qid] = [prefix + did for did in doc_ids]
        
        logger.info(f"  ✓ {dtype}: {len(queries)} queries, {len(corpus)} docs")
    
    logger.info(f"  Total: {len(all_queries)} queries, {len(all_corpus)} docs")
    
    return {
        "queries": all_queries,
        "corpus": all_corpus,
        "relevant_docs": all_relevant_docs
    }


def evaluate_model(model_path: str, dataset_type: str = "smpl") -> Dict:
    """
    Evaluate embedding model on ETI test dataset.
    
    Args:
        model_path: Model name from HuggingFace Hub or local path
        dataset_type: 'smpl', 'adv', or 'both'
    
    Returns:
        Dict with evaluation metrics
    """
    logger.info(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)
    
    # Load test data
    test_data = load_eti_test(dataset_type)
    
    # Create evaluator
    logger.info("Creating InformationRetrievalEvaluator...")
    evaluator = InformationRetrievalEvaluator(
        queries=test_data["queries"],
        corpus=test_data["corpus"],
        relevant_docs=test_data["relevant_docs"],
        name=f"eti_test_{dataset_type}",
        show_progress_bar=True,
    )
    
    # Evaluate
    logger.info("Running evaluation...")
    results = evaluator(model)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - ETI Test ({dataset_type})")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding model on ETI test dataset"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Model name from HuggingFace Hub or local path (default: intfloat/multilingual-e5-small)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="both",
        choices=["smpl", "adv", "both"],
        help="Which ETI test dataset to use (default: both)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(args.model, args.dataset_type)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✓ Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()