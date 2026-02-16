#!/usr/bin/env python3
"""
Evaluate the trained model specifically on ETI test set to analyze precision/recall improvements.
"""

import os
import sys
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_loader_eti import load_eti_data

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def evaluate_eti_specific(model_path, baseline_model_path=None):
    """
    Evaluate model performance specifically on ETI test set.
    
    Args:
        model_path: Path to the trained model
        baseline_model_path: Path to baseline model for comparison (optional)
    """
    logger = setup_logging()
    
    # Load ETI data
    logger.info("Loading ETI dataset...")
    eti_train, eti_dev, eti_test = load_eti_data()
    
    # Convert format to {anchor, positive}
    def convert_eti_format(example):
        return {
            'anchor': example['anchor'],
            'positive': example['positive'],
        }
    
    eti_test = eti_test.map(convert_eti_format)
    logger.info(f"ETI test set: {len(eti_test):,} samples")
    
    # Load trained model
    logger.info(f"Loading trained model from: {model_path}")
    trained_model = SentenceTransformer(model_path)
    
    # Load baseline model for comparison
    baseline_model = None
    if baseline_model_path:
        logger.info(f"Loading baseline model from: {baseline_model_path}")
        baseline_model = SentenceTransformer(baseline_model_path)
    
    # Create test evaluator for ETI
    logger.info("Setting up ETI test evaluation...")
    test_sample_size = min(1000, len(eti_test))
    test_sample = eti_test.shuffle(seed=42).select(range(test_sample_size))
    
    queries = {}
    corpus = {}
    relevant = {}
    
    for idx, item in enumerate(test_sample):
        qid = f"q{idx}"
        did = f"d{idx}"
        queries[qid] = item['anchor']
        corpus[did] = item['positive']
        relevant[qid] = {did}
    
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name="eti_test",
        show_progress_bar=True,
    )
    
    # Evaluate trained model
    logger.info("Evaluating trained model on ETI test set...")
    trained_results = evaluator(trained_model)
    
    # Evaluate baseline if provided
    baseline_results = None
    if baseline_model:
        logger.info("Evaluating baseline model on ETI test set...")
        baseline_results = evaluator(baseline_model)
    
    # Report results
    logger.info("\n" + "="*70)
    logger.info("ETI TEST SET EVALUATION RESULTS")
    logger.info("="*70)
    
    # Trained model results
    logger.info(f"\nTRAINED MODEL (Domain Adapted):")
    logger.info(f"  NDCG@10:     {trained_results.get('eti_test_cosine_ndcg@10', 0):.4f}")
    logger.info(f"  MRR@10:      {trained_results.get('eti_test_cosine_mrr@10', 0):.4f}")
    logger.info(f"  MAP@100:     {trained_results.get('eti_test_cosine_map@100', 0):.4f}")
    logger.info(f"  Accuracy@1:  {trained_results.get('eti_test_cosine_accuracy@1', 0):.4f}")
    logger.info(f"  Accuracy@3:  {trained_results.get('eti_test_cosine_accuracy@3', 0):.4f}")
    logger.info(f"  Accuracy@5:  {trained_results.get('eti_test_cosine_accuracy@5', 0):.4f}")
    logger.info(f"  Accuracy@10: {trained_results.get('eti_test_cosine_accuracy@10', 0):.4f}")
    logger.info(f"  Precision@1: {trained_results.get('eti_test_cosine_precision@1', 0):.4f}")
    logger.info(f"  Precision@3: {trained_results.get('eti_test_cosine_precision@3', 0):.4f}")
    logger.info(f"  Precision@5: {trained_results.get('eti_test_cosine_precision@5', 0):.4f}")
    logger.info(f"  Precision@10:{trained_results.get('eti_test_cosine_precision@10', 0):.4f}")
    logger.info(f"  Recall@1:    {trained_results.get('eti_test_cosine_recall@1', 0):.4f}")
    logger.info(f"  Recall@3:    {trained_results.get('eti_test_cosine_recall@3', 0):.4f}")
    logger.info(f"  Recall@5:    {trained_results.get('eti_test_cosine_recall@5', 0):.4f}")
    logger.info(f"  Recall@10:   {trained_results.get('eti_test_cosine_recall@10', 0):.4f}")
    
    # Baseline comparison
    if baseline_results:
        logger.info(f"\nBASELINE MODEL (Original EmbeddingGemma-300M):")
        logger.info(f"  NDCG@10:     {baseline_results.get('eti_test_cosine_ndcg@10', 0):.4f}")
        logger.info(f"  MRR@10:      {baseline_results.get('eti_test_cosine_mrr@10', 0):.4f}")
        logger.info(f"  MAP@100:     {baseline_results.get('eti_test_cosine_map@100', 0):.4f}")
        logger.info(f"  Accuracy@1:  {baseline_results.get('eti_test_cosine_accuracy@1', 0):.4f}")
        logger.info(f"  Accuracy@10: {baseline_results.get('eti_test_cosine_accuracy@10', 0):.4f}")
        logger.info(f"  Precision@1: {baseline_results.get('eti_test_cosine_precision@1', 0):.4f}")
        logger.info(f"  Recall@1:    {baseline_results.get('eti_test_cosine_recall@1', 0):.4f}")
        logger.info(f"  Recall@10:   {baseline_results.get('eti_test_cosine_recall@10', 0):.4f}")
        
        # Calculate improvements
        logger.info(f"\nIMPROVEMENT (Trained vs Baseline):")
        improvements = {}
        for metric in ['ndcg@10', 'mrr@10', 'map@100', 'accuracy@1', 'accuracy@10', 'precision@1', 'recall@1', 'recall@10']:
            trained_val = trained_results.get(f'eti_test_cosine_{metric}', 0)
            baseline_val = baseline_results.get(f'eti_test_cosine_{metric}', 0)
            if baseline_val > 0:
                improvement = ((trained_val - baseline_val) / baseline_val) * 100
                improvements[metric] = improvement
                logger.info(f"  {metric.upper():12}: {improvement:+.2f}% ({baseline_val:.4f} → {trained_val:.4f})")
        
        # Summary
        logger.info(f"\nSUMMARY:")
        if improvements:
            avg_improvement = sum(improvements.values()) / len(improvements)
            logger.info(f"  Average improvement: {avg_improvement:+.2f}%")
            
            positive_improvements = [imp for imp in improvements.values() if imp > 0]
            if positive_improvements:
                logger.info(f"  Positive improvements in {len(positive_improvements)}/{len(improvements)} metrics")
            
            best_metric = max(improvements.items(), key=lambda x: x[1])
            worst_metric = min(improvements.items(), key=lambda x: x[1])
            logger.info(f"  Best improvement: {best_metric[0]} ({best_metric[1]:+.2f}%)")
            logger.info(f"  Worst change: {worst_metric[0]} ({worst_metric[1]:+.2f}%)")
    
    # Return results for programmatic use
    return {
        'trained': trained_results,
        'baseline': baseline_results,
        'test_size': len(test_sample)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ETI-specific performance')
    parser.add_argument('--trained-model', 
                        default='models/embeddinggemma-300m-domain-adapted/final',
                        help='Path to trained model')
    parser.add_argument('--baseline-model', 
                        default='google/embeddinggemma-300m',
                        help='Path to baseline model for comparison')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip baseline comparison')
    
    args = parser.parse_args()
    
    baseline_path = None if args.no_baseline else args.baseline_model
    results = evaluate_eti_specific(args.trained_model, baseline_path)