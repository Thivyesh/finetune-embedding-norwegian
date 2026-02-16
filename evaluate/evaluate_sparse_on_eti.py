#!/usr/bin/env python3
"""
Evaluate sparse embedding model on ETI test dataset.

Usage:
    python scripts/evaluate_sparse_on_eti.py --model naver/splade-cocondenser-ensembledistil
    python scripts/evaluate_sparse_on_eti.py --model path/to/finetuned-sparse --dataset-type smpl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SparseEncoder
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Patch PreTrainedModel.initialize_weights() to handle models with bias=None layers
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig as OriginalAutoConfig

_original_initialize_weights = PreTrainedModel.initialize_weights

def _patched_initialize_weights(self):
    """Patched initialize_weights that gracefully handles bias=None."""
    try:
        return _original_initialize_weights(self)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'data'" in str(e):
            logger.warning(f"Skipping weight initialization due to layers with bias=None")
            return
        raise

PreTrainedModel.initialize_weights = _patched_initialize_weights


# Also patch AutoConfig to add missing attributes for GptBertConfig compatibility
_original_from_pretrained = OriginalAutoConfig.from_pretrained

@classmethod
def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Patched from_pretrained that adds missing attributes to config."""
    config = _original_from_pretrained(pretrained_model_name_or_path, **kwargs)
    
    # Patch missing attributes for compatibility
    if not hasattr(config, 'is_decoder'):
        config.is_decoder = False
    if not hasattr(config, 'add_cross_attention'):
        config.add_cross_attention = False
    
    return config

OriginalAutoConfig.from_pretrained = _patched_from_pretrained


def compute_sparse_similarity(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """
    Compute dot product similarity between sparse embedding vectors.
    
    Args:
        query_vec: Dense query embedding (with zeros for sparse dimensions)
        doc_vec: Dense document embedding (with zeros for sparse dimensions)
    
    Returns:
        Dot product similarity score
    """
    return float(np.dot(query_vec, doc_vec))


def load_eti_test(dataset_type: str = "smpl") -> Dict:
    """
    Load ETI test dataset.
    
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
        
        queries = data.get('queries', {})
        corpus = data.get('corpus', {})
        relevant_docs = data.get('relevant_docs', {})
        
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


def evaluate_sparse_model(model_path: str, dataset_type: str = "smpl") -> Dict:
    """
    Evaluate sparse embedding model on ETI test dataset.
    
    Args:
        model_path: Model name or path
        dataset_type: 'smpl', 'adv', or 'both'
    
    Returns:
        Dict with evaluation metrics
    """
    # Load model using sentence-transformers SparseEncoder
    logger.info(f"Loading sparse model: {model_path}")
    encoder = SparseEncoder(model_path, trust_remote_code=True)
    
    # Load test data
    test_data = load_eti_test(dataset_type)
    
    queries = test_data["queries"]
    corpus = test_data["corpus"]
    relevant_docs = test_data["relevant_docs"]
    
    # Encode queries and corpus
    logger.info("Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = encoder.encode(query_texts, convert_to_sparse_tensor=False, device='cpu')
    
    # Apply threshold for sparsity
    threshold = 0.05
    query_embeddings[query_embeddings < threshold] = 0
    
    logger.info("Encoding corpus...")
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    doc_embeddings = encoder.encode(doc_texts, convert_to_sparse_tensor=False, device='cpu')
    
    # Apply threshold for sparsity
    doc_embeddings[doc_embeddings < threshold] = 0
    
    # Map embeddings back to IDs
    query_vecs = {query_ids[i]: query_embeddings[i] for i in range(len(query_ids))}
    doc_vecs = {doc_ids[i]: doc_embeddings[i] for i in range(len(doc_ids))}
    
    # Log sparsity stats
    active_dims_queries = (query_embeddings > 0).sum() / (len(query_embeddings) * query_embeddings.shape[1]) * 100
    active_dims_docs = (doc_embeddings > 0).sum() / (len(doc_embeddings) * doc_embeddings.shape[1]) * 100
    logger.info(f"Query sparsity: {100 - active_dims_queries:.2f}% sparse")
    logger.info(f"Document sparsity: {100 - active_dims_docs:.2f}% sparse")
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(query_vecs, doc_vecs, relevant_docs)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"SPARSE MODEL EVALUATION - ETI Test ({dataset_type})")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 60)
    
    return metrics


def compute_metrics(
    query_vecs: Dict[str, np.ndarray],
    doc_vecs: Dict[str, np.ndarray],
    relevant_docs: Dict[str, List[str]]
) -> Dict:
    """
    Compute retrieval metrics.
    
    Args:
        query_vecs: Query sparse embeddings
        doc_vecs: Document sparse embeddings
        relevant_docs: Relevant documents for each query
    
    Returns:
        Dict with metrics
    """
    metrics = {
        "accuracy@1": 0.0,
        "accuracy@3": 0.0,
        "accuracy@5": 0.0,
        "accuracy@10": 0.0,
        "precision@10": 0.0,
        "recall@10": 0.0,
        "recall@50": 0.0,
        "recall@100": 0.0,
        "mrr@10": 0.0,
        "ndcg@10": 0.0,
    }
    
    num_queries = len(query_vecs)
    
    for query_id, query_vec in tqdm(query_vecs.items(), desc="Evaluating"):
        # Compute similarities
        scores = []
        for doc_id, doc_vec in doc_vecs.items():
            similarity = compute_sparse_similarity(query_vec, doc_vec)
            scores.append((doc_id, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_docs = [doc_id for doc_id, _ in scores]
        
        # Get relevant docs for this query
        rel_docs = set(relevant_docs.get(query_id, []))
        
        if not rel_docs:
            continue
        
        # Accuracy@k
        for k in [1, 3, 5, 10]:
            top_k = set(ranked_docs[:k])
            if top_k & rel_docs:
                metrics[f"accuracy@{k}"] += 1.0 / num_queries
        
        # Recall@k
        for k in [10, 50, 100]:
            top_k = set(ranked_docs[:k])
            recall = len(top_k & rel_docs) / len(rel_docs)
            metrics[f"recall@{k}"] += recall / num_queries
        
        # Precision@10
        top_10 = set(ranked_docs[:10])
        precision_10 = len(top_10 & rel_docs) / 10
        metrics["precision@10"] += precision_10 / num_queries
        
        # MRR@10
        for rank, doc_id in enumerate(ranked_docs[:10], 1):
            if doc_id in rel_docs:
                metrics["mrr@10"] += (1.0 / rank) / num_queries
                break
        
        # NDCG@10
        dcg = 0.0
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_docs), 10)))
        for rank, doc_id in enumerate(ranked_docs[:10], 1):
            if doc_id in rel_docs:
                dcg += 1.0 / np.log2(rank + 1)
        if idcg > 0:
            metrics["ndcg@10"] += (dcg / idcg) / num_queries
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sparse embedding model on ETI test dataset"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="naver/splade-cocondenser-ensembledistil",
        help="Sparse model name or path (default: naver/splade-cocondenser-ensembledistil)"
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
        results = evaluate_sparse_model(args.model, args.dataset_type)
        
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