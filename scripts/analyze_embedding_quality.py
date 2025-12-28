"""
Post-Training Embedding Quality Analysis

This script performs comprehensive analysis of a trained embedding model to understand:
1. Similarity distributions (positive vs negative pairs)
2. Embedding space geometry (norm distribution, clustering)
3. Sample impact analysis (which samples matter most)
4. Multi-checkpoint comparison (training progression)
5. Training effectiveness metrics
6. Potential improvements

Run this AFTER training completes to avoid slowing down training.

Usage:
    python scripts/analyze_embedding_quality.py --config scripts/analysis_config.yaml
    python scripts/analyze_embedding_quality.py --model models/norbert4-base-nli-norwegian-v2
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import torch
import glob

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Analysis will not be logged to MLflow.")


class EmbeddingQualityAnalyzer:
    """
    Analyze embedding model quality through various metrics.
    """

    def __init__(self, config: Dict):
        """
        Initialize analyzer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.results = {}

    def load_model(self):
        """Load the trained model."""
        model_path = self.config['model']['path']
        logger.info(f"Loading model from: {model_path}")

        self.model = SentenceTransformer(model_path)
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def load_dataset(self):
        """Load evaluation dataset."""
        dataset_config = self.config['dataset']

        logger.info(f"Loading dataset: {dataset_config['name']}")
        dataset = load_dataset(
            dataset_config['name'],
            split=dataset_config['split']
        )

        # Sample if configured
        max_samples = dataset_config.get('max_samples', None)
        if max_samples and max_samples < len(dataset):
            logger.info(f"Sampling {max_samples} from {len(dataset)} examples")
            # Deterministic sampling
            np.random.seed(42)
            indices = np.random.choice(len(dataset), size=max_samples, replace=False)
            dataset = dataset.select(indices)

        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataset

    def analyze_similarity_distributions(self, dataset) -> Dict:
        """
        Analyze similarity distributions between anchor-positive and anchor-negative pairs.

        This is the KEY metric for understanding if the model learned to distinguish
        similar from dissimilar sentences.
        """
        logger.info("\n" + "="*70)
        logger.info("ANALYZING SIMILARITY DISTRIBUTIONS")
        logger.info("="*70)

        # Extract triplets
        anchors = dataset[self.config['dataset']['anchor_column']]
        positives = dataset[self.config['dataset']['positive_column']]
        negatives = dataset[self.config['dataset']['negative_column']]

        # EFFICIENT BATCH ENCODING
        batch_size = self.config['analysis']['batch_size']

        logger.info("Encoding anchors...")
        anchor_embs = self.model.encode(
            anchors,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info("Encoding positives...")
        positive_embs = self.model.encode(
            positives,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info("Encoding negatives...")
        negative_embs = self.model.encode(
            negatives,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Compute similarities
        logger.info("Computing cosine similarities...")
        positive_sims = self._batch_cosine_similarity(anchor_embs, positive_embs)
        negative_sims = self._batch_cosine_similarity(anchor_embs, negative_embs)

        # Analyze distributions
        results = {
            'positive_mean': float(np.mean(positive_sims)),
            'positive_std': float(np.std(positive_sims)),
            'positive_min': float(np.min(positive_sims)),
            'positive_max': float(np.max(positive_sims)),
            'negative_mean': float(np.mean(negative_sims)),
            'negative_std': float(np.std(negative_sims)),
            'negative_min': float(np.min(negative_sims)),
            'negative_max': float(np.max(negative_sims)),
            'separation': float(np.mean(positive_sims) - np.mean(negative_sims)),
            'overlap': self._compute_overlap(positive_sims, negative_sims),
            'accuracy': float(np.mean(positive_sims > negative_sims)),
        }

        # Log results
        logger.info("\n" + "-"*70)
        logger.info("SIMILARITY DISTRIBUTION RESULTS:")
        logger.info("-"*70)
        logger.info(f"Positive pairs:")
        logger.info(f"  Mean: {results['positive_mean']:.4f}")
        logger.info(f"  Std:  {results['positive_std']:.4f}")
        logger.info(f"  Range: [{results['positive_min']:.4f}, {results['positive_max']:.4f}]")
        logger.info(f"\nNegative pairs:")
        logger.info(f"  Mean: {results['negative_mean']:.4f}")
        logger.info(f"  Std:  {results['negative_std']:.4f}")
        logger.info(f"  Range: [{results['negative_min']:.4f}, {results['negative_max']:.4f}]")
        logger.info(f"\nSeparation: {results['separation']:.4f}")
        logger.info(f"Overlap: {results['overlap']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.2%}")
        logger.info("-"*70)

        # Interpretation
        logger.info("\nINTERPRETATION:")
        if results['separation'] > 0.5:
            logger.info("✓ EXCELLENT separation between positive and negative pairs")
        elif results['separation'] > 0.3:
            logger.info("✓ GOOD separation between positive and negative pairs")
        elif results['separation'] > 0.1:
            logger.info("⚠ MODERATE separation - could be improved")
        else:
            logger.info("❌ POOR separation - model needs more training")

        if results['overlap'] < 0.1:
            logger.info("✓ EXCELLENT - minimal distribution overlap")
        elif results['overlap'] < 0.3:
            logger.info("✓ GOOD - low distribution overlap")
        else:
            logger.info("⚠ HIGH overlap - model struggles to distinguish pairs")

        # Store raw data for plotting
        results['positive_sims'] = positive_sims
        results['negative_sims'] = negative_sims

        return results

    def analyze_embedding_space(self, dataset) -> Dict:
        """
        Analyze the geometry of the embedding space.

        Checks:
        - Embedding norm distribution
        - Dimensionality usage
        - Clustering quality
        """
        logger.info("\n" + "="*70)
        logger.info("ANALYZING EMBEDDING SPACE GEOMETRY")
        logger.info("="*70)

        # Sample for efficiency
        sample_size = min(5000, len(dataset))
        texts = dataset[self.config['dataset']['anchor_column']][:sample_size]

        logger.info(f"Encoding {sample_size} samples...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.config['analysis']['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Compute norms
        norms = np.linalg.norm(embeddings, axis=1)

        # Compute dimensionality metrics
        # Check if embeddings use all dimensions or collapse to lower-dim subspace
        mean_emb = np.mean(embeddings, axis=0)
        centered = embeddings - mean_emb
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

        # Effective dimensionality (# of dimensions capturing 90% variance)
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        effective_dim = np.searchsorted(cumsum, 0.90) + 1

        results = {
            'norm_mean': float(np.mean(norms)),
            'norm_std': float(np.std(norms)),
            'norm_min': float(np.min(norms)),
            'norm_max': float(np.max(norms)),
            'effective_dimensionality': int(effective_dim),
            'full_dimensionality': int(embeddings.shape[1]),
            'dimensionality_usage': float(effective_dim / embeddings.shape[1]),
            'explained_variance_90': float(cumsum[effective_dim - 1]),
        }

        logger.info("\n" + "-"*70)
        logger.info("EMBEDDING SPACE RESULTS:")
        logger.info("-"*70)
        logger.info(f"Embedding norms:")
        logger.info(f"  Mean: {results['norm_mean']:.4f}")
        logger.info(f"  Std:  {results['norm_std']:.4f}")
        logger.info(f"  Range: [{results['norm_min']:.4f}, {results['norm_max']:.4f}]")
        logger.info(f"\nDimensionality:")
        logger.info(f"  Full dimension: {results['full_dimensionality']}")
        logger.info(f"  Effective dimension (90% variance): {results['effective_dimensionality']}")
        logger.info(f"  Usage: {results['dimensionality_usage']:.1%}")
        logger.info("-"*70)

        logger.info("\nINTERPRETATION:")
        if results['dimensionality_usage'] > 0.7:
            logger.info("✓ GOOD - embeddings use most dimensions effectively")
        elif results['dimensionality_usage'] > 0.4:
            logger.info("⚠ MODERATE - some dimensional collapse, consider regularization")
        else:
            logger.info("❌ POOR - severe dimensional collapse, model may be undertrained")

        # Store for plotting
        results['norms'] = norms
        results['eigenvalues'] = eigenvalues[:50]  # Top 50 for plotting

        return results

    def generate_visualizations(self, similarity_results: Dict, space_results: Dict):
        """
        Create visualizations of the analysis results.
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)

        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Similarity Distributions
        ax1.hist(similarity_results['positive_sims'], bins=50, alpha=0.6, label='Positive pairs', color='green', density=True)
        ax1.hist(similarity_results['negative_sims'], bins=50, alpha=0.6, label='Negative pairs', color='red', density=True)
        ax1.axvline(similarity_results['positive_mean'], color='darkgreen', linestyle='--', linewidth=2, label=f"Pos mean: {similarity_results['positive_mean']:.3f}")
        ax1.axvline(similarity_results['negative_mean'], color='darkred', linestyle='--', linewidth=2, label=f"Neg mean: {similarity_results['negative_mean']:.3f}")
        ax1.set_xlabel('Cosine Similarity', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'Similarity Distributions (Separation: {similarity_results["separation"]:.3f})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Embedding Norm Distribution
        ax2.hist(space_results['norms'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(space_results['norm_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {space_results['norm_mean']:.3f}")
        ax2.set_xlabel('Embedding Norm', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Embedding Norm Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Eigenvalue Spectrum (Dimensionality Analysis)
        ax3.plot(range(1, len(space_results['eigenvalues']) + 1), space_results['eigenvalues'], 'o-', color='purple', markersize=4)
        ax3.axvline(space_results['effective_dimensionality'], color='red', linestyle='--', linewidth=2, label=f"90% variance: dim {space_results['effective_dimensionality']}")
        ax3.set_xlabel('Principal Component', fontsize=12)
        ax3.set_ylabel('Eigenvalue', fontsize=12)
        ax3.set_title(f'Eigenvalue Spectrum ({space_results["dimensionality_usage"]:.1%} dimensionality usage)', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary Statistics Table
        ax4.axis('off')
        summary_data = [
            ['Metric', 'Value', 'Quality'],
            ['', '', ''],
            ['Similarity Separation', f"{similarity_results['separation']:.3f}", self._quality_label(similarity_results['separation'], [0.5, 0.3, 0.1])],
            ['Distribution Overlap', f"{similarity_results['overlap']:.3f}", self._quality_label(similarity_results['overlap'], [0.1, 0.3, 1.0], inverse=True)],
            ['Triplet Accuracy', f"{similarity_results['accuracy']:.1%}", self._quality_label(similarity_results['accuracy'], [0.95, 0.90, 0.80])],
            ['', '', ''],
            ['Embedding Norm (mean)', f"{space_results['norm_mean']:.3f}", ''],
            ['Dimensionality Usage', f"{space_results['dimensionality_usage']:.1%}", self._quality_label(space_results['dimensionality_usage'], [0.7, 0.4, 0.2])],
            ['Effective Dimensions', f"{space_results['effective_dimensionality']}/{space_results['full_dimensionality']}", ''],
        ]

        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                          colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Add overall title
        model_name = self.config['model'].get('name', Path(self.config['model']['path']).name)
        fig.suptitle(f'Embedding Quality Analysis: {model_name}',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save
        output_path = output_dir / 'embedding_quality_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Visualization saved: {output_path}")

        plt.close()

    def generate_report(self, similarity_results: Dict, space_results: Dict):
        """
        Generate a text report with actionable insights.
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING ANALYSIS REPORT")
        logger.info("="*70)

        output_dir = Path(self.config['output']['output_dir'])
        report_path = output_dir / 'analysis_report.md'

        model_name = self.config['model'].get('name', Path(self.config['model']['path']).name)

        with open(report_path, 'w') as f:
            f.write(f"# Embedding Quality Analysis Report\n\n")
            f.write(f"**Model**: `{model_name}`\n\n")
            f.write(f"**Dataset**: `{self.config['dataset']['name']}` ({self.config['dataset']['split']} split)\n\n")
            f.write(f"**Samples Analyzed**: {self.config['dataset'].get('max_samples', 'all')}\n\n")

            f.write("---\n\n")

            # Similarity Analysis
            f.write("## 1. Similarity Distribution Analysis\n\n")
            f.write("### Metrics\n\n")
            f.write(f"| Metric | Positive Pairs | Negative Pairs |\n")
            f.write(f"|--------|----------------|----------------|\n")
            f.write(f"| Mean | {similarity_results['positive_mean']:.4f} | {similarity_results['negative_mean']:.4f} |\n")
            f.write(f"| Std Dev | {similarity_results['positive_std']:.4f} | {similarity_results['negative_std']:.4f} |\n")
            f.write(f"| Range | [{similarity_results['positive_min']:.4f}, {similarity_results['positive_max']:.4f}] | [{similarity_results['negative_min']:.4f}, {similarity_results['negative_max']:.4f}] |\n\n")

            f.write(f"**Separation**: {similarity_results['separation']:.4f}\n\n")
            f.write(f"**Overlap**: {similarity_results['overlap']:.4f}\n\n")
            f.write(f"**Triplet Accuracy**: {similarity_results['accuracy']:.2%}\n\n")

            # Embedding Space Analysis
            f.write("## 2. Embedding Space Geometry\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Embedding Dimension | {space_results['full_dimensionality']} |\n")
            f.write(f"| Effective Dimension (90% variance) | {space_results['effective_dimensionality']} |\n")
            f.write(f"| Dimensionality Usage | {space_results['dimensionality_usage']:.1%} |\n")
            f.write(f"| Mean Embedding Norm | {space_results['norm_mean']:.4f} |\n")
            f.write(f"| Norm Std Dev | {space_results['norm_std']:.4f} |\n\n")

            # Recommendations
            f.write("## 3. Recommendations for Improvement\n\n")

            recommendations = []

            # Based on separation
            if similarity_results['separation'] < 0.3:
                recommendations.append("❌ **Low separation**: Consider training longer or using more in-batch negatives (larger batch size)")

            # Based on overlap
            if similarity_results['overlap'] > 0.3:
                recommendations.append("⚠️ **High overlap**: Model struggles to distinguish pairs - try harder negatives or different loss function")

            # Based on accuracy
            if similarity_results['accuracy'] < 0.90:
                recommendations.append("⚠️ **Low accuracy**: Model performance below 90% - increase training epochs or adjust learning rate")

            # Based on dimensionality
            if space_results['dimensionality_usage'] < 0.5:
                recommendations.append("⚠️ **Dimensional collapse**: Add regularization or use Matryoshka loss to improve dimension usage")

            # Based on norms
            if space_results['norm_mean'] > 20:
                recommendations.append("⚠️ **High embedding norms**: May slow convergence - consider norm regularization")

            if not recommendations:
                recommendations.append("✅ **Model looks great!** All metrics are within good ranges.")

            for rec in recommendations:
                f.write(f"- {rec}\n")

            f.write("\n---\n\n")
            f.write("*Generated by Embedding Quality Analyzer*\n")

        logger.info(f"✓ Report saved: {report_path}")

    @staticmethod
    def _batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity efficiently."""
        # Normalize
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        # Dot product (element-wise for pairs)
        return np.sum(a_norm * b_norm, axis=1)

    @staticmethod
    def _compute_overlap(pos_sims: np.ndarray, neg_sims: np.ndarray) -> float:
        """Compute distribution overlap (0 = no overlap, 1 = complete overlap)."""
        pos_min = np.min(pos_sims)
        neg_max = np.max(neg_sims)

        if pos_min > neg_max:
            return 0.0  # Perfect separation

        # Overlap region
        overlap_start = max(np.min(pos_sims), np.min(neg_sims))
        overlap_end = min(np.max(pos_sims), np.max(neg_sims))
        overlap_range = overlap_end - overlap_start

        total_range = max(np.max(pos_sims), np.max(neg_sims)) - min(np.min(pos_sims), np.min(neg_sims))

        return overlap_range / (total_range + 1e-8)

    @staticmethod
    def _quality_label(value: float, thresholds: List[float], inverse: bool = False) -> str:
        """Return quality label based on thresholds."""
        if inverse:
            if value <= thresholds[0]:
                return "✓ Excellent"
            elif value <= thresholds[1]:
                return "✓ Good"
            else:
                return "⚠ Needs improvement"
        else:
            if value >= thresholds[0]:
                return "✓ Excellent"
            elif value >= thresholds[1]:
                return "✓ Good"
            else:
                return "⚠ Needs improvement"

    def analyze_sample_impact(self, dataset, anchor_embs, positive_embs, negative_embs,
                              positive_sims, negative_sims) -> Dict:
        """
        Identify high-impact samples that affect model performance.

        Categories:
        - Hard negatives: negatives with high similarity to anchor
        - Hard positives: positives with low similarity to anchor
        - Failure cases: where negative_sim > positive_sim
        - Boundary cases: very close positive and negative similarities
        """
        if not self.config['analysis'].get('sample_impact', {}).get('enabled', True):
            return {}

        logger.info("\n" + "="*70)
        logger.info("ANALYZING SAMPLE IMPACT")
        logger.info("="*70)

        impact_config = self.config['analysis']['sample_impact']
        top_k = impact_config.get('top_k', 100)
        boundary_thresh = impact_config.get('boundary_threshold', 0.1)

        results = {}
        sample_data = []

        # Get column names
        anchor_col = self.config['dataset']['anchor_column']
        positive_col = self.config['dataset']['positive_column']
        negative_col = self.config['dataset']['negative_column']

        # 1. Hard negatives: negatives with high similarity
        hard_neg_indices = np.argsort(negative_sims)[-top_k:][::-1]
        results['hard_negatives'] = {
            'count': len(hard_neg_indices),
            'mean_sim': float(np.mean(negative_sims[hard_neg_indices])),
            'min_sim': float(np.min(negative_sims[hard_neg_indices])),
            'samples': []
        }

        for idx in hard_neg_indices[:10]:  # Top 10 for logging
            sample_data.append({
                'category': 'hard_negative',
                'index': int(idx),
                'anchor': dataset[int(idx)][anchor_col],
                'positive': dataset[int(idx)][positive_col],
                'negative': dataset[int(idx)][negative_col],
                'positive_sim': float(positive_sims[idx]),
                'negative_sim': float(negative_sims[idx]),
                'margin': float(positive_sims[idx] - negative_sims[idx])
            })

        # 2. Hard positives: positives with low similarity
        hard_pos_indices = np.argsort(positive_sims)[:top_k]
        results['hard_positives'] = {
            'count': len(hard_pos_indices),
            'mean_sim': float(np.mean(positive_sims[hard_pos_indices])),
            'max_sim': float(np.max(positive_sims[hard_pos_indices])),
            'samples': []
        }

        for idx in hard_pos_indices[:10]:
            sample_data.append({
                'category': 'hard_positive',
                'index': int(idx),
                'anchor': dataset[int(idx)][anchor_col],
                'positive': dataset[int(idx)][positive_col],
                'negative': dataset[int(idx)][negative_col],
                'positive_sim': float(positive_sims[idx]),
                'negative_sim': float(negative_sims[idx]),
                'margin': float(positive_sims[idx] - negative_sims[idx])
            })

        # 3. Failure cases: where model gets it wrong
        failure_mask = positive_sims < negative_sims
        failure_indices = np.where(failure_mask)[0]
        results['failure_cases'] = {
            'count': int(np.sum(failure_mask)),
            'rate': float(np.mean(failure_mask)),
            'samples': []
        }

        for idx in failure_indices[:20]:  # Top 20 failures
            sample_data.append({
                'category': 'failure',
                'index': int(idx),
                'anchor': dataset[int(idx)][anchor_col],
                'positive': dataset[int(idx)][positive_col],
                'negative': dataset[int(idx)][negative_col],
                'positive_sim': float(positive_sims[idx]),
                'negative_sim': float(negative_sims[idx]),
                'margin': float(positive_sims[idx] - negative_sims[idx])
            })

        # 4. Boundary cases: very close margins
        margins = positive_sims - negative_sims
        boundary_mask = np.abs(margins) < boundary_thresh
        boundary_indices = np.where(boundary_mask)[0]
        results['boundary_cases'] = {
            'count': int(np.sum(boundary_mask)),
            'rate': float(np.mean(boundary_mask)),
            'mean_margin': float(np.mean(margins[boundary_mask])) if np.any(boundary_mask) else 0.0,
            'samples': []
        }

        for idx in boundary_indices[:10]:
            sample_data.append({
                'category': 'boundary',
                'index': int(idx),
                'anchor': dataset[int(idx)][anchor_col],
                'positive': dataset[int(idx)][positive_col],
                'negative': dataset[int(idx)][negative_col],
                'positive_sim': float(positive_sims[idx]),
                'negative_sim': float(negative_sims[idx]),
                'margin': float(positive_sims[idx] - negative_sims[idx])
            })

        # Log summary
        logger.info(f"\nHard negatives: {results['hard_negatives']['count']} (mean sim: {results['hard_negatives']['mean_sim']:.4f})")
        logger.info(f"Hard positives: {results['hard_positives']['count']} (mean sim: {results['hard_positives']['mean_sim']:.4f})")
        logger.info(f"Failure cases: {results['failure_cases']['count']} ({results['failure_cases']['rate']:.2%})")
        logger.info(f"Boundary cases: {results['boundary_cases']['count']} ({results['boundary_cases']['rate']:.2%})")

        # Save to CSV if configured
        if impact_config.get('save_to_csv', True):
            output_dir = Path(self.config['output']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / impact_config.get('csv_path', 'sample_impact_analysis.csv')

            df = pd.DataFrame(sample_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"✓ Sample impact analysis saved: {csv_path}")
            results['csv_path'] = str(csv_path)

        return results

    def log_to_mlflow(self, results: Dict, checkpoint_name: str = None):
        """Log analysis results to MLflow."""
        if not MLFLOW_AVAILABLE or not self.config.get('mlflow', {}).get('use_mlflow', False):
            return

        mlflow_config = self.config['mlflow']

        try:
            # Set experiment
            experiment_name = mlflow_config.get('experiment_name', 'embedding-quality-analysis')
            mlflow.set_experiment(experiment_name)

            # Start run
            run_name = mlflow_config.get('run_name')
            if checkpoint_name and not run_name:
                run_name = f"analysis_{checkpoint_name}"
            elif not run_name:
                run_name = f"analysis_{Path(self.config['model']['path']).name}"

            tags = mlflow_config.get('tags', {})
            if checkpoint_name:
                tags['checkpoint'] = checkpoint_name

            with mlflow.start_run(run_name=run_name, tags=tags):
                # Log parameters
                if mlflow_config.get('log_params', True):
                    mlflow.log_param('model_path', self.config['model']['path'])
                    mlflow.log_param('dataset', self.config['dataset']['name'])
                    mlflow.log_param('split', self.config['dataset']['split'])
                    mlflow.log_param('batch_size', self.config['analysis']['batch_size'])

                # Log similarity metrics
                if mlflow_config.get('log_metrics', True) and 'similarity' in results:
                    sim = results['similarity']
                    mlflow.log_metric('separation', sim['separation'])
                    mlflow.log_metric('overlap', sim['overlap'])
                    mlflow.log_metric('accuracy', sim['accuracy'])
                    mlflow.log_metric('positive_mean', sim['positive_mean'])
                    mlflow.log_metric('positive_std', sim['positive_std'])
                    mlflow.log_metric('negative_mean', sim['negative_mean'])
                    mlflow.log_metric('negative_std', sim['negative_std'])

                # Log embedding space metrics
                if mlflow_config.get('log_metrics', True) and 'space' in results:
                    space = results['space']
                    mlflow.log_metric('norm_mean', space['norm_mean'])
                    mlflow.log_metric('norm_std', space['norm_std'])
                    mlflow.log_metric('effective_dimensionality', space['effective_dimensionality'])
                    mlflow.log_metric('dimensionality_usage', space['dimensionality_usage'])

                # Log sample impact metrics
                if mlflow_config.get('log_sample_impact', True) and 'sample_impact' in results:
                    impact = results['sample_impact']
                    mlflow.log_metric('failure_rate', impact['failure_cases']['rate'])
                    mlflow.log_metric('boundary_rate', impact['boundary_cases']['rate'])
                    mlflow.log_metric('hard_negative_rate', impact['hard_negatives']['count'] / len(results.get('dataset_size', 1)))

                # Log artifacts
                if mlflow_config.get('log_artifacts', True):
                    output_dir = Path(self.config['output']['output_dir'])
                    if output_dir.exists():
                        # Log plots
                        plot_path = output_dir / 'embedding_quality_analysis.png'
                        if plot_path.exists():
                            mlflow.log_artifact(str(plot_path))

                        # Log report
                        report_path = output_dir / 'analysis_report.md'
                        if report_path.exists():
                            mlflow.log_artifact(str(report_path))

                        # Log CSV if exists
                        if 'sample_impact' in results and 'csv_path' in results['sample_impact']:
                            csv_path = results['sample_impact']['csv_path']
                            if Path(csv_path).exists():
                                mlflow.log_artifact(csv_path)

                logger.info("✓ Results logged to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def discover_checkpoints(self) -> List[str]:
        """Auto-discover checkpoint directories from training output."""
        checkpoint_dir = Path(self.config['model'].get('checkpoint_dir', ''))
        if not checkpoint_dir.exists():
            return []

        # Find all checkpoint-* directories
        checkpoints = sorted(checkpoint_dir.glob('checkpoint-*'))
        checkpoint_paths = [str(cp) for cp in checkpoints]

        # Add final model if it exists
        if checkpoint_dir.exists() and (checkpoint_dir / 'config.json').exists():
            checkpoint_paths.append(str(checkpoint_dir))

        return checkpoint_paths

    def run_single_analysis(self, model_path: str = None, checkpoint_name: str = None) -> Dict:
        """Run analysis on a single model/checkpoint."""
        if model_path:
            self.config['model']['path'] = model_path

        # Load model
        self.load_model()

        # Load dataset
        dataset = self.load_dataset()

        # Encode all data once (for efficiency)
        logger.info("\nEncoding dataset for analysis...")
        batch_size = self.config['analysis']['batch_size']

        anchors = dataset[self.config['dataset']['anchor_column']]
        positives = dataset[self.config['dataset']['positive_column']]
        negatives = dataset[self.config['dataset']['negative_column']]

        anchor_embs = self.model.encode(anchors, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        positive_embs = self.model.encode(positives, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        negative_embs = self.model.encode(negatives, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

        # Compute similarities
        positive_sims = self._batch_cosine_similarity(anchor_embs, positive_embs)
        negative_sims = self._batch_cosine_similarity(anchor_embs, negative_embs)

        # Run similarity analysis (using pre-computed embeddings)
        similarity_results = {
            'positive_mean': float(np.mean(positive_sims)),
            'positive_std': float(np.std(positive_sims)),
            'positive_min': float(np.min(positive_sims)),
            'positive_max': float(np.max(positive_sims)),
            'negative_mean': float(np.mean(negative_sims)),
            'negative_std': float(np.std(negative_sims)),
            'negative_min': float(np.min(negative_sims)),
            'negative_max': float(np.max(negative_sims)),
            'separation': float(np.mean(positive_sims) - np.mean(negative_sims)),
            'overlap': self._compute_overlap(positive_sims, negative_sims),
            'accuracy': float(np.mean(positive_sims > negative_sims)),
            'positive_sims': positive_sims,
            'negative_sims': negative_sims
        }

        # Run embedding space analysis
        space_results = self.analyze_embedding_space(dataset)

        # Run sample impact analysis
        sample_impact_results = {}
        if self.config['analysis'].get('run_sample_impact_analysis', True):
            sample_impact_results = self.analyze_sample_impact(
                dataset, anchor_embs, positive_embs, negative_embs,
                positive_sims, negative_sims
            )

        # Combine all results
        all_results = {
            'similarity': similarity_results,
            'space': space_results,
            'sample_impact': sample_impact_results,
            'dataset_size': len(dataset),
            'model_path': self.config['model']['path']
        }

        # Generate visualizations
        if self.config['output'].get('generate_plots', True):
            self.generate_visualizations(similarity_results, space_results)

        # Generate report
        if self.config['output'].get('generate_report', True):
            self.generate_report(similarity_results, space_results)

        # Log to MLflow
        self.log_to_mlflow(all_results, checkpoint_name)

        return all_results

    def run_analysis(self):
        """Run complete analysis pipeline (single or multi-checkpoint)."""
        logger.info("\n" + "="*70)
        logger.info("EMBEDDING QUALITY ANALYSIS")
        logger.info("="*70)

        # Check if multi-checkpoint analysis is requested
        checkpoints = self.config['model'].get('checkpoints')
        auto_discover = self.config['model'].get('auto_discover_checkpoints', False)

        if auto_discover:
            checkpoints = self.discover_checkpoints()
            if checkpoints:
                logger.info(f"Auto-discovered {len(checkpoints)} checkpoints")

        # Single model analysis
        if not checkpoints:
            results = self.run_single_analysis()

            logger.info("\n" + "="*70)
            logger.info("✓ ANALYSIS COMPLETE!")
            logger.info("="*70)
            logger.info(f"\nResults saved to: {self.config['output']['output_dir']}")
            return results

        # Multi-checkpoint analysis
        logger.info(f"\n Running analysis on {len(checkpoints)} checkpoints...")

        all_checkpoint_results = []
        for i, checkpoint_path in enumerate(checkpoints, 1):
            checkpoint_name = Path(checkpoint_path).name
            logger.info(f"\n{'='*70}")
            logger.info(f"CHECKPOINT {i}/{len(checkpoints)}: {checkpoint_name}")
            logger.info(f"{'='*70}")

            # Update output directory for this checkpoint
            base_output_dir = self.config['output']['output_dir']
            checkpoint_output_dir = Path(base_output_dir) / checkpoint_name
            self.config['output']['output_dir'] = str(checkpoint_output_dir)

            # Run analysis
            results = self.run_single_analysis(checkpoint_path, checkpoint_name)
            results['checkpoint_name'] = checkpoint_name
            all_checkpoint_results.append(results)

        # Generate comparison report
        if self.config.get('checkpoint_comparison', {}).get('plot_progression', True):
            self.generate_checkpoint_comparison(all_checkpoint_results)

        logger.info("\n" + "="*70)
        logger.info("✓ MULTI-CHECKPOINT ANALYSIS COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {base_output_dir}")

        return all_checkpoint_results

    def generate_checkpoint_comparison(self, all_results: List[Dict]):
        """Generate comparison plots and reports across checkpoints."""
        logger.info("\n" + "="*70)
        logger.info("GENERATING CHECKPOINT COMPARISON")
        logger.info("="*70)

        comparison_config = self.config.get('checkpoint_comparison', {})
        track_metrics = comparison_config.get('track_metrics', [
            'separation', 'overlap', 'accuracy', 'effective_dimensionality',
            'embedding_norm_mean', 'failure_rate'
        ])

        # Extract metrics for plotting
        checkpoint_names = [r['checkpoint_name'] for r in all_results]
        checkpoint_indices = list(range(len(checkpoint_names)))

        # Prepare data
        metrics_data = {metric: [] for metric in track_metrics}

        for result in all_results:
            sim = result.get('similarity', {})
            space = result.get('space', {})
            impact = result.get('sample_impact', {})

            # Map config metrics to actual result keys
            metric_mapping = {
                'separation': sim.get('separation', 0),
                'overlap': sim.get('overlap', 0),
                'accuracy': sim.get('accuracy', 0),
                'effective_dimensionality': space.get('effective_dimensionality', 0),
                'embedding_norm_mean': space.get('norm_mean', 0),
                'failure_rate': impact.get('failure_cases', {}).get('rate', 0) if impact else 0,
            }

            for metric in track_metrics:
                metrics_data[metric].append(metric_mapping.get(metric, 0))

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(track_metrics):
            if i >= len(axes):
                break

            ax = axes[i]
            ax.plot(checkpoint_indices, metrics_data[metric], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Checkpoint', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Progression', fontsize=14, fontweight='bold')
            ax.set_xticks(checkpoint_indices)
            ax.set_xticklabels([f"CP{i}" for i in checkpoint_indices], rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        base_output_dir = Path(self.config['output']['output_dir']).parent
        output_path = base_output_dir / 'checkpoint_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Checkpoint comparison saved: {output_path}")
        plt.close()

        # Create comparison table
        if comparison_config.get('log_comparison_table', True):
            comparison_df = pd.DataFrame({
                'Checkpoint': checkpoint_names,
                **{metric: metrics_data[metric] for metric in track_metrics}
            })

            csv_path = base_output_dir / 'checkpoint_comparison.csv'
            comparison_df.to_csv(csv_path, index=False)
            logger.info(f"✓ Comparison table saved: {csv_path}")

            # Log to MLflow if enabled
            if MLFLOW_AVAILABLE and self.config.get('mlflow', {}).get('use_mlflow', False):
                try:
                    mlflow_config = self.config['mlflow']
                    mlflow.set_experiment(mlflow_config.get('experiment_name', 'embedding-quality-analysis'))
                    with mlflow.start_run(run_name="checkpoint_comparison", tags={'analysis_type': 'comparison'}):
                        mlflow.log_artifact(str(csv_path))
                        mlflow.log_artifact(str(output_path))
                        logger.info("✓ Comparison logged to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log comparison to MLflow: {e}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Analyze embedding model quality")
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/analysis_config.yaml',
        help='Path to analysis configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Override model path from config'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override model path if provided
    if args.model:
        config['model']['path'] = args.model
        # Extract model name for report
        config['model']['name'] = Path(args.model).name

    # Run analysis
    analyzer = EmbeddingQualityAnalyzer(config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
