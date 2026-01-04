#!/usr/bin/env python3
"""
Checkpoint Degradation Analysis Script

This script analyzes Stage 2 (Group B) training checkpoints to identify:
1. Which evaluation samples caused metric degradation
2. Which datasets are most affected
3. Patterns in overfitting behavior

Usage:
    # Analyze dev set (used during training evaluation)
    uv run python scripts/analyze_checkpoint_degradation.py \
        --checkpoints models/norbert4-v6-stage2-group-b/checkpoint-{500,1500,2000} \
        --split dev \
        --output-dir results/checkpoint_analysis_dev

    # Analyze test set (final evaluation)
    uv run python scripts/analyze_checkpoint_degradation.py \
        --checkpoints models/norbert4-v6-stage2-group-b/checkpoint-{500,1500,2000} \
        --split test \
        --output-dir results/checkpoint_analysis_test
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader_paws import load_paws_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointAnalyzer:
    """Main orchestrator for checkpoint degradation analysis."""

    def __init__(self, checkpoint_paths: List[str], output_dir: Path, device: str = 'mps'):
        self.checkpoint_paths = checkpoint_paths
        self.output_dir = Path(output_dir)
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checkpoint Analyzer initialized")
        logger.info(f"Checkpoints: {len(checkpoint_paths)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def run_analysis(self, split: str = 'dev'):
        """Run complete checkpoint degradation analysis.

        Args:
            split: Which split to analyze - 'dev' or 'test'
        """
        logger.info("\n" + "="*70)
        logger.info(f"CHECKPOINT DEGRADATION ANALYSIS ({split.upper()} SET)")
        logger.info("="*70 + "\n")

        # Step 1: Rebuild eval data with source labels
        logger.info(f"Step 1: Loading evaluation data with source labels ({split} set)...")
        eval_dataset = self.rebuild_eval_with_labels(split=split)
        logger.info(f"✓ Loaded {len(eval_dataset):,} evaluation samples from {split} set")

        # Step 2: Per-sample evaluation across all checkpoints
        logger.info("\nStep 2: Evaluating all checkpoints...")
        df_results = self.evaluate_all_checkpoints(eval_dataset)
        logger.info(f"✓ Evaluated {len(df_results):,} samples across {len(self.checkpoint_paths)} checkpoints")

        # Step 3: Identify problematic samples
        logger.info("\nStep 3: Identifying problematic samples...")
        df_problematic = self.identify_problematic_samples(df_results)
        logger.info(f"✓ Found {len(df_problematic):,} problematic samples")

        # Step 4: Dataset breakdown
        logger.info("\nStep 4: Computing dataset breakdown...")
        df_breakdown = self.compute_dataset_breakdown(df_results)
        logger.info(f"✓ Analyzed {len(df_breakdown)} datasets")

        # Step 5: Generate reports
        logger.info("\nStep 5: Generating reports...")
        self.generate_reports(df_results, df_problematic, df_breakdown)
        logger.info(f"✓ Reports saved to {self.output_dir}")

        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info(f"  - summary_report.txt")
        logger.info(f"  - per_sample_scores.csv ({len(df_results):,} samples)")
        logger.info(f"  - degraded_samples.csv ({len(df_problematic):,} samples)")
        logger.info(f"  - dataset_breakdown.csv ({len(df_breakdown)} datasets)")

    def rebuild_eval_with_labels(self, split: str = 'test') -> Dataset:
        """
        Rebuild Group B evaluation data with source dataset labels.

        Replicates the logic from utils/data_loader_group_b.py but adds:
        - source_dataset: Which dataset the sample came from
        - sample_id: Unique identifier for tracking

        Args:
            split: Which split to load - 'dev' or 'test'

        Returns:
            Dataset with columns: ['query', 'positive', 'source_dataset', 'sample_id']
        """
        all_eval = []

        # 1. NorQuAD
        logger.info(f"  Loading NorQuAD {split} set...")
        norquad = load_dataset('ltg/norquad', trust_remote_code=True)
        if split == 'dev':
            norquad_eval = norquad['validation']
        else:
            norquad_eval = norquad['test']
        norquad_eval = norquad_eval.map(
            lambda x: {
                'query': x['question'],
                'positive': x['context'],
                'source_dataset': 'NorQuAD'
            },
            remove_columns=[col for col in norquad_eval.column_names if col not in ['question', 'context']]
        )
        logger.info(f"    ✓ NorQuAD: {len(norquad_eval):,} samples")
        all_eval.append(norquad_eval)

        # 2. NorOpenBookQA
        logger.info(f"  Loading NorOpenBookQA {split} set...")
        openbookqa = load_dataset('ltg/noropenbookqa', 'nb', trust_remote_code=True)
        # Split test set in half (dev/test)
        test_size = len(openbookqa['test'])
        dev_size = test_size // 2
        if split == 'dev':
            openbookqa_eval = openbookqa['test'].select(range(dev_size))
        else:
            openbookqa_eval = openbookqa['test'].select(range(dev_size, test_size))
        openbookqa_eval = openbookqa_eval.map(
            lambda x: {
                'query': x['question_stem'],
                'positive': x['fact'] if x.get('fact') else x['question_stem'],
                'source_dataset': 'OpenBookQA'
            },
            remove_columns=[col for col in openbookqa_eval.column_names if col not in []]
        )
        logger.info(f"    ✓ OpenBookQA: {len(openbookqa_eval):,} samples")
        all_eval.append(openbookqa_eval)

        # 3. Supervised-DA (Danish)
        logger.info(f"  Loading Supervised-DA {split} set...")
        supervised = load_dataset('jealk/supervised-da', split='train', trust_remote_code=True)
        total = len(supervised)
        train_size = int(0.8 * total)
        dev_size = int(0.1 * total)
        if split == 'dev':
            supervised_eval = supervised.select(range(train_size, train_size + dev_size))
        else:
            supervised_eval = supervised.select(range(train_size + dev_size, total))
        supervised_eval = supervised_eval.map(
            lambda x: {
                'query': x['query'],
                'positive': x['pos'],
                'source_dataset': 'Supervised-DA'
            },
            remove_columns=[col for col in supervised_eval.column_names if col not in []]
        )
        logger.info(f"    ✓ Supervised-DA: {len(supervised_eval):,} samples")
        all_eval.append(supervised_eval)

        # 4. PAWS-X (paraphrases only)
        logger.info(f"  Loading PAWS-X {split} set (paraphrases only)...")
        try:
            paws_train, paws_dev, paws_test_raw = load_paws_dataset(data_dir='data/paws')
            if split == 'dev':
                paws_eval_raw = paws_dev
            else:
                paws_eval_raw = paws_test_raw
            # Filter to paraphrases only (label=1)
            paws_eval = paws_eval_raw.filter(lambda x: x['label'] == 1)
            paws_eval = paws_eval.map(
                lambda x: {
                    'query': x['sentence1'],
                    'positive': x['sentence2'],
                    'source_dataset': 'PAWS'
                },
                remove_columns=[col for col in paws_eval.column_names if col not in []]
            )
            logger.info(f"    ✓ PAWS: {len(paws_eval):,} samples")
            all_eval.append(paws_eval)
        except Exception as e:
            logger.warning(f"    ⚠ Failed to load PAWS: {e}")
            logger.warning(f"    Continuing without PAWS data...")

        # Combine all datasets
        logger.info("  Combining all datasets...")
        combined = concatenate_datasets(all_eval)

        # Add sample IDs
        combined = combined.map(
            lambda x, idx: {**x, 'sample_id': f"sample_{idx}"},
            with_indices=True
        )

        # Verify total count
        logger.info(f"  ✓ Total combined: {len(combined):,} samples ({split} set)")
        logger.info(f"  Dataset distribution:")
        for dataset_name in combined.unique('source_dataset'):
            count = len(combined.filter(lambda x: x['source_dataset'] == dataset_name))
            pct = count / len(combined) * 100
            logger.info(f"    - {dataset_name}: {count:,} ({pct:.1f}%)")

        return combined

    def evaluate_all_checkpoints(self, eval_dataset: Dataset) -> pd.DataFrame:
        """
        Evaluate all checkpoints using memory-efficient load-encode-unload pattern.

        For each checkpoint:
        1. Load model
        2. Encode all queries and positive documents
        3. Compute cosine similarities
        4. Unload model and clear cache

        Returns:
            DataFrame with columns: sample_id, source_dataset, query, positive,
                                   ckpt{N}_similarity for each checkpoint
        """
        # Initialize results DataFrame
        results = {
            'sample_id': eval_dataset['sample_id'],
            'source_dataset': eval_dataset['source_dataset'],
            'query': eval_dataset['query'],
            'positive': eval_dataset['positive'],
        }

        # Evaluate each checkpoint
        for ckpt_path in self.checkpoint_paths:
            ckpt_name = Path(ckpt_path).name
            logger.info(f"\n  Evaluating {ckpt_name}...")

            # Load model
            model = SentenceTransformer(ckpt_path, device=self.device, trust_remote_code=True)
            logger.info(f"    Loaded model to {self.device}")

            # Compute similarities for all samples
            similarities = []
            batch_size = 128

            for i in tqdm(range(0, len(eval_dataset), batch_size), desc=f"    {ckpt_name}"):
                batch = eval_dataset[i:i+batch_size]
                queries = batch['query']
                positives = batch['positive']

                # Encode
                query_embs = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
                positive_embs = model.encode(positives, convert_to_numpy=True, show_progress_bar=False)

                # Compute cosine similarities (pairwise between corresponding query-positive)
                batch_sims = []
                for q_emb, p_emb in zip(query_embs, positive_embs):
                    sim = cosine_similarity([q_emb], [p_emb])[0][0]
                    batch_sims.append(sim)

                similarities.extend(batch_sims)

            # Store results
            results[f'{ckpt_name}_similarity'] = similarities

            # Compute summary stats
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            median_sim = np.median(similarities)
            logger.info(f"    ✓ Mean similarity: {mean_sim:.4f} (±{std_sim:.4f}), Median: {median_sim:.4f}")

            # Unload model
            del model
            if self.device == 'mps':
                torch.mps.empty_cache()
            elif self.device == 'cuda':
                torch.cuda.empty_cache()

        # Create DataFrame
        df = pd.DataFrame(results)

        # Compute degradation (difference between best and worst checkpoint)
        checkpoint_cols = [col for col in df.columns if col.endswith('_similarity')]
        if len(checkpoint_cols) >= 2:
            # Find best and worst checkpoint for each sample
            checkpoint_sims = df[checkpoint_cols].values
            df['best_similarity'] = checkpoint_sims.max(axis=1)
            df['worst_similarity'] = checkpoint_sims.min(axis=1)
            df['degradation'] = df['best_similarity'] - df['worst_similarity']

        return df

    def identify_problematic_samples(
        self,
        df: pd.DataFrame,
        large_drop_threshold: float = 0.1,
        low_quality_threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Identify problematic samples using multi-criteria detection.

        Criteria:
        1. Large drop: degradation > threshold (default 0.1 = 10%)
        2. Low quality: worst checkpoint similarity < threshold (default 0.8)
        3. Monotonic decline: scores drop at every checkpoint

        Returns:
            DataFrame of problematic samples sorted by degradation
        """
        # Get checkpoint columns
        checkpoint_cols = sorted([col for col in df.columns if col.endswith('_similarity')])

        if len(checkpoint_cols) < 2:
            logger.warning("Not enough checkpoints to identify degradation patterns")
            return df.head(0)  # Return empty DataFrame

        # Criteria 1: Large drop
        large_drop = df['degradation'] > large_drop_threshold

        # Criteria 2: Low quality
        low_quality = df['worst_similarity'] < low_quality_threshold

        # Criteria 3: Monotonic decline
        monotonic = pd.Series([False] * len(df), index=df.index)
        if len(checkpoint_cols) >= 3:
            # Check if similarity decreases across checkpoints
            for i in range(len(df)):
                sims = [df.iloc[i][col] for col in checkpoint_cols]
                is_monotonic = all(sims[j] >= sims[j+1] for j in range(len(sims)-1))
                monotonic.iloc[i] = is_monotonic

        # Categorize samples
        df = df.copy()
        df['category'] = 'normal'
        df.loc[large_drop & low_quality & monotonic, 'category'] = 'severe'
        df.loc[large_drop & ~(low_quality & monotonic), 'category'] = 'large_drop'
        df.loc[low_quality & monotonic & ~large_drop, 'category'] = 'monotonic_decline'

        # Filter to problematic samples
        problematic = df[large_drop | (low_quality & monotonic)].copy()
        problematic = problematic.sort_values('degradation', ascending=False)

        # Log summary
        logger.info(f"  Identified {len(problematic):,} problematic samples:")
        logger.info(f"    - Severe: {len(problematic[problematic['category'] == 'severe']):,}")
        logger.info(f"    - Large drop: {len(problematic[problematic['category'] == 'large_drop']):,}")
        logger.info(f"    - Monotonic decline: {len(problematic[problematic['category'] == 'monotonic_decline']):,}")

        return problematic

    def compute_dataset_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute performance breakdown by source dataset.

        Returns:
            DataFrame with mean/std for each checkpoint grouped by source_dataset
        """
        checkpoint_cols = [col for col in df.columns if col.endswith('_similarity')]

        # Group by dataset and compute statistics
        breakdown = df.groupby('source_dataset').agg({
            **{col: ['mean', 'std', 'min', 'max'] for col in checkpoint_cols},
            'degradation': ['mean', 'std', 'max'],
            'sample_id': 'count'
        })

        # Flatten column names
        breakdown.columns = ['_'.join(col).strip() for col in breakdown.columns.values]
        breakdown.rename(columns={'sample_id_count': 'count'}, inplace=True)
        breakdown.reset_index(inplace=True)

        return breakdown

    def generate_reports(
        self,
        df_results: pd.DataFrame,
        df_problematic: pd.DataFrame,
        df_breakdown: pd.DataFrame
    ):
        """Generate all analysis reports."""

        # 1. Save CSVs
        df_results.to_csv(self.output_dir / 'per_sample_scores.csv', index=False)
        df_problematic.to_csv(self.output_dir / 'degraded_samples.csv', index=False)
        df_breakdown.to_csv(self.output_dir / 'dataset_breakdown.csv', index=False)

        # 2. Generate text summary
        self._generate_text_summary(df_results, df_problematic, df_breakdown)

        # 3. Save metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'checkpoints': self.checkpoint_paths,
            'total_samples': len(df_results),
            'problematic_samples': len(df_problematic),
            'datasets_analyzed': list(df_breakdown['source_dataset'].values)
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _generate_text_summary(
        self,
        df_results: pd.DataFrame,
        df_problematic: pd.DataFrame,
        df_breakdown: pd.DataFrame
    ):
        """Generate human-readable text summary report."""

        checkpoint_cols = sorted([col for col in df_results.columns if col.endswith('_similarity')])

        lines = []
        lines.append("=" * 79)
        lines.append("CHECKPOINT DEGRADATION ANALYSIS REPORT")
        lines.append("=" * 79)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 1. Checkpoint Performance Summary
        lines.append("1. CHECKPOINT PERFORMANCE SUMMARY")
        lines.append("-" * 79)
        lines.append("Metric: Mean Cosine Similarity (higher is better)")
        lines.append("")
        lines.append(f"{'Checkpoint':<25} | {'Mean':<8} | {'Std Dev':<8} | {'Median':<8} | {'Min':<6} | {'Max':<6}")
        lines.append("-" * 79)

        for col in checkpoint_cols:
            ckpt_name = col.replace('_similarity', '')
            mean_val = df_results[col].mean()
            std_val = df_results[col].std()
            median_val = df_results[col].median()
            min_val = df_results[col].min()
            max_val = df_results[col].max()
            lines.append(f"{ckpt_name:<25} | {mean_val:.4f}   | {std_val:.4f}   | {median_val:.4f}   | {min_val:.2f} | {max_val:.2f}")

        if len(checkpoint_cols) >= 2:
            first_mean = df_results[checkpoint_cols[0]].mean()
            last_mean = df_results[checkpoint_cols[-1]].mean()
            degradation = first_mean - last_mean
            degradation_pct = (degradation / first_mean) * 100
            lines.append("")
            lines.append(f"Overall degradation: {degradation:.4f} ({degradation_pct:+.2f}%)")

        lines.append("")

        # 2. Dataset Breakdown
        lines.append("2. DATASET BREAKDOWN")
        lines.append("-" * 79)
        lines.append(f"{'Dataset':<15} | {'Count':<8} | {'First Ckpt':<10} | {'Last Ckpt':<10} | {'Degradation':<12}")
        lines.append("-" * 79)

        for _, row in df_breakdown.iterrows():
            dataset = row['source_dataset']
            count = int(row['count'])
            first_col = f"{checkpoint_cols[0]}_mean"
            last_col = f"{checkpoint_cols[-1]}_mean"
            first_mean = row[first_col]
            last_mean = row[last_col]
            deg = first_mean - last_mean
            lines.append(f"{dataset:<15} | {count:<8,} | {first_mean:.4f}     | {last_mean:.4f}     | {deg:+.4f}")

        # Find most affected dataset
        if len(df_breakdown) > 0:
            first_col = f"{checkpoint_cols[0]}_mean"
            last_col = f"{checkpoint_cols[-1]}_mean"
            df_breakdown['deg'] = df_breakdown[first_col] - df_breakdown[last_col]
            most_affected = df_breakdown.loc[df_breakdown['deg'].idxmax()]
            lines.append("")
            lines.append(f"INSIGHT: {most_affected['source_dataset']} shows highest degradation ({most_affected['deg']:+.4f})")

        lines.append("")

        # 3. Problematic Samples
        lines.append("3. PROBLEMATIC SAMPLES")
        lines.append("-" * 79)
        lines.append(f"Found {len(df_problematic):,} samples ({len(df_problematic)/len(df_results)*100:.1f}%) with significant degradation")
        lines.append("")

        if len(df_problematic) > 0:
            lines.append(f"Top 5 most degraded samples:")
            for i, (_, row) in enumerate(df_problematic.head(5).iterrows(), 1):
                lines.append(f"\n{i}. {row['sample_id']} ({row['source_dataset']}):")
                lines.append(f"   Best similarity: {row['best_similarity']:.4f}")
                lines.append(f"   Worst similarity: {row['worst_similarity']:.4f}")
                lines.append(f"   Degradation: {row['degradation']:.4f} ({row['degradation']/row['best_similarity']*100:.1f}%)")
                lines.append(f"   Query: {row['query'][:100]}...")
                lines.append(f"   Category: {row['category']}")

        lines.append("")
        lines.append("")

        # 4. Key Findings
        lines.append("4. KEY FINDINGS")
        lines.append("-" * 79)

        if len(checkpoint_cols) >= 2:
            overall_deg_pct = abs(degradation_pct)
            if overall_deg_pct < 1:
                lines.append("✓ Overall degradation is minimal (<1%)")
            elif overall_deg_pct < 3:
                lines.append("✓ Overall degradation is moderate (1-3%)")
            elif overall_deg_pct < 5:
                lines.append("⚠ Overall degradation is noticeable (3-5%)")
            else:
                lines.append("✗ Overall degradation is significant (>5%)")

        severe_count = len(df_problematic[df_problematic['category'] == 'severe'])
        if severe_count > 0:
            lines.append(f"✗ {severe_count:,} samples show severe degradation")

        problematic_pct = len(df_problematic) / len(df_results) * 100
        if problematic_pct > 5:
            lines.append(f"⚠ {problematic_pct:.1f}% of samples are problematic")

        # Overfitting assessment
        if len(checkpoint_cols) >= 3:
            # Check if later checkpoints consistently worse
            means = [df_results[col].mean() for col in checkpoint_cols]
            is_monotonic_decline = all(means[i] >= means[i+1] for i in range(len(means)-1))
            if is_monotonic_decline:
                lines.append("⚠ Monotonic performance decline suggests overfitting")

        lines.append("")

        # 5. Recommendations
        lines.append("5. RECOMMENDATIONS")
        lines.append("-" * 79)

        if len(checkpoint_cols) >= 2:
            best_ckpt = checkpoint_cols[0]
            best_mean = df_results[best_ckpt].mean()
            lines.append(f"1. Use {best_ckpt.replace('_similarity', '')} (best performance: {best_mean:.4f})")

        if problematic_pct > 3:
            lines.append(f"2. Review {most_affected['source_dataset']} training - needs better regularization")

        if overall_deg_pct > 2:
            lines.append("3. Consider retraining with:")
            lines.append("   - Lower learning rate (e.g., 1.0e-5)")
            lines.append("   - Increased early stopping patience (e.g., 5)")
            lines.append("   - Additional regularization (gradient clipping, higher weight decay)")
        else:
            lines.append(f"3. Degradation is acceptable - can proceed to Stage 3")

        lines.append("")
        lines.append("=" * 79)

        # Write to file
        summary_path = self.output_dir / 'summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"  ✓ Summary report written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze checkpoint degradation for Stage 2 training')
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        required=True,
        help='Paths to checkpoint directories to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis results (default: results/checkpoint_analysis_<split>_<date>)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['dev', 'test'],
        help='Which split to analyze - "dev" (used during training eval) or "test" (final eval)'
    )

    args = parser.parse_args()

    # Set default output dir if not specified
    if args.output_dir is None:
        args.output_dir = f'results/checkpoint_analysis_{args.split}_{datetime.now().strftime("%Y-%m-%d")}'

    # Validate checkpoints exist
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)

    # Run analysis
    analyzer = CheckpointAnalyzer(
        checkpoint_paths=args.checkpoints,
        output_dir=Path(args.output_dir),
        device=args.device
    )

    try:
        analyzer.run_analysis(split=args.split)
        logger.info("\n✅ Analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\n❌ Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
