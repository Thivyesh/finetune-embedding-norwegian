"""
Custom callback for logging embedding-specific metrics during training.

This callback extends the default logging to include metrics that are particularly
important for understanding embedding model performance:
1. Gradient norms (convergence indicators)
2. Embedding space statistics (norm distribution, diversity)
3. Similarity distributions (positive vs negative pairs)
4. Model health metrics (weight updates, learning dynamics)

References:
- MLflow Sentence Transformers: https://mlflow.org/docs/latest/llms/sentence-transformers/index.html
- HuggingFace Callbacks: https://huggingface.co/docs/transformers/main_classes/callback
- Embedding Metrics Research: https://arxiv.org/html/2403.05440v1
"""

from transformers import TrainerCallback
import torch
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingMetricsCallback(TrainerCallback):
    """
    Custom callback to log embedding-specific metrics during training.

    This callback computes and logs additional metrics beyond standard loss/accuracy:
    - Gradient norms (L2 norm of gradients)
    - Embedding norms (magnitude distribution)
    - Weight update ratios (how much weights change per step)
    - Similarity statistics (if eval dataset available)
    """

    def __init__(self, log_frequency: int = 100):
        """
        Initialize the callback.

        Args:
            log_frequency: How often to log metrics (in training steps)
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.previous_weights = None

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step.

        Logs gradient norms and weight statistics.
        """
        if state.global_step % self.log_frequency != 0:
            return

        model = kwargs.get('model')
        if model is None:
            return

        try:
            metrics = {}

            # 1. GRADIENT NORMS
            # Track gradient magnitudes to understand convergence
            # Research shows gradient norm is inversely related to embedding norm
            # Reference: https://arxiv.org/pdf/2502.09252
            gradient_norms = self._compute_gradient_norms(model)
            metrics.update(gradient_norms)

            # 2. WEIGHT UPDATE RATIOS
            # Ratio of weight change to weight magnitude
            # Helps identify if learning is stagnating
            if self.previous_weights is not None:
                update_ratios = self._compute_weight_update_ratios(model)
                metrics.update(update_ratios)

            # Store current weights for next comparison
            self.previous_weights = {
                name: param.data.clone().detach()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

            # 3. EMBEDDING LAYER STATISTICS
            # Track embedding norm distribution
            # Important for understanding embedding space geometry
            embedding_stats = self._compute_embedding_stats(model)
            metrics.update(embedding_stats)

            # Log to MLflow via the trainer's log mechanism
            # This integrates with existing MLflow tracking
            if metrics:
                # Add prefix to distinguish from default metrics
                # IMPORTANT: Use dots instead of slashes - MLflow stores "/" as nested directories
                # which the UI can't read properly. Dots create flat files that show in UI.
                prefixed_metrics = {f"custom.{k.replace('/', '.')}": v for k, v in metrics.items()}

                # CRITICAL: Actually log to trainer state
                # This will be picked up by MLflow callback
                try:
                    import mlflow
                    if mlflow.active_run():
                        mlflow.log_metrics(prefixed_metrics, step=state.global_step)
                except Exception as e:
                    logger.debug(f"MLflow logging failed: {e}")

                # Also log for debugging
                for key, value in prefixed_metrics.items():
                    if not np.isnan(value) and not np.isinf(value):
                        logger.info(f"Step {state.global_step}: {key} = {value:.6f}")

        except Exception as e:
            logger.warning(f"Failed to compute custom metrics: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation.

        Can add custom evaluation metrics here.
        """
        if metrics is None:
            return

        try:
            model = kwargs.get('model')
            if model is None:
                return

            # Add embedding space quality metrics during evaluation
            eval_metrics = {}

            # Model health indicators
            model_stats = self._compute_model_health(model)
            eval_metrics.update(model_stats)

            # Log to MLflow directly
            try:
                import mlflow
                if mlflow.active_run():
                    # Use dots instead of slashes for MLflow UI compatibility
                    prefixed = {f"custom.{k.replace('/', '.')}": v for k, v in eval_metrics.items()}
                    mlflow.log_metrics(prefixed, step=state.global_step)
            except Exception as e:
                logger.debug(f"MLflow eval logging failed: {e}")

            # Add to metrics dict with custom prefix (for HF trainer logging)
            for key, value in eval_metrics.items():
                if not np.isnan(value) and not np.isinf(value):
                    # Use dots for consistency
                    metric_name = f"custom.{key.replace('/', '.')}"
                    metrics[metric_name] = value
                    logger.info(f"Eval step {state.global_step}: {metric_name} = {value:.6f}")

        except Exception as e:
            logger.warning(f"Failed to compute custom eval metrics: {e}")

    def _compute_gradient_norms(self, model) -> Dict[str, float]:
        """
        Compute L2 norms of gradients for different parameter groups.

        Gradient norms help understand:
        - Convergence speed (decreasing norms = converging)
        - Gradient vanishing/explosion
        - Layer-wise learning dynamics

        Returns:
            Dictionary with gradient norm statistics
        """
        metrics = {}

        total_norm = 0.0
        encoder_norm = 0.0
        pooler_norm = 0.0
        num_params = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1

                # Track different components separately
                if 'encoder' in name:
                    encoder_norm += param_norm ** 2
                elif 'pooler' in name or 'pooling' in name:
                    pooler_norm += param_norm ** 2

        if num_params > 0:
            metrics['gradient_norm/total'] = np.sqrt(total_norm)
            metrics['gradient_norm/mean'] = np.sqrt(total_norm / num_params)

            if encoder_norm > 0:
                metrics['gradient_norm/encoder'] = np.sqrt(encoder_norm)
            if pooler_norm > 0:
                metrics['gradient_norm/pooler'] = np.sqrt(pooler_norm)

        return metrics

    def _compute_weight_update_ratios(self, model) -> Dict[str, float]:
        """
        Compute ratio of weight updates to weight magnitudes.

        Update ratio = ||w_new - w_old|| / ||w_old||

        Small ratios (<0.001) may indicate:
        - Learning has stagnated
        - Learning rate too low
        - Local minimum reached

        Large ratios (>0.1) may indicate:
        - Learning rate too high
        - Unstable training

        Returns:
            Dictionary with update ratio statistics
        """
        metrics = {}

        total_update_ratio = 0.0
        num_layers = 0

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.previous_weights:
                continue

            old_weight = self.previous_weights[name]
            update = (param.data - old_weight).norm(2).item()
            weight_norm = param.data.norm(2).item()

            if weight_norm > 1e-8:  # Avoid division by zero
                ratio = update / weight_norm
                total_update_ratio += ratio
                num_layers += 1

        if num_layers > 0:
            metrics['weight_update/mean_ratio'] = total_update_ratio / num_layers

        return metrics

    def _compute_embedding_stats(self, model) -> Dict[str, float]:
        """
        Compute statistics about the embedding layer.

        Embedding norm distribution is crucial for embedding models:
        - High norms can slow convergence (quadratic relationship)
        - Norms should be relatively stable, not growing unbounded
        - Distribution width indicates embedding diversity

        Reference: https://arxiv.org/pdf/2502.09252

        Returns:
            Dictionary with embedding statistics
        """
        metrics = {}

        try:
            # Access the embedding layer (structure may vary by model)
            # For BERT-style models, embeddings are usually in model[0].auto_model.embeddings
            if hasattr(model, 'module'):  # Unwrap DDP
                base_model = model.module
            else:
                base_model = model

            # Try to find embedding layer
            embedding_layer = None
            if hasattr(base_model, '_first_module'):
                auto_model = base_model._first_module().auto_model
                if hasattr(auto_model, 'embeddings'):
                    embedding_layer = auto_model.embeddings.word_embeddings

            if embedding_layer is not None and hasattr(embedding_layer, 'weight'):
                embeddings = embedding_layer.weight.data

                # Compute norms for each embedding
                norms = embeddings.norm(2, dim=1)

                metrics['embedding/norm_mean'] = norms.mean().item()
                metrics['embedding/norm_std'] = norms.std().item()
                metrics['embedding/norm_max'] = norms.max().item()
                metrics['embedding/norm_min'] = norms.min().item()

                # Embedding diversity: ratio of std to mean
                # Higher = more diverse embeddings
                if metrics['embedding/norm_mean'] > 1e-8:
                    metrics['embedding/diversity_ratio'] = (
                        metrics['embedding/norm_std'] / metrics['embedding/norm_mean']
                    )

        except Exception as e:
            logger.debug(f"Could not compute embedding stats: {e}")

        return metrics

    def _compute_model_health(self, model) -> Dict[str, float]:
        """
        Compute overall model health indicators.

        Useful for detecting:
        - NaN/Inf in weights (training collapse)
        - Dead neurons (zero gradients)
        - Weight saturation

        Returns:
            Dictionary with model health metrics
        """
        metrics = {}

        total_params = 0
        zero_grad_params = 0
        large_weight_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            total_params += param.numel()

            # Check for dead neurons (always zero gradient)
            if param.grad is not None:
                zero_grads = (param.grad.abs() < 1e-8).sum().item()
                zero_grad_params += zero_grads

            # Check for saturated weights
            large_weights = (param.data.abs() > 10.0).sum().item()
            large_weight_params += large_weights

        if total_params > 0:
            metrics['model_health/dead_params_ratio'] = zero_grad_params / total_params
            metrics['model_health/large_weights_ratio'] = large_weight_params / total_params

        return metrics


class SimilarityDistributionCallback(TrainerCallback):
    """
    Track similarity distributions during evaluation.

    This callback samples triplets from the evaluation set and computes:
    - Cosine similarity between anchor-positive pairs
    - Cosine similarity between anchor-negative pairs
    - Distribution statistics (mean, std, overlap)

    Useful for understanding if the model is learning to distinguish
    positive from negative examples.
    """

    def __init__(self, num_samples: int = 1000):
        """
        Initialize callback.

        Args:
            num_samples: Number of triplets to sample for similarity analysis
        """
        super().__init__()
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation.

        Computes similarity distributions.
        """
        if metrics is None:
            return

        try:
            model = kwargs.get('model')
            eval_dataset = kwargs.get('eval_dataset')

            if model is None or eval_dataset is None:
                return

            # Sample random triplets
            num_eval = len(eval_dataset)
            sample_indices = np.random.choice(
                num_eval,
                size=min(self.num_samples, num_eval),
                replace=False
            )

            positive_sims = []
            negative_sims = []

            # Set model to eval mode
            model.eval()

            with torch.no_grad():
                for idx in sample_indices:
                    triplet = eval_dataset[int(idx)]

                    # Get embeddings (assuming dataset returns dict with anchor/positive/negative)
                    if isinstance(triplet, dict):
                        anchor_text = triplet.get('anchor', '')
                        positive_text = triplet.get('positive', '')
                        negative_text = triplet.get('negative', '')

                        if anchor_text and positive_text and negative_text:
                            # Encode texts - handle wrapped model
                            # Model might be wrapped in DDP or other wrappers
                            base_model = model.module if hasattr(model, 'module') else model

                            # Encode using the SentenceTransformer
                            anchor_emb = base_model.encode([anchor_text], convert_to_numpy=True)[0]
                            positive_emb = base_model.encode([positive_text], convert_to_numpy=True)[0]
                            negative_emb = base_model.encode([negative_text], convert_to_numpy=True)[0]

                            # Compute cosine similarities
                            pos_sim = self._cosine_similarity(anchor_emb, positive_emb)
                            neg_sim = self._cosine_similarity(anchor_emb, negative_emb)

                            positive_sims.append(pos_sim)
                            negative_sims.append(neg_sim)

            if positive_sims and negative_sims:
                # Compute distribution statistics
                sim_metrics = {
                    'custom/similarity/positive_mean': np.mean(positive_sims),
                    'custom/similarity/positive_std': np.std(positive_sims),
                    'custom/similarity/negative_mean': np.mean(negative_sims),
                    'custom/similarity/negative_std': np.std(negative_sims),
                    'custom/similarity/separation': np.mean(positive_sims) - np.mean(negative_sims),
                    'custom/similarity/overlap': self._compute_distribution_overlap(positive_sims, negative_sims)
                }

                # Log to MLflow directly
                try:
                    import mlflow
                    if mlflow.active_run():
                        mlflow.log_metrics(sim_metrics, step=state.global_step)
                except Exception as e:
                    logger.debug(f"MLflow similarity logging failed: {e}")

                # Also add to metrics dict
                metrics.update(sim_metrics)

                # Log for visibility
                for key, value in sim_metrics.items():
                    logger.info(f"Eval step {state.global_step}: {key} = {value:.4f}")

        except Exception as e:
            logger.warning(f"Failed to compute similarity distributions: {e}")

    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    @staticmethod
    def _compute_distribution_overlap(pos_sims, neg_sims):
        """
        Compute overlap between positive and negative similarity distributions.

        Uses simple threshold-based approach:
        - Lower overlap = better separation
        - Overlap = 0: perfect separation
        - Overlap = 1: complete overlap
        """
        pos_min = np.min(pos_sims)
        neg_max = np.max(neg_sims)

        # If no overlap at all
        if pos_min > neg_max:
            return 0.0

        # Compute fraction of overlap
        pos_range = np.max(pos_sims) - np.min(pos_sims)
        overlap_amount = pos_min - neg_max  # Negative if overlap

        if pos_range > 1e-8:
            return max(0.0, min(1.0, -overlap_amount / pos_range))

        return 1.0  # Complete overlap if no variance
