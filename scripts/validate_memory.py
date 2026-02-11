#!/usr/bin/env python3
"""
GPU Memory Estimation Script for Transformer Training

This script estimates the GPU memory required for training based on:
- Model size (number of parameters)
- Batch size
- Sequence length
- Hidden dimensions
- Training configuration (precision, optimizer, checkpointing)

Based on formulas from:
- EleutherAI's "Transformer Math 101": https://blog.eleuther.ai/transformer-math/
- HuggingFace's Training Efficiency Guide: https://huggingface.co/docs/transformers/perf_train_gpu_one

Memory Components:
1. Model weights: 2-4 bytes per parameter (depends on precision)
2. Optimizer states: 8-12 bytes per parameter (AdamW)
3. Gradients: 2-4 bytes per parameter
4. Activations: scales with batch_size × seq_length × hidden_size × num_layers

Usage:
    python scripts/validate_memory.py --config configs/training_config_large_multidataset_v2.yaml
    python scripts/validate_memory.py --model ltg/norbert4-large --batch-size 8 --seq-length 512
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_current_memory_usage() -> Tuple[float, float, float]:
    """
    Get current system memory usage.
    
    Returns:
        Tuple of (used_gb, available_gb, total_gb)
    """
    if not HAS_PSUTIL:
        # Fallback to typical values if psutil not available
        return (16.0, 48.0, 64.0)
    
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    
    return (used_gb, available_gb, total_gb)


@dataclass
class ModelSpec:
    """Model specifications for memory estimation."""
    name: str
    num_parameters: int  # in millions
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int


# Known model specifications
KNOWN_MODELS = {
    "ltg/norbert4-base": ModelSpec(
        name="NorBERT4 Base",
        num_parameters=150,  # ~150M params
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    ),
    "ltg/norbert4-large": ModelSpec(
        name="NorBERT4 Large",
        num_parameters=360,  # ~360M params
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
    ),
    "ltg/norbert4-xlarge": ModelSpec(
        name="NorBERT4 XLarge",
        num_parameters=987,  # ~987M params
        hidden_size=1536,
        num_layers=32,
        num_attention_heads=24,
        intermediate_size=4096,
    ),
    "answerdotai/ModernBERT-base": ModelSpec(
        name="ModernBERT Base",
        num_parameters=150,
        hidden_size=768,
        num_layers=22,
        num_attention_heads=12,
        intermediate_size=1152,
    ),
    "answerdotai/ModernBERT-large": ModelSpec(
        name="ModernBERT Large",
        num_parameters=400,
        hidden_size=1024,
        num_layers=28,
        num_attention_heads=16,
        intermediate_size=1536,
    ),
}


@dataclass
class TrainingConfig:
    """Training configuration for memory estimation."""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 256
    precision: str = "bf16"  # fp32, fp16, bf16
    optimizer: str = "adamw"  # adamw, adam_8bit, sgd, adafactor
    gradient_checkpointing: bool = False


@dataclass  
class MemoryBreakdown:
    """Detailed memory breakdown in GB."""
    model_weights: float
    optimizer_states: float
    gradients: float
    activations: float
    total: float
    peak_estimate: float  # With overhead
    training_memory: float = 0.0  # Calibrated training memory (above baseline)
    baseline_memory: float = 0.0  # Current system baseline memory
    available_memory: float = 0.0  # Currently available system memory


# =============================================================================
# CALIBRATED MEMORY ESTIMATION
# =============================================================================
# These formulas are calibrated from actual measurements on Apple Silicon M4
# for Sentence-Transformers training with NorBERT4-large (360M params).
#
# Measurements used for calibration:
#   batch=8,  seq=256, ckpt=True:  ~6.5 GB training memory
#   batch=8,  seq=512, ckpt=False: ~12.4 GB training memory  
#   batch=16, seq=512, ckpt=False: ~21.8 GB training memory
#   batch=32, seq=512, ckpt=False: ~20 GB training memory (stabilized)
#   batch=32, seq=512, ckpt=True:  ~6.4 GB training memory
#
# Peak system memory = baseline (~15-18 GB) + training_memory
# =============================================================================


def estimate_training_memory_calibrated(
    spec: ModelSpec,
    config: TrainingConfig,
) -> float:
    """
    Calibrated estimate of training memory (above system baseline).
    
    Based on actual measurements from Sentence-Transformers training.
    This is the memory ADDED during training, not total system memory.
    """
    batch = config.batch_size
    seq = config.max_seq_length
    params_m = spec.num_parameters
    
    # Scale factor for model size (calibrated on 360M params)
    size_factor = params_m / 360.0
    
    if config.gradient_checkpointing:
        # With checkpointing: memory is nearly constant (activations are recomputed)
        # Base overhead: model + optimizer + gradients ≈ 6 GB for 360M model
        base_memory = 6.0 * size_factor
        # Small scaling with batch*seq for input tensors
        activation_memory = batch * seq * 0.00001 * size_factor
        training_memory = base_memory + activation_memory
    else:
        # Without checkpointing: activations scale with batch * seq
        # Base: ~3 GB (model loading overhead)
        # Scaling: ~0.002 GB per (batch * seq) unit
        base_memory = 3.0 * size_factor
        
        # Activation memory scales with batch * seq
        # But has diminishing returns at larger batches (memory optimizations)
        activation_coefficient = 0.0023 * size_factor
        
        # Apply diminishing returns for larger batches (observed in measurements)
        if batch > 16:
            efficiency_factor = 0.7  # Memory optimizations kick in
        else:
            efficiency_factor = 1.0
        
        activation_memory = batch * seq * activation_coefficient * efficiency_factor
        training_memory = base_memory + activation_memory
    
    return training_memory


def estimate_model_memory(num_params_millions: float, precision: str) -> float:
    """
    Estimate memory for model weights.
    
    - fp32: 4 bytes per parameter
    - fp16/bf16 mixed: 2 bytes for model + 4 bytes for fp32 copy = 6 bytes
    - Pure fp16/bf16: 2 bytes per parameter
    """
    num_params = num_params_millions * 1e6
    
    if precision == "fp32":
        bytes_per_param = 4
    elif precision in ("fp16", "bf16"):
        # Mixed precision training: fp16/bf16 model + fp32 master weights
        # But for sentence-transformers, typically just model weights
        bytes_per_param = 2
    else:
        bytes_per_param = 4
    
    bytes_total = num_params * bytes_per_param
    return bytes_total / (1024**3)  # Convert to GB


def estimate_optimizer_memory(num_params_millions: float, optimizer: str) -> float:
    """
    Estimate optimizer state memory.
    
    - AdamW: 8 bytes (momentum + variance, both fp32)
    - AdamW 8-bit: 2 bytes (quantized states)
    - SGD with momentum: 4 bytes
    - Adafactor: ~4 bytes (uses row/col statistics)
    
    Plus fp32 master copy: 4 bytes for mixed precision
    """
    num_params = num_params_millions * 1e6
    
    if optimizer == "adamw":
        # AdamW: momentum (4 bytes) + variance (4 bytes) + fp32 copy (4 bytes) = 12 bytes
        bytes_per_param = 12
    elif optimizer == "adam_8bit":
        # 8-bit Adam: momentum (1 byte) + variance (1 byte) + fp32 copy (4 bytes) = 6 bytes
        bytes_per_param = 6
    elif optimizer == "sgd":
        # SGD with momentum: momentum (4 bytes) + fp32 copy (4 bytes) = 8 bytes
        bytes_per_param = 8
    elif optimizer == "adafactor":
        # Adafactor: ~4 bytes + some extra
        bytes_per_param = 4.5
    else:
        bytes_per_param = 12  # Default to AdamW
    
    bytes_total = num_params * bytes_per_param
    return bytes_total / (1024**3)


def estimate_gradient_memory(num_params_millions: float, precision: str) -> float:
    """
    Estimate gradient memory.
    
    Gradients are typically stored in the same precision as activations:
    - fp32: 4 bytes
    - fp16/bf16: 2 bytes (but often accumulated in fp32 = 4 bytes)
    """
    num_params = num_params_millions * 1e6
    
    # Gradients often kept in fp32 for stability
    bytes_per_param = 4
    
    bytes_total = num_params * bytes_per_param
    return bytes_total / (1024**3)


def estimate_activation_memory(
    spec: ModelSpec,
    batch_size: int,
    seq_length: int,
    gradient_checkpointing: bool = False,
    tensor_parallel: int = 1,
) -> float:
    """
    Estimate activation memory using the formula from Megatron-LM paper:
    "Reducing Activation Recomputation in Large Transformer Models"
    
    Without checkpointing:
        memory = s * b * h * L * (10 + 24/t + 5*a*s/(h*t))
    
    With selective checkpointing:
        memory = s * b * h * L * (10 + 24/t)
    
    With full checkpointing:
        memory = 2 * s * b * h * L
    
    Where:
        s = sequence length
        b = batch size per GPU
        h = hidden size
        L = number of layers
        a = number of attention heads
        t = tensor parallel degree
    """
    s = seq_length
    b = batch_size
    h = spec.hidden_size
    L = spec.num_layers
    a = spec.num_attention_heads
    t = tensor_parallel
    
    if gradient_checkpointing:
        # Full recomputation - minimal memory
        # Only need to store inputs to checkpointed segments
        bytes_total = 2 * s * b * h * L
    else:
        # No checkpointing - full activation memory
        # Simplified formula: includes attention scores, intermediate activations, etc.
        attention_term = 5 * a * s / (h * t)
        bytes_total = s * b * h * L * (10 + 24/t + attention_term)
    
    # Convert to GB (activations stored in fp16/bf16 = 2 bytes)
    return bytes_total / (1024**3)


def estimate_total_memory(
    spec: ModelSpec,
    config: TrainingConfig,
) -> MemoryBreakdown:
    """Estimate total GPU memory required for training."""
    
    # Get current system memory baseline
    baseline_used, available, total_system = get_current_memory_usage()
    
    # 1. Model weights
    model_memory = estimate_model_memory(spec.num_parameters, config.precision)
    
    # 2. Optimizer states
    optimizer_memory = estimate_optimizer_memory(spec.num_parameters, config.optimizer)
    
    # 3. Gradients
    gradient_memory = estimate_gradient_memory(spec.num_parameters, config.precision)
    
    # 4. Activations (only for one micro-batch at a time)
    activation_memory = estimate_activation_memory(
        spec,
        config.batch_size,
        config.max_seq_length,
        config.gradient_checkpointing,
    )
    
    # Total (theoretical)
    total = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    # Peak estimate with ~20% overhead for temporary buffers, CUDA kernels, etc.
    peak_estimate = total * 1.2 + 1.5  # Add ~1.5GB for CUDA overhead
    
    # CALIBRATED training memory (based on actual measurements)
    training_memory = estimate_training_memory_calibrated(spec, config)
    
    # Use ACTUAL baseline instead of hardcoded value
    calibrated_peak = baseline_used + training_memory
    
    return MemoryBreakdown(
        model_weights=model_memory,
        optimizer_states=optimizer_memory,
        gradients=gradient_memory,
        activations=activation_memory,
        total=total,
        peak_estimate=calibrated_peak,  # Use calibrated estimate with actual baseline
        training_memory=training_memory,
        baseline_memory=baseline_used,
        available_memory=available,
    )


def print_memory_breakdown(
    spec: ModelSpec,
    config: TrainingConfig,
    breakdown: MemoryBreakdown,
    available_memory: float = 80.0,  # Default A100 80GB
):
    """Print detailed memory breakdown."""
    # Get total system memory
    _, _, total_system = get_current_memory_usage()
    
    print("=" * 80)
    print("MEMORY ESTIMATION FOR TRANSFORMER TRAINING")
    print("=" * 80)
    
    print(f"\n📦 MODEL: {spec.name}")
    print(f"   Parameters: {spec.num_parameters}M")
    print(f"   Hidden size: {spec.hidden_size}")
    print(f"   Layers: {spec.num_layers}")
    print(f"   Attention heads: {spec.num_attention_heads}")
    
    print("\n⚙️  TRAINING CONFIG:")
    print(f"   Batch size per device: {config.batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Max sequence length: {config.max_seq_length}")
    print(f"   Precision: {config.precision}")
    print(f"   Optimizer: {config.optimizer}")
    print(f"   Gradient checkpointing: {config.gradient_checkpointing}")
    
    print("\n💾 MEMORY ESTIMATE (calibrated from actual measurements):")
    print("-" * 60)
    print(f"   Training memory required:     {breakdown.training_memory:>6.1f} GB")
    print(f"   + Current system baseline:    {breakdown.baseline_memory:>6.1f} GB")
    print(f"   -----------------------------------------")
    print(f"   = Expected peak usage:        {breakdown.peak_estimate:>6.1f} GB")
    print(f"   Total system memory:          {total_system:>6.1f} GB")
    margin = total_system - breakdown.peak_estimate
    print(f"   Headroom:                     {margin:>6.1f} GB")
    print("-" * 60)
    
    # Simple verdict
    if margin > 10:
        print(f"\n   ✅ SAFE: {margin:.1f} GB headroom")
    elif margin > 5:
        print(f"\n   ⚠️  TIGHT: Only {margin:.1f} GB headroom - close other apps")
    else:
        print(f"\n   ❌ RISKY: Only {margin:.1f} GB headroom - may OOM")
    
    # GPU compatibility (for when training on NVIDIA later)
    print("\n🖥️  NVIDIA GPU COMPATIBILITY (training memory only):")
    gpu_options = [
        ("RTX 3090/4090", 24),
        ("A10G", 24),
        ("A40", 48),
        ("A100 40GB", 40),
        ("A100 80GB", 80),
        ("H100 80GB", 80),
    ]
    
    for gpu_name, memory in gpu_options:
        # For dedicated GPU, only training memory matters (no system baseline)
        fits = breakdown.training_memory < memory * 0.90  # 10% safety margin
        status = "✅" if fits else "❌"
        gpu_margin = memory - breakdown.training_memory
        print(f"   {gpu_name:<15} ({memory}GB): {status} (margin: {gpu_margin:+.1f} GB)")
    
    print("\n" + "=" * 80)

def load_config_from_yaml(config_path: str) -> tuple[Optional[ModelSpec], TrainingConfig]:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get model spec
    base_model = config.get("model", {}).get("base_model", "")
    spec = KNOWN_MODELS.get(base_model)
    
    # Get training config
    training = config.get("training", {})
    model_config = config.get("model", {})
    
    train_config = TrainingConfig(
        batch_size=training.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
        max_seq_length=model_config.get("max_seq_length", 256),
        precision="bf16" if training.get("bf16", False) else ("fp16" if training.get("fp16", False) else "fp32"),
        optimizer=training.get("optim", "adamw_torch").replace("_torch", "").replace("_hf", ""),
        gradient_checkpointing=training.get("gradient_checkpointing", False),
    )
    
    return spec, train_config


def find_optimal_batch_size(
    spec: ModelSpec,
    max_seq_length: int,
    target_memory: float,
    precision: str = "bf16",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> int:
    """Find the maximum batch size that fits in target memory."""
    
    for batch_size in [64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]:
        config = TrainingConfig(
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            max_seq_length=max_seq_length,
            precision=precision,
            optimizer=optimizer,
            gradient_checkpointing=gradient_checkpointing,
        )
        breakdown = estimate_total_memory(spec, config)
        if breakdown.peak_estimate < target_memory * 0.9:  # Leave 10% margin
            return batch_size
    
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for transformer training"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ltg/norbert4-large",
        help="Model name (from known models) or HuggingFace model ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Training precision",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adam_8bit", "sgd", "adafactor"],
        default="adamw",
        help="Optimizer type",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=56.0,
        help="Available GPU/unified memory in GB for recommendations (default: 56GB for M4 Pro Max)",
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help="Find optimal batch size for target GPU memory",
    )
    
    args = parser.parse_args()
    
    # Load from config file or command line args
    if args.config:
        spec, config = load_config_from_yaml(args.config)
        if spec is None:
            print(f"⚠️  Unknown model in config. Using command line --model argument.")
            spec = KNOWN_MODELS.get(args.model)
    else:
        spec = KNOWN_MODELS.get(args.model)
        config = TrainingConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.seq_length,
            precision=args.precision,
            optimizer=args.optimizer,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    
    if spec is None:
        print(f"❌ Unknown model: {args.model}")
        print(f"   Known models: {list(KNOWN_MODELS.keys())}")
        print("\n   For custom models, add them to KNOWN_MODELS in this script.")
        sys.exit(1)
    
    # Calculate memory
    breakdown = estimate_total_memory(spec, config)
    
    # Print results
    print_memory_breakdown(spec, config, breakdown, args.gpu_memory)
    
    # Find optimal configurations if requested
    if args.find_optimal:
        find_optimal_configurations(spec, config, args.gpu_memory)


def find_optimal_configurations(spec: ModelSpec, config: TrainingConfig, available_memory: float):
    """Find optimal batch size and sequence length combinations."""
    print("\n" + "=" * 80)
    print("🔍 FINDING OPTIMAL CONFIGURATIONS")
    print("=" * 80)
    
    seq_lengths = [128, 256, 384, 512, 768, 1024]
    batch_sizes = [4, 8, 12, 16, 24, 32, 48, 64]
    
    # Apply Apple Silicon overhead if memory looks like unified memory
    is_apple_silicon = available_memory in [18, 28, 56, 112]
    overhead = 1.1 if is_apple_silicon else 1.0
    
    print(f"\nTarget memory: {available_memory} GB {'(Apple Silicon)' if is_apple_silicon else '(NVIDIA GPU)'}")
    
    # Find best configs for different strategies
    results = []
    
    for seq_len in seq_lengths:
        for batch in batch_sizes:
            for checkpointing in [False, True]:
                # Try different grad_accum values
                for grad_accum in [1, 2, 4, 8, 16]:
                    effective_batch = batch * grad_accum
                    
                    test_config = TrainingConfig(
                        batch_size=batch,
                        gradient_accumulation_steps=grad_accum,
                        max_seq_length=seq_len,
                        precision=config.precision,
                        optimizer=config.optimizer,
                        gradient_checkpointing=checkpointing,
                    )
                    
                    breakdown = estimate_total_memory(spec, test_config)
                    peak = breakdown.peak_estimate * overhead
                    
                    if peak < available_memory * 0.95:
                        # Throughput score: prioritize larger batch (GPU efficiency) and longer sequences
                        throughput_score = batch * (seq_len ** 0.5)  # Favor batch over seq length
                        results.append({
                            'batch': batch,
                            'grad_accum': grad_accum,
                            'effective_batch': effective_batch,
                            'seq_len': seq_len,
                            'checkpointing': checkpointing,
                            'peak_gb': peak,
                            'throughput_score': throughput_score,
                        })
    
    if not results:
        print("❌ No valid configurations found for this memory limit!")
        return
    
    # Group by effective batch size targets
    print("\n📊 OPTIMAL CONFIGURATIONS BY EFFECTIVE BATCH SIZE:")
    print("-" * 90)
    print(f"{'Eff Batch':<12} {'Batch':<8} {'Grad Acc':<10} {'Seq Len':<10} {'Checkpoint':<12} {'Peak GB':<10}")
    print("-" * 90)
    
    for target_eff_batch in [32, 48, 64, 96, 128]:
        # Filter configs with this effective batch
        matching = [r for r in results if r['effective_batch'] == target_eff_batch]
        if not matching:
            continue
            
        # Find best by throughput (largest batch * seq_len)
        best = max(matching, key=lambda x: x['throughput_score'])
        
        checkpoint_str = "Yes" if best['checkpointing'] else "No"
        print(f"{best['effective_batch']:<12} {best['batch']:<8} {best['grad_accum']:<10} "
              f"{best['seq_len']:<10} {checkpoint_str:<12} {best['peak_gb']:<10.1f}")
    
    # Best overall recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDED CONFIGURATIONS:")
    print("=" * 80)
    
    # Filter for effective batch >= 32
    good_configs = [r for r in results if r['effective_batch'] >= 32 and not r['checkpointing']]
    
    if good_configs:
        # Best for speed (largest batch without checkpointing)
        best_speed = max(good_configs, key=lambda x: x['batch'])
        print(f"\n🚀 FASTEST TRAINING (largest batch):")
        print(f"   batch_size: {best_speed['batch']}")
        print(f"   gradient_accumulation_steps: {best_speed['grad_accum']}")
        print(f"   max_seq_length: {best_speed['seq_len']}")
        print(f"   gradient_checkpointing: {best_speed['checkpointing']}")
        print(f"   → Effective batch: {best_speed['effective_batch']}, Peak: {best_speed['peak_gb']:.1f} GB")
        
        # Best for context (longest sequence)
        best_context = max(good_configs, key=lambda x: x['seq_len'])
        print(f"\n📚 LONGEST CONTEXT (max sequence length):")
        print(f"   batch_size: {best_context['batch']}")
        print(f"   gradient_accumulation_steps: {best_context['grad_accum']}")
        print(f"   max_seq_length: {best_context['seq_len']}")
        print(f"   gradient_checkpointing: {best_context['checkpointing']}")
        print(f"   → Effective batch: {best_context['effective_batch']}, Peak: {best_context['peak_gb']:.1f} GB")
        
        # Balanced (effective batch 64, reasonable seq length)
        balanced = [r for r in good_configs if r['effective_batch'] == 64 and r['seq_len'] >= 256]
        if balanced:
            best_balanced = max(balanced, key=lambda x: x['seq_len'])
            print(f"\n⚖️  BALANCED (effective batch 64, good context):")
            print(f"   batch_size: {best_balanced['batch']}")
            print(f"   gradient_accumulation_steps: {best_balanced['grad_accum']}")
            print(f"   max_seq_length: {best_balanced['seq_len']}")
            print(f"   gradient_checkpointing: {best_balanced['checkpointing']}")
            print(f"   → Effective batch: {best_balanced['effective_batch']}, Peak: {best_balanced['peak_gb']:.1f} GB")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
