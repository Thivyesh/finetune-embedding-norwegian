#!/usr/bin/env python3
"""
Pre-Training Validation Script

This script combines theoretical memory estimation with actual measurement
to find and validate the optimal training configuration before starting a long run.

Workflow:
1. Calculate theoretical memory estimates for various configurations
2. Select top candidates that should fit
3. Measure actual memory usage for the best candidates
4. Recommend the optimal configuration

Usage:
    python scripts/pretrain_validate.py --config configs/training_config_large_multidataset_v2.yaml
    python scripts/pretrain_validate.py --model ltg/norbert4-large --target-memory 56
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_memory import (
    KNOWN_MODELS, 
    ModelSpec, 
    TrainingConfig, 
    estimate_total_memory,
    MemoryBreakdown,
)


@dataclass
class ConfigCandidate:
    """A candidate training configuration."""
    batch_size: int
    grad_accum: int
    seq_length: int
    gradient_checkpointing: bool
    estimated_gb: float
    measured_gb: Optional[float] = None
    throughput_score: float = 0.0
    
    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum
    
    def __str__(self):
        ckpt = "✓" if self.gradient_checkpointing else "✗"
        measured = f"{self.measured_gb:.1f}" if self.measured_gb else "N/A"
        return (f"batch={self.batch_size:>2}, grad_accum={self.grad_accum:>2}, "
                f"seq={self.seq_length:>4}, ckpt={ckpt}, "
                f"est={self.estimated_gb:.1f}GB, measured={measured}GB")


def generate_candidates(
    spec: ModelSpec,
    target_memory: float,
    precision: str = "bf16",
    optimizer: str = "adamw",
) -> list[ConfigCandidate]:
    """Generate candidate configurations that might fit in target memory."""
    
    candidates = []
    
    # Configuration options to explore
    batch_sizes = [4, 8, 12, 16, 24, 32]
    seq_lengths = [256, 384, 512, 768]
    grad_accums = [1, 2, 4, 8, 16]
    
    # Add overhead for Apple Silicon (MPS backend)
    apple_overhead = 1.1
    
    for seq_len in seq_lengths:
        for batch in batch_sizes:
            for grad_accum in grad_accums:
                for checkpointing in [False, True]:
                    effective_batch = batch * grad_accum
                    
                    # Skip if effective batch is too small or too large
                    if effective_batch < 32 or effective_batch > 128:
                        continue
                    
                    config = TrainingConfig(
                        batch_size=batch,
                        gradient_accumulation_steps=grad_accum,
                        max_seq_length=seq_len,
                        precision=precision,
                        optimizer=optimizer,
                        gradient_checkpointing=checkpointing,
                    )
                    
                    breakdown = estimate_total_memory(spec, config)
                    estimated = breakdown.peak_estimate * apple_overhead
                    
                    # Only include if it might fit (with some margin)
                    if estimated < target_memory * 1.1:
                        # Throughput score: prioritize larger batch (GPU efficiency)
                        # and reasonable sequence length
                        throughput_score = batch * (seq_len ** 0.3)
                        
                        candidates.append(ConfigCandidate(
                            batch_size=batch,
                            grad_accum=grad_accum,
                            seq_length=seq_len,
                            gradient_checkpointing=checkpointing,
                            estimated_gb=estimated,
                            throughput_score=throughput_score,
                        ))
    
    return candidates


def select_top_candidates(
    candidates: list[ConfigCandidate],
    target_memory: float,
    num_candidates: int = 5,
) -> list[ConfigCandidate]:
    """Select the most promising candidates to measure."""
    
    # Filter to those that should fit
    fitting = [c for c in candidates if c.estimated_gb < target_memory * 0.95]
    
    if not fitting:
        # If nothing fits comfortably, take the smallest ones
        fitting = sorted(candidates, key=lambda c: c.estimated_gb)[:num_candidates]
        return fitting
    
    # Sort by throughput score (higher is better)
    fitting.sort(key=lambda c: c.throughput_score, reverse=True)
    
    # Select diverse candidates:
    # - Best throughput without checkpointing
    # - Best throughput with checkpointing  
    # - Best for long context
    # - Balanced option
    selected = []
    
    # Best throughput without checkpointing
    no_ckpt = [c for c in fitting if not c.gradient_checkpointing]
    if no_ckpt:
        selected.append(no_ckpt[0])
    
    # Best throughput with checkpointing
    with_ckpt = [c for c in fitting if c.gradient_checkpointing]
    if with_ckpt:
        selected.append(with_ckpt[0])
    
    # Best for long context (highest seq_length that fits)
    long_ctx = sorted(fitting, key=lambda c: (c.seq_length, c.batch_size), reverse=True)
    if long_ctx and long_ctx[0] not in selected:
        selected.append(long_ctx[0])
    
    # Balanced (effective batch 64, reasonable seq length)
    balanced = [c for c in fitting if c.effective_batch == 64 and c.seq_length >= 384]
    if balanced:
        best_balanced = max(balanced, key=lambda c: c.seq_length)
        if best_balanced not in selected:
            selected.append(best_balanced)
    
    # Fill remaining slots with highest throughput
    for c in fitting:
        if len(selected) >= num_candidates:
            break
        if c not in selected:
            selected.append(c)
    
    return selected[:num_candidates]


def measure_candidate(
    candidate: ConfigCandidate,
    model_name: str,
    num_steps: int = 5,
) -> float:
    """Measure actual memory usage for a candidate configuration."""
    
    from scripts.measure_memory import measure_training_memory
    
    measurements = measure_training_memory(
        model_name=model_name,
        batch_size=candidate.batch_size,
        seq_length=candidate.seq_length,
        num_steps=num_steps,
        gradient_checkpointing=candidate.gradient_checkpointing,
    )
    
    return measurements['summary']['peak']


def print_comparison_table(candidates: list[ConfigCandidate], target_memory: float):
    """Print a comparison table of candidates."""
    
    print("\n" + "=" * 100)
    print("CONFIGURATION COMPARISON")
    print("=" * 100)
    
    header = (f"{'Batch':<6} {'Acc':<4} {'EffBatch':<9} {'SeqLen':<7} {'Ckpt':<5} "
              f"{'Est.GB':<8} {'Meas.GB':<9} {'Status':<12}")
    print(header)
    print("-" * 100)
    
    for c in candidates:
        ckpt = "Yes" if c.gradient_checkpointing else "No"
        measured = f"{c.measured_gb:.1f}" if c.measured_gb else "---"
        
        if c.measured_gb:
            if c.measured_gb < target_memory * 0.85:
                status = "✅ SAFE"
            elif c.measured_gb < target_memory * 0.95:
                status = "⚠️ TIGHT"
            else:
                status = "❌ RISKY"
        else:
            if c.estimated_gb < target_memory * 0.85:
                status = "📊 Est.SAFE"
            elif c.estimated_gb < target_memory * 0.95:
                status = "📊 Est.TIGHT"
            else:
                status = "📊 Est.RISKY"
        
        row = (f"{c.batch_size:<6} {c.grad_accum:<4} {c.effective_batch:<9} "
               f"{c.seq_length:<7} {ckpt:<5} {c.estimated_gb:<8.1f} {measured:<9} {status}")
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-training validation: find and verify optimal configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training config YAML (to extract model info)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ltg/norbert4-large",
        help="Model name",
    )
    parser.add_argument(
        "--target-memory",
        type=float,
        default=56.0,
        help="Target available memory in GB (default: 56GB for M4 Pro Max)",
    )
    parser.add_argument(
        "--measure-top",
        type=int,
        default=3,
        help="Number of top candidates to measure (default: 3)",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=5,
        help="Number of steps for each measurement (default: 5)",
    )
    parser.add_argument(
        "--skip-measure",
        action="store_true",
        help="Skip actual measurement, only show theoretical estimates",
    )
    
    args = parser.parse_args()
    
    # Load model spec
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        model_name = config.get("model", {}).get("base_model", args.model)
    else:
        model_name = args.model
    
    spec = KNOWN_MODELS.get(model_name)
    if spec is None:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Known models: {list(KNOWN_MODELS.keys())}")
        sys.exit(1)
    
    print("=" * 100)
    print("PRE-TRAINING VALIDATION")
    print("=" * 100)
    print(f"\n📦 Model: {spec.name} ({spec.num_parameters}M params)")
    print(f"🎯 Target memory: {args.target_memory} GB")
    
    # Step 1: Generate candidates
    print("\n" + "=" * 100)
    print("STEP 1: Generating candidate configurations...")
    print("=" * 100)
    
    candidates = generate_candidates(spec, args.target_memory)
    print(f"   Generated {len(candidates)} potential configurations")
    
    # Step 2: Select top candidates
    print("\n" + "=" * 100)
    print("STEP 2: Selecting top candidates to validate...")
    print("=" * 100)
    
    top_candidates = select_top_candidates(candidates, args.target_memory, args.measure_top + 2)
    print(f"   Selected {len(top_candidates)} candidates for validation")
    
    # Print initial estimates
    print_comparison_table(top_candidates, args.target_memory)
    
    # Step 3: Measure actual memory
    if not args.skip_measure:
        print("\n" + "=" * 100)
        print(f"STEP 3: Measuring actual memory for top {args.measure_top} candidates...")
        print("=" * 100)
        
        # Sort by estimated (smallest first for safety)
        to_measure = sorted(top_candidates, key=lambda c: c.estimated_gb)[:args.measure_top]
        
        for i, candidate in enumerate(to_measure):
            print(f"\n📏 Measuring candidate {i+1}/{len(to_measure)}:")
            print(f"   {candidate}")
            
            try:
                peak = measure_candidate(
                    candidate, 
                    model_name, 
                    args.measure_steps,
                )
                candidate.measured_gb = peak
                print(f"   ✅ Measured: {peak:.1f} GB")
            except Exception as e:
                print(f"   ❌ Measurement failed: {e}")
                candidate.measured_gb = None
        
        # Print updated comparison
        print_comparison_table(top_candidates, args.target_memory)
    
    # Step 4: Recommend best configuration
    print("\n" + "=" * 100)
    print("STEP 4: RECOMMENDATIONS")
    print("=" * 100)
    
    # Find best measured candidate
    measured = [c for c in top_candidates if c.measured_gb is not None]
    
    if measured:
        # Best that fits safely
        safe = [c for c in measured if c.measured_gb < args.target_memory * 0.85]
        if safe:
            best = max(safe, key=lambda c: c.throughput_score)
            margin = args.target_memory - best.measured_gb
            
            print(f"\n🏆 RECOMMENDED CONFIGURATION:")
            print(f"   per_device_train_batch_size: {best.batch_size}")
            print(f"   gradient_accumulation_steps: {best.grad_accum}")
            print(f"   max_seq_length: {best.seq_length}")
            print(f"   gradient_checkpointing: {best.gradient_checkpointing}")
            print(f"\n   → Effective batch size: {best.effective_batch}")
            print(f"   → Measured memory: {best.measured_gb:.1f} GB")
            print(f"   → Safety margin: {margin:.1f} GB")
        else:
            # Nothing fits safely - recommend with checkpointing
            smallest = min(measured, key=lambda c: c.measured_gb)
            print(f"\n⚠️  No configuration fits safely within {args.target_memory}GB")
            print(f"\n   Smallest measured: {smallest}")
            print(f"   Consider enabling gradient_checkpointing or reducing batch_size")
    else:
        # Use theoretical estimates
        safe = [c for c in top_candidates if c.estimated_gb < args.target_memory * 0.80]
        if safe:
            best = max(safe, key=lambda c: c.throughput_score)
            print(f"\n📊 RECOMMENDED (based on estimates):")
            print(f"   per_device_train_batch_size: {best.batch_size}")
            print(f"   gradient_accumulation_steps: {best.grad_accum}")
            print(f"   max_seq_length: {best.seq_length}")
            print(f"   gradient_checkpointing: {best.gradient_checkpointing}")
            print(f"\n   → Estimated memory: {best.estimated_gb:.1f} GB")
            print(f"\n   ⚠️  Run with --measure-top to verify before long training!")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
