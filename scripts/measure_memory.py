#!/usr/bin/env python3
"""
Measure actual memory usage during Sentence-Transformers training.

This script runs a short training test and measures real memory consumption
on Apple Silicon (MPS) or NVIDIA GPU.

Usage:
    python scripts/measure_memory.py --config configs/training_config_large_multidataset_v2.yaml
    python scripts/measure_memory.py --model ltg/norbert4-large --batch-size 8 --seq-length 512
"""

import argparse
import gc
import os
import sys
import time
import platform
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage():
    """Get current memory usage in GB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def get_system_memory():
    """Get system memory info."""
    import psutil
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),
        'available': mem.available / (1024**3),
        'used': mem.used / (1024**3),
        'percent': mem.percent,
    }


def get_mps_memory():
    """Get MPS (Apple Silicon) memory usage if available."""
    try:
        import torch
        if torch.backends.mps.is_available():
            # MPS doesn't have direct memory query like CUDA
            # We'll use system memory as proxy for unified memory
            return get_system_memory()
    except:
        pass
    return None


def get_cuda_memory():
    """Get CUDA GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / (1024**3),
                'reserved': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**3),
            }
    except:
        pass
    return None


def clear_memory():
    """Clear memory caches."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except:
        pass


def measure_training_memory(
    model_name: str,
    batch_size: int,
    seq_length: int,
    num_steps: int = 10,
    gradient_checkpointing: bool = False,
):
    """
    Measure actual memory usage during training.
    
    Returns dict with memory measurements at each stage.
    """
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader
    from datasets import Dataset
    import numpy as np
    
    measurements = {}
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        print(f"💻 Using CPU")
    
    print(f"\n📊 Measuring memory for:")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Gradient checkpointing: {gradient_checkpointing}")
    print(f"   Test steps: {num_steps}")
    
    # Baseline memory
    clear_memory()
    time.sleep(1)
    baseline_sys = get_system_memory()
    measurements['baseline'] = {
        'system_used': baseline_sys['used'],
        'system_available': baseline_sys['available'],
    }
    print(f"\n📍 Baseline: {baseline_sys['used']:.1f} GB system memory used")
    
    # Load model
    print(f"\n⏳ Loading model...")
    model = SentenceTransformer(
        model_name, 
        device=device,
        trust_remote_code=True,  # Required for custom models like NorBERT
    )
    model.max_seq_length = seq_length
    
    if gradient_checkpointing:
        # Enable gradient checkpointing on the underlying transformer
        if hasattr(model._first_module(), 'auto_model'):
            model._first_module().auto_model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled")
    
    time.sleep(1)
    after_model_sys = get_system_memory()
    cuda_mem = get_cuda_memory()
    
    measurements['after_model_load'] = {
        'system_used': after_model_sys['used'],
        'model_memory': after_model_sys['used'] - baseline_sys['used'],
    }
    if cuda_mem:
        measurements['after_model_load']['cuda_allocated'] = cuda_mem['allocated']
    
    print(f"📍 After model load: {after_model_sys['used']:.1f} GB (+{measurements['after_model_load']['model_memory']:.1f} GB)")
    
    # Create dummy training data (triplets for NLI-style training)
    print(f"\n⏳ Creating dummy training data...")
    
    # Generate random text-like data
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
             "a", "is", "was", "are", "were", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "will", "would",
             "could", "should", "may", "might", "must", "shall"]
    
    def random_sentence(length=20):
        return " ".join(np.random.choice(vocab, size=length))
    
    # Create triplet data (anchor, positive, negative)
    num_samples = batch_size * num_steps * 2  # Extra samples for safety
    data = {
        'anchor': [random_sentence() for _ in range(num_samples)],
        'positive': [random_sentence() for _ in range(num_samples)],
        'negative': [random_sentence() for _ in range(num_samples)],
    }
    dataset = Dataset.from_dict(data)
    
    # Create loss function (TripletLoss like NLI training)
    train_loss = losses.TripletLoss(model=model)
    
    # Create training dataset in the format expected by ST trainer
    train_dataset = Dataset.from_dict({
        'anchor': data['anchor'],
        'positive': data['positive'], 
        'negative': data['negative'],
    })
    
    after_data_sys = get_system_memory()
    measurements['after_data_load'] = {
        'system_used': after_data_sys['used'],
    }
    print(f"📍 After data load: {after_data_sys['used']:.1f} GB")
    
    # Training using SentenceTransformerTrainer (same as actual training)
    print(f"\n⏳ Running {num_steps} training steps with SentenceTransformerTrainer...")
    
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import TrainerCallback
    import tempfile
    
    # Custom callback to measure memory at each step
    class MemoryMonitorCallback(TrainerCallback):
        def __init__(self):
            self.memories = []
            self.peak = 0
        
        def on_step_end(self, args, state, control, **kwargs):
            mem = get_system_memory()['used']
            self.memories.append(mem)
            self.peak = max(self.peak, mem)
            print(f"   Step {state.global_step}: {mem:.1f} GB")
    
    mem_monitor = MemoryMonitorCallback()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=1,
            max_steps=num_steps,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            dataloader_drop_last=True,
            report_to="none",
        )
        
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
            callbacks=[mem_monitor],
        )
        
        # Run training
        print("   Starting training...")
        before_train = get_system_memory()['used']
        
        trainer.train()
        
        # Get peak memory
        after_train = get_system_memory()['used']
        peak_memory = max(before_train, after_train, mem_monitor.peak)
        step_memories = mem_monitor.memories if mem_monitor.memories else [after_train]
    
    # Manual memory measurement after training
    time.sleep(2)
    final_sys = get_system_memory()
    cuda_mem = get_cuda_memory()
    
    measurements['during_training'] = {
        'peak_system': peak_memory,
        'avg_system': sum(step_memories) / len(step_memories) if step_memories else 0,
        'min_system': min(step_memories) if step_memories else 0,
        'max_system': max(step_memories) if step_memories else 0,
    }
    if cuda_mem:
        measurements['during_training']['cuda_max_allocated'] = cuda_mem['max_allocated']
    
    # Calculate training memory usage
    training_memory = peak_memory - baseline_sys['used']
    
    measurements['summary'] = {
        'baseline': baseline_sys['used'],
        'peak': peak_memory,
        'training_memory': training_memory,
    }
    
    # Cleanup
    del model, trainer, train_dataset
    clear_memory()
    time.sleep(1)
    
    after_cleanup_sys = get_system_memory()
    measurements['after_cleanup'] = {
        'system_used': after_cleanup_sys['used'],
    }
    
    return measurements


def main():
    parser = argparse.ArgumentParser(description="Measure actual training memory usage")
    parser.add_argument("--config", type=str, help="Path to training config YAML")
    parser.add_argument("--model", type=str, default="ltg/norbert4-large", help="Model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        model_name = config.get("model", {}).get("base_model", args.model)
        batch_size = config.get("training", {}).get("per_device_train_batch_size", args.batch_size)
        seq_length = config.get("model", {}).get("max_seq_length", args.seq_length)
        gradient_checkpointing = config.get("training", {}).get("gradient_checkpointing", args.gradient_checkpointing)
    else:
        model_name = args.model
        batch_size = args.batch_size
        seq_length = args.seq_length
        gradient_checkpointing = args.gradient_checkpointing
    
    print("=" * 70)
    print("MEMORY MEASUREMENT TEST")
    print("=" * 70)
    print(f"\n🖥️  System: {platform.system()} {platform.machine()}")
    
    sys_mem = get_system_memory()
    print(f"📊 Total RAM: {sys_mem['total']:.1f} GB")
    print(f"📊 Available: {sys_mem['available']:.1f} GB")
    print(f"📊 Used: {sys_mem['used']:.1f} GB ({sys_mem['percent']:.1f}%)")
    
    # Run measurement
    measurements = measure_training_memory(
        model_name=model_name,
        batch_size=batch_size,
        seq_length=seq_length,
        num_steps=args.steps,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 MEMORY MEASUREMENT RESULTS")
    print("=" * 70)
    
    summary = measurements['summary']
    print(f"\n🔹 Baseline system memory:    {summary['baseline']:.1f} GB")
    print(f"🔹 Peak during training:      {summary['peak']:.1f} GB")
    print(f"🔹 Training memory usage:     {summary['training_memory']:.1f} GB")
    
    training = measurements['during_training']
    print(f"\n📈 Training memory range: {training['min_system']:.1f} - {training['max_system']:.1f} GB")
    print(f"📈 Average during training: {training['avg_system']:.1f} GB")
    
    # Compare with available memory
    available = sys_mem['total'] - summary['baseline']
    margin = available - summary['training_memory']
    
    print(f"\n💾 Available for training: {available:.1f} GB")
    print(f"💾 Safety margin: {margin:.1f} GB")
    
    if margin > 10:
        print(f"\n✅ SAFE: Good margin for this configuration")
        print(f"   You could potentially increase batch_size or seq_length")
    elif margin > 5:
        print(f"\n⚠️  TIGHT: Limited margin, be careful with increases")
    else:
        print(f"\n❌ RISKY: Very little margin, consider reducing batch_size")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
