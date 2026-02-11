# MTEB Evaluation - Quick Guide

## Overview

This evaluation setup uses the official **Scandinavian Embedding Benchmark (SEB)**, which is now integrated into MTEB as `MTEB(Scandinavian, v1)`. Your results will be directly comparable to models on the [SEB leaderboard](https://kennethenevoldsen.com/scandinavian-embedding-benchmark/).

## Quick Start

### Method 1: Config-Only (Recommended)

1. **Set your model in the config:**
   ```yaml
   # In configs/evaluation_config.yaml
   model_path: "models/norbert4-base-multidataset-exp1/final"
   ```

2. **Run evaluation:**
   ```bash
   uv run python scripts/evaluate_mteb.py
   ```

That's it! The script uses the config automatically.

### Method 2: Command Line Override

Override the model from command line:
```bash
uv run python scripts/evaluate_mteb.py --model models/your-model
```

### Method 3: Multiple Configs

Create different configs for different models:
```bash
# Evaluate base model
uv run python scripts/evaluate_mteb.py --config configs/evaluation_config.yaml

# Evaluate large model
uv run python scripts/evaluate_mteb.py --config configs/evaluation_config_large.yaml
```

## What Gets Evaluated?

The script evaluates on **13 Norwegian tasks** from the Scandinavian benchmark:

- **Classification (6 tasks):** NoRecClassification, NorwegianParliamentClassification, NordicLangClassification, ScalaClassification, MassiveIntentClassification, MassiveScenarioClassification
- **Retrieval (2 tasks):** NorQuadRetrieval, SNLRetrieval
- **Clustering (4 tasks):** SNLHierarchicalClusteringS2S/P2P, VGHierarchicalClusteringS2S/P2P
- **BitextMining (1 task):** NorwegianCourtsBitextMining

## Configuration Options

### Model Selection
```yaml
# Simple path
model_path: "models/my-model"

# Or checkpoint
model_path: "models/my-model/checkpoint-1000"
```

### Benchmark Customization
```yaml
benchmark:
  # Use different benchmark
  name: "MTEB(Scandinavian, v1)"
  
  # Filter languages
  languages: [nob, nno]
  
  # Filter task types
  task_types: [Classification, Retrieval]  # Only these types
  
  # Exclude specific tasks
  exclude_tasks: [Tatoeba]
```

### Model Settings
```yaml
model:
  # Enable trust_remote_code
  trust_remote_code: true
  
  # Force specific device
  device: "cuda"  # or "cpu", "mps"
  
  # Model kwargs (e.g., quantization)
  model_kwargs:
    torch_dtype: "float16"
```

### Evaluation Settings
```yaml
evaluation:
  # Output location
  output_dir: "results/mteb"
  
  # Verbosity (0=minimal, 1=normal, 2=detailed)
  verbosity: 2
  
  # Batch size
  batch_size: 32
  
  # Display options
  show_detailed_results: true
  show_overall_score: true
```

## Understanding Results

### Overall Score
The script calculates an overall score averaging all tasks. This score is directly comparable to the SEB leaderboard.

Example output:
```
==================================================================
OVERALL SCORE (Average across all tasks)
==================================================================
  0.6314

  This score is comparable to the Scandinavian Embedding Benchmark
  leaderboard: https://kennethenevoldsen.com/scandinavian-embedding-benchmark/
==================================================================
```

### Results Location
Results are saved to:
```
results/mteb/
└── model-name/
    ├── NoRecClassification.json
    ├── NorQuadRetrieval.json
    └── ... (one file per task)
```

## Command Line Options

```bash
# Show help
uv run python scripts/evaluate_mteb.py --help

# Override model
uv run python scripts/evaluate_mteb.py --model models/my-model

# Use custom config
uv run python scripts/evaluate_mteb.py --config configs/custom.yaml

# Override output directory
uv run python scripts/evaluate_mteb.py --output-dir results/custom

# Change verbosity
uv run python scripts/evaluate_mteb.py --verbosity 0  # minimal output
```

## Example Workflows

### Evaluating Multiple Models
```bash
# Base model
uv run python scripts/evaluate_mteb.py --model models/norbert4-base-multidataset-exp1/final

# Large model
uv run python scripts/evaluate_mteb.py --model models/norbert4-large-multidataset-exp1/final

# Compare results in results/mteb/ directory
```

### Testing During Training
```bash
# Evaluate specific checkpoint
uv run python scripts/evaluate_mteb.py --model models/my-model/checkpoint-5000
```

### Custom Benchmark
```yaml
# In your custom config
benchmark:
  name: "MTEB(Scandinavian, v1)"
  task_types: [Retrieval, Classification]  # Only these
  exclude_tasks: [MassiveIntentClassification]  # Skip slow tasks
```

## Comparing to SEB Leaderboard

Your overall score can be directly compared to:
- **Official SEB leaderboard:** https://kennethenevoldsen.com/scandinavian-embedding-benchmark/
- **MTEB leaderboard:** https://huggingface.co/spaces/mteb/leaderboard (filter by Norwegian)

The benchmark uses the same 13 Norwegian tasks as the official leaderboard.

## Troubleshooting

### Missing Dependencies
```bash
uv add mteb sentence-transformers pyyaml
```

### Model Loading Issues
Ensure `trust_remote_code: true` is set if your model requires it.

### Memory Issues
Reduce batch size in config:
```yaml
evaluation:
  batch_size: 16  # or lower
```

### Cache Issues
Clear MTEB cache:
```bash
rm -rf ~/.cache/mteb
```

## Tips

1. **First run:** May take longer as datasets are downloaded
2. **Caching:** MTEB caches results - delete `results/mteb/model-name/` to re-evaluate
3. **Speed:** Evaluation takes 30-60 minutes depending on model size
4. **Comparison:** Keep all results in `results/mteb/` for easy comparison
