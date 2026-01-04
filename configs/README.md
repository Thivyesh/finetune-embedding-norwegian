# Training Configurations

## Active Configurations

### `training_config_base_multidataset_final.yaml`
**Best performing base model configuration** (NorBERT4-base, 640 dim)
- Model: `ltg/norbert4-base`
- Max sequence length: 256 tokens
- Multi-dataset training with ROUND_ROBIN sampling
- Combines NLI (556k) + QA (100k) + DDSC retrieval (949k)
- Total: ~1.6M samples (Norwegian + Danish + Swedish)
- **Results**: NorQuad ndcg@10=0.232 (+11%), SNL ndcg@10=0.818 (+6.9%)
- **Deployed**: https://huggingface.co/thivy/norbert4-base-scandinavian-embedding

### `training_config_large_multidataset.yaml`
**Large model configuration** (NorBERT4-large, 1024 dim)
- Model: `ltg/norbert4-large`
- Max sequence length: 256 tokens (configurable to 512/1024)
- Same multi-dataset approach as base model
- Gradient checkpointing enabled for memory efficiency
- Target: https://huggingface.co/thivy/norbert4-large-scandinavian-embedding

## Training Approach

Both configurations use the **multi-dataset training approach** which outperforms staged training:
1. **ROUND_ROBIN sampling**: Equal representation from all three datasets
2. **No warmup**: Prevents early overfitting peak
3. **Low learning rate**: 5.0e-6 (75% reduction from standard)
4. **Strong regularization**: weight_decay=0.015, gradient clipping
5. **Single epoch**: Through combined ~1.6M samples

## Usage

**Train base model:**
```bash
caffeinate -i uv run python -m utils.trainer_multidataset \
  configs/training_config_base_multidataset_final.yaml
```

**Train large model:**
```bash
caffeinate -i uv run python -m utils.trainer_multidataset \
  configs/training_config_large_multidataset.yaml
```

## Archive

The `archive/` directory contains old staged training configurations that were superseded by the multi-dataset approach. These are kept for reference only.
