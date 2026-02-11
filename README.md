# Scandinavian Embedding Models

Training state-of-the-art embedding models for Norwegian, Danish, and Swedish using multi-dataset approach.

## 🎯 Overview

This project trains **embedding models** that convert Scandinavian text into numerical vectors for semantic search, retrieval, and question answering. The models are trained on a combination of:

- **NLI data** (Natural Language Inference): Teaching semantic relationships
- **QA data** (Question-Answer pairs): Teaching information retrieval  
- **Retrieval data** (DDSC): Teaching document similarity

### Key Achievement

**NorBERT4-base model trained with this approach achieves:**
- NorQuad ndcg@10: **0.232** (+11% improvement)
- SNL ndcg@10: **0.818** (+6.9% improvement)

**Deployed model**: [thivy/norbert4-base-scandinavian-embedding](https://huggingface.co/thivy/norbert4-base-scandinavian-embedding)

## 🏆 Why Multi-Dataset Training?

Traditional staged training (NLI → QA → Retrieval) suffers from catastrophic forgetting. This project uses **simultaneous multi-dataset training** with ROUND_ROBIN sampling:

✅ **Better performance**: All tasks improve together  
✅ **No forgetting**: Earlier datasets remain relevant  
✅ **Efficient**: One training run instead of three  
✅ **Balanced**: Equal representation from each dataset

## 🏗️ Project Structure

```
finetune-embedding-norwegian/
├── configs/
│   ├── training_config_base_multidataset_final.yaml    # ⭐ Best base model config
│   ├── training_config_large_multidataset.yaml         # Large model config
│   └── README.md                                       # Config documentation
├── utils/
│   ├── trainer_multidataset.py      # Multi-dataset trainer (ROUND_ROBIN)
│   ├── trainer_nli.py               # Single NLI trainer
│   ├── data_loader_nli.py           # NLI dataset loader
│   ├── data_loader_scandi_qa.py     # Multi-source QA datasets (NO+DA+SV)
│   ├── data_loader_ddsc.py          # DDSC retrieval data (all tasks)
│   └── data_loader_paws.py          # PAWS-X paraphrase data
├── scripts/
│   ├── evaluate_mteb.py             # MTEB evaluation
│   └── analyze_checkpoint_degradation.py
└── main.py                          # Legacy single-dataset trainer
```

## � Prerequisites

### Azure ML Compute Setup

This project can be trained on an Azure Machine Learning VM. To connect to your Azure ML VM:

**Connection Details (replace with your VM's values):**
```
Compute: azure-ml-vm
Username: azureuser
Public IP: <YOUR_VM_PUBLIC_IP>    # Get from Azure Portal
Private IP: 10.0.0.4
SSH Port: 50000                   # Default SSH port for Azure ML VMs
Private Key: ~/.ssh/est_test_key_pair.pem
```

**Example (your current VM):**
```
Public IP: 20.8.94.92
SSH Port: 50000
```

**Setup SSH Connection (Windows/Mac/Linux):**

1. **Place your SSH private key in the correct location:**
   ```bash
   # Windows
   C:\Users\<username>\.ssh\est_test_key_pair.pem
   
   # Mac/Linux
   ~/.ssh/est_test_key_pair.pem
   ```

2. **Fix SSH key permissions (required):**
   ```bash
   # macOS/Linux
   chmod 600 ~/.ssh/est_test_key_pair.pem
   
   # Windows (PowerShell as Admin)
   icacls "C:\Users\<username>\.ssh\est_test_key_pair.pem" /inheritance:r /grant:r "$($env:USERNAME):(F)"
   ```

3. **Connect to the VM:**
   ```bash
   # Replace 20.8.94.92 with your VM's public IP
   ssh -i ~/.ssh/est_test_key_pair.pem azureuser@<YOUR_VM_PUBLIC_IP> -p 50000
   
   # Example with our current VM:
   ssh -i ~/.ssh/est_test_key_pair.pem azureuser@20.8.94.92 -p 50000
   ```

4. **Configure SSH Config (optional, for convenience):**
   
   Add to `~/.ssh/config` (replace <YOUR_VM_PUBLIC_IP> with your VM's IP):
   ```
   Host azure-ml-vm
       HostName <YOUR_VM_PUBLIC_IP>
       User azureuser
       Port 50000
       IdentityFile ~/.ssh/est_test_key_pair.pem
       StrictHostKeyChecking no
   ```
   
   Example with our current VM:
   ```
   Host azure-ml-vm
       HostName 20.8.94.92
       User azureuser
       Port 50000
       IdentityFile ~/.ssh/est_test_key_pair.pem
       StrictHostKeyChecking no
   ```
   
   Then connect with:
   ```bash
   ssh azure-ml-vm
   ```

**Troubleshooting SSH Connection:**
- Verify the VM is running in Azure Portal
- Check that port 50000 is open in Network Security Group (NSG) inbound rules
- Ensure you're connected to the correct network/VPN if required
- Verify SSH key is in the correct location with proper permissions

## �🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Train Base Model

Train the proven NorBERT4-base model (640 dimensions, 256 tokens):

```bash
caffeinate -i uv run python -m utils.trainer_multidataset \
  configs/training_config_base_multidataset_final.yaml
```

**Training details:**
- **Datasets**: ~1.6M samples (Norwegian + Danish + Swedish)
  - NLI: 556k triplets
  - QA: 100k query-answer pairs  
  - DDSC: 949k retrieval pairs
- **Time**: ~18-24 hours on Apple Silicon M2 Pro
- **Memory**: ~25-30GB RAM
- **Output**: `models/norbert4-base-multidataset-exp1/`

### 3. Train Large Model

Train the NorBERT4-large model (1024 dimensions):

```bash
caffeinate -i uv run python -m utils.trainer_multidataset \
  configs/training_config_large_multidataset.yaml
```

**Requirements:**
- **64GB RAM** (uses gradient checkpointing)
- **Time**: ~24-30 hours on Apple Silicon
- **Memory**: ~35-45GB RAM

### 4. Evaluate with MTEB

```bash
uv run python scripts/evaluate_mteb.py \
  --model models/norbert4-base-multidataset-exp1
```
## 💡 Using Trained Models

### Load and Use

```python
from sentence_transformers import SentenceTransformer

# Load your trained model
model = SentenceTransformer("models/norbert4-base-multidataset-exp1")

# Encode Norwegian text
queries = ["Hva er hovedstaden i Norge?"]
documents = [
    "Oslo er Norges hovedstad og største by.",
    "Bergen er kjent for sine syv fjell.",
]

query_embedding = model.encode(queries)
doc_embeddings = model.encode(documents)

# Compute similarity
from sentence_transformers.util import cos_sim
similarities = cos_sim(query_embedding, doc_embeddings)
print(f"Similarities: {similarities}")
```

### Push to HuggingFace Hub

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/norbert4-base-multidataset-exp1")
model.push_to_hub("your-username/norbert4-scandinavian-embedding")
```

## 📊 Training Approach

### Multi-Dataset Strategy

The trainer uses **ROUND_ROBIN sampling** to combine three datasets:

```yaml
multi_dataset_batch_sampler: "ROUND_ROBIN"
```

**How it works:**
1. Each training step samples equally from all three datasets
2. Model learns all tasks simultaneously
3. No catastrophic forgetting
4. Better generalization

### Key Training Parameters

```yaml
training:
  per_device_train_batch_size: 16     # Batch size per device
  gradient_accumulation_steps: 2      # Effective batch = 16 * 2 = 32
  learning_rate: 5.0e-6               # Low LR prevents overfitting
  warmup_steps: 0                     # No warmup (avoids early peak)
  weight_decay: 0.015                 # Strong regularization
```

### Anti-Overfitting Strategy

1. **No warmup**: Prevents hitting optimal point too early
2. **Low learning rate**: Slow, steady convergence (5e-6 vs standard 2e-5)
3. **Strong weight decay**: 0.015 regularization
4. **Single epoch**: Through combined 1.6M samples
5. **Early stopping**: Monitors average loss across all datasets

### Memory Optimization

For large models on Apple Silicon:

```yaml
# Gradient checkpointing automatically enabled
# Reduces memory by 30-50% with ~20% slowdown
```

## 📈 Datasets Used

### 1. NLI (Natural Language Inference)
- **Source**: [Fremtind/all-nli-norwegian](https://huggingface.co/datasets/Fremtind/all-nli-norwegian)
- **Samples**: 556k triplets
- **Format**: (anchor, positive, negative)
- **Purpose**: Teaches semantic relationships and entailment

### 2. QA (Question-Answer)
- **Sources**: 
  - ltg/norquad (Norwegian QA)
  - ltg/norbookqa (Norwegian OpenBookQA)
  - alexandrainst/scandi-qa (Norwegian + Danish + Swedish)
  - Supervised Danish pairs
- **Samples**: ~100k pairs
- **Format**: (query, positive)
- **Purpose**: Teaches question-answer matching

### 3. DDSC Retrieval
- **Source**: [DDSC/nordic-embedding-training-data](https://huggingface.co/datasets/DDSC/nordic-embedding-training-data)
- **Samples**: 949k pairs (Norwegian + Danish + Swedish)
- **Format**: (query, positive, [negative])
- **Purpose**: Teaches document retrieval and similarity

**Total**: ~1.6 million training samples across three Scandinavian languages

## 🎯 Results

### Base Model Performance

**Model**: [thivy/norbert4-base-scandinavian-embedding](https://huggingface.co/thivy/norbert4-base-scandinavian-embedding)

| Task | Metric | Score | vs Previous |
|------|--------|-------|-------------|
| NorQuad | ndcg@10 | **0.232** | +11.0% |
| SNL | ndcg@10 | **0.818** | +6.9% |

**Configuration:**
- Base model: ltg/norbert4-base
- Embedding dim: 640
- Context length: 256 tokens
- Training: Multi-dataset ROUND_ROBIN

### Why It Works

✅ **Multi-dataset training** beats staged approach  
✅ **Low learning rate** (5e-6) prevents overfitting  
✅ **No warmup** avoids early performance peak  
✅ **Strong regularization** improves generalization  
✅ **ROUND_ROBIN sampling** balances all tasks

## ⚙️ Configuration

See `configs/README.md` for detailed configuration documentation.

### Key Parameters to Adjust

**Sequence length** (input token limit):
```yaml
model:
  max_seq_length: 256  # 256, 512, or 1024
```

**Batch size** (memory vs speed):
```yaml
training:
  per_device_train_batch_size: 16  # Reduce if OOM
  gradient_accumulation_steps: 2   # Maintain effective batch size
```

**Learning rate** (convergence speed):
```yaml
training:
  learning_rate: 5.0e-6  # Lower = more stable, slower
```

### Hardware Requirements

**Base model (640 dim, 256 seq):**
- RAM: 16GB minimum, 32GB recommended
- Time: 18-24 hours on M2 Pro

**Large model (1024 dim, 256 seq):**
- RAM: 32GB minimum, 64GB recommended
- Time: 24-30 hours on M2 Pro
- Gradient checkpointing: automatic

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Reduce batch size:**
```yaml
training:
  per_device_train_batch_size: 8  # Try 8, 4, or 2
  gradient_accumulation_steps: 4  # Maintain effective batch = 32
```

**For large models:**
- Gradient checkpointing is automatic
- Reduces memory by 30-50%
- Slightly slower (~20%)

### Training Crashes

**Use caffeinate** to prevent sleep on macOS:
```bash
caffeinate -i uv run python -m utils.trainer_multidataset configs/...
```

**Check logs:**
```bash
tail -f models/norbert4-base-multidataset-exp1/training.log
```

### Dataset Download Issues

**Manual test:**
```bash
python -c "from datasets import load_dataset; load_dataset('Fremtind/all-nli-norwegian')"
```

**Clear cache if corrupted:**
```bash
rm -rf ~/.cache/huggingface/datasets/
```

### Slow Training

**Optimize for speed:**
- Increase batch size if memory allows
- Reduce `max_seq_length` (512 → 256)
- Use base model instead of large
- Train on subset for testing first

## � Documentation

- **configs/README.md**: Configuration file documentation
- **docs/**: Detailed guides on training strategies, datasets, and approaches

### Key Documents

- `SCANDINAVIAN_DATASETS_COMPREHENSIVE.md`: All available Scandinavian datasets
- `SOTA_TRAINING_APPROACHES.md`: Analysis of state-of-the-art methods
- `TRAINING_SUMMARY.md`: Training insights and lessons learned
- `MODERNBERT_EXPLAINED.md`: Understanding the base architecture

## 🔬 Advanced Topics

### Monitoring Training

**MLflow tracking:**
```bash
mlflow ui
# Open http://localhost:5000
```

**TensorBoard:**
```bash
tensorboard --logdir models/norbert4-base-multidataset-exp1/logs
```

### Custom Datasets

Add your own dataset to the multi-dataset trainer:

1. Create a data loader in `utils/data_loader_*.py`
2. Update the trainer to include it
3. Adjust ROUND_ROBIN sampling proportions

### Evaluation

**MTEB benchmarks:**
```bash
uv run python scripts/evaluate_mteb.py --model <model_path>
```

**Checkpoint analysis:**
```bash
uv run python scripts/analyze_checkpoint_degradation.py
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional Scandinavian datasets
- Memory optimization techniques
- Evaluation on domain-specific tasks
- Support for other Nordic languages (Icelandic, Faroese)

## 📄 License

MIT License - see LICENSE file for details.

## � Acknowledgments

- **Language Technology Group (LTG)** at University of Oslo for NorBERT models
- **Fremtind** for Norwegian NLI dataset
- **DDSC** for Nordic embedding training data
- **Alexandra Institute** for Scandinavian QA data