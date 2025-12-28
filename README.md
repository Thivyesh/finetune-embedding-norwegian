# Norwegian Embedding Model Fine-Tuning

Fine-tune Norwegian sentence embedding models using triplet data with the sentence-transformers library.

## 📚 What is This Project?

This project helps you **fine-tune embedding models** for Norwegian text. An embedding model converts text into numerical vectors (embeddings) where semantically similar texts have similar vectors. This is essential for:

- **Semantic search**: Find documents by meaning, not just keywords
- **Clustering**: Group similar texts together
- **Question answering**: Match questions to relevant answers
- **Recommendation systems**: Find similar content

### What is Triplet Training?

Triplet training uses three types of sentences:
- **Anchor**: A reference sentence
- **Positive**: A sentence similar to the anchor (should be close in vector space)
- **Negative**: A sentence different from the anchor (should be far in vector space)

Example triplet:
```
Anchor:   "Hundene leker i snøen."
Positive: "Tre hunder leker med et leketøy i snøen."
Negative: "Det er veldig varmt."
```

The model learns to make the anchor and positive embeddings closer together, while pushing the negative farther away.

## 🏗️ Project Structure

```
finetune-embedding-norwegian/
├── configs/
│   └── training_config.yaml    # All training parameters (EDIT THIS!)
├── utils/
│   ├── read_config.py          # Loads YAML config
│   ├── data_loader.py          # Loads Norwegian NLI dataset
│   └── trainer.py              # Training pipeline
├── main.py                     # Main training script
├── models/                     # Saved models will go here
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Understand the Config File

Open [`configs/training_config.yaml`](configs/training_config.yaml) and read through it. Every parameter is documented with comments explaining:
- What it does
- Why it matters
- What values to use

**Key parameters to understand:**

```yaml
model:
  name: "ltg/norbert3-large"  # The base Norwegian model to fine-tune
  max_seq_length: 128          # Maximum length of sentences (in tokens)

dataset:
  name: "Fremtind/all-nli-norwegian"  # Norwegian NLI triplet dataset (569K samples)

training:
  num_train_epochs: 1          # How many times to go through the dataset
  per_device_train_batch_size: 16  # Batch size (larger = faster but more memory)
  learning_rate: 2.0e-5        # How fast to learn (2e-5 is safe default)
```

### 3. Run a Quick Test First

Before training on the full dataset, test that everything works:

```bash
python main.py --quick-test
```

This runs training on just 1,000 samples to verify:
- ✓ Config is valid
- ✓ Dataset downloads correctly
- ✓ Model loads successfully
- ✓ Training loop works
- ✓ Checkpoints are saved

**This should take 2-5 minutes** depending on your hardware.

### 4. Run Full Training

Once the quick test works, run full training:

```bash
python main.py
```

**Expected duration:**
- **CPU**: 10-20 hours for 1 epoch (569K samples)
- **GPU (CUDA)**: 1-3 hours for 1 epoch
- **Apple Silicon (MPS)**: 3-6 hours for 1 epoch

### 5. Use Your Trained Model

After training completes, use your model like this:

```python
from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
model = SentenceTransformer("models/norbert3-large-nli-norwegian")

# Encode Norwegian sentences
sentences = [
    "Hva er hovedstaden i Norge?",
    "Oslo er Norges hovedstad.",
    "Jeg liker å spise pizza."
]

embeddings = model.encode(sentences)

# Compute similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity between Q&A: {similarity[0][0]:.3f}")  # Should be high!

similarity = cos_sim(embeddings[0], embeddings[2])
print(f"Similarity between Q and unrelated: {similarity[0][0]:.3f}")  # Should be low
```

## 🎓 Understanding the Training Process

### What Happens During Training?

1. **Load base model**: We start with a pre-trained Norwegian BERT model (`ltg/norbert3-large`)
2. **Prepare data**: Load triplets from the Fremtind Norwegian NLI dataset
3. **Training loop**: For each batch of triplets:
   - Model encodes anchor, positive, and negative into embeddings
   - Loss function calculates how to improve:
     - Pull anchor and positive closer
     - Push anchor and negative farther apart
   - Backpropagation updates model weights
4. **Evaluation**: Periodically check accuracy on validation set
5. **Save checkpoints**: Best models are saved automatically

### Key Concepts

**Loss Function: MultipleNegativesRankingLoss**
- Most efficient loss for triplet training
- Uses other samples in the batch as additional negatives
- Larger batch sizes → better training (but more memory)

**Evaluation Metric: Triplet Accuracy**
- For each triplet, check if: `distance(anchor, positive) < distance(anchor, negative)`
- Accuracy = % of triplets where this is true
- Goal: Get accuracy as high as possible (80-95% is good)

**Learning Rate Warmup**
- Learning rate starts low and gradually increases
- Prevents unstable training at the beginning
- `warmup_ratio: 0.1` = 10% of training uses warmup

## ⚙️ Configuration Guide

### Adjusting Training Speed vs Quality

**Faster training (lower quality):**
```yaml
training:
  per_device_train_batch_size: 32  # Larger batches
  num_train_epochs: 1              # Fewer epochs

dataset:
  max_train_samples: 50000         # Use subset of data
```

**Better quality (slower training):**
```yaml
training:
  per_device_train_batch_size: 16  # Smaller batches (more updates)
  num_train_epochs: 3              # More epochs
  learning_rate: 1.0e-5            # Lower learning rate (more careful)

dataset:
  max_train_samples: null          # Use all data
```

### Memory Issues?

If you run out of GPU/RAM memory:

```yaml
training:
  per_device_train_batch_size: 8   # Reduce batch size
  gradient_accumulation_steps: 2   # Simulate larger batches
  # Effective batch size = 8 * 2 = 16
```

### Different Base Models

Try other Norwegian models:

```yaml
model:
  name: "ltg/norbert3-base"        # Smaller, faster (110M params)
  # name: "ltg/norbert3-large"     # Larger, better quality (340M params)
  # name: "NbAiLab/nb-bert-base"   # Alternative Norwegian BERT
```

## 📊 Monitoring Training

### Training Logs

Watch the console output for:
- **Loss**: Should decrease over time
- **Eval accuracy**: Should increase over time
- **Learning rate**: Should warm up then gradually decrease

### TensorBoard (Optional)

View detailed training metrics:

```bash
# Install tensorboard
pip install tensorboard

# Launch tensorboard
tensorboard --logdir models/norbert3-large-nli-norwegian/logs

# Open browser to http://localhost:6006
```

## 🧪 Testing and Validation

### Test the Data Loader

```bash
python utils/data_loader.py
```

This will:
- Show dataset structure
- Display example triplets
- Verify columns are correct

### Test the Config Loader

```bash
python utils/read_config.py
```

This will:
- Load your config
- Validate all required fields
- Show key parameters

### Advanced: Custom Config

Create a custom config for experimentation:

```bash
cp configs/training_config.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python main.py --config configs/my_experiment.yaml
```

## 🔧 Troubleshooting

### "CUDA out of memory"

**Solution**: Reduce batch size
```yaml
training:
  per_device_train_batch_size: 8  # Try 8, 4, or even 2
```

### "Dataset not found"

**Solution**: Check internet connection and dataset name
```bash
# Manually test dataset loading
python -c "from datasets import load_dataset; load_dataset('Fremtind/all-nli-norwegian')"
```

### Training is too slow on CPU

**Solutions**:
1. Use a smaller model: `ltg/norbert3-base` instead of `large`
2. Reduce dataset size: `max_train_samples: 50000`
3. Use a cloud GPU (Google Colab, AWS, etc.)

### "Config file not found"

**Solution**: Run from project root directory
```bash
# Make sure you're in the right directory
pwd  # Should show: .../finetune-embedding-norwegian
python main.py
```

## 📖 Learning Resources

### Understanding Embeddings
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [HuggingFace Training Guide](https://huggingface.co/blog/train-sentence-transformers)

### Norwegian NLP Resources
- [Fremtind/all-nli-norwegian Dataset](https://huggingface.co/datasets/Fremtind/all-nli-norwegian)
- [NorBERT Models](https://huggingface.co/ltg)

### Advanced Topics
- [Loss Functions](https://www.sbert.net/docs/package_reference/losses.html)
- [Evaluation Metrics](https://www.sbert.net/docs/package_reference/evaluation.html)

## 🎯 Next Steps

After successfully fine-tuning on triplets:

1. **Evaluate on your domain**: Test the model on your specific use case
2. **Try multi-dataset training**: Combine NLI + STS data (see original guides)
3. **LlamaIndex fine-tuning**: Further specialize for RAG with your documents
4. **Share your model**: Push to HuggingFace Hub for others to use

## 📝 Notes

- Training is deterministic (same config = same results) thanks to `seed: 42`
- Checkpoints are saved every 500 steps by default
- Best model is automatically selected based on eval accuracy
- First run downloads ~2-5GB of model + data (then cached)

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue!

## 📄 License

This project follows the same license as your original work. Model and dataset licenses may vary - check their respective pages on HuggingFace.