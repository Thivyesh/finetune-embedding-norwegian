#!/usr/bin/env python3
"""
Push the trained embedding model to HuggingFace Hub with proper model card.
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi

def create_model_card():
    """Create a clean, professional model card."""
    return """---
language:
- "no"
- "en"
license: gemma
base_model: google/embeddinggemma-300m
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- norwegian
- health
- medical
- information-retrieval
- rag
- domain-adaptation
library_name: sentence-transformers
pipeline_tag: feature-extraction
datasets:
- thivy/eti-embedding-training-data-2048
- NorQuAD
- ScandiQA
metrics:
- cosine_accuracy@1
- cosine_accuracy@3  
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: EmbeddingGemma 300M Norwegian Health Domain Adapted
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      type: norwegian-health-qa
      name: Norwegian Health QA Test Set
    metrics:
    - type: cosine_ndcg@10
      value: 0.9565
      name: NDCG@10
    - type: cosine_mrr@10  
      value: 0.9472
      name: MRR@10
    - type: cosine_accuracy@1
      value: 0.925
      name: Accuracy@1
    - type: cosine_accuracy@10
      value: 0.985
      name: Accuracy@10
---

# EmbeddingGemma 300M Norwegian Health Domain Adapted

This is a domain-adapted version of Google's EmbeddingGemma-300M model, fine-tuned specifically for Norwegian health and medical information retrieval tasks.

## Model Description

**Base Model**: google/embeddinggemma-300m  
**Architecture**: Gemma-based encoder with pooling and normalization layers  
**Embedding Dimension**: 768  
**Max Sequence Length**: 2048 tokens  
**Language**: Norwegian (with English support)  
**Domain**: Health, medical, and general Norwegian text  

## Training Data

The model was fine-tuned on a curated dataset of Norwegian question-answer pairs focusing on health and medical information:

- **ETI Medical Data**: 77,310 training pairs from Norwegian health documents
- **NorQuAD**: Norwegian reading comprehension questions  
- **ScandiQA**: Scandinavian question-answering dataset
- **Supervised Domain Adaptation**: Curated Norwegian QA pairs

**Total Training Samples**: ~80,000 (anchor, positive) pairs  
**Validation**: 788 pairs  
**Test**: 790 pairs  

## Performance

The model demonstrates excellent retrieval performance on Norwegian health information:

| Metric | Score |
|--------|--------|
| NDCG@10 | 95.65% |
| MRR@10 | 94.72% |
| Accuracy@1 | 92.50% |
| Accuracy@10 | 98.50% |
| MAP@100 | 94.76% |

## Training Configuration

- **Learning Rate**: 5e-6 (conservative for domain adaptation)
- **Epochs**: 1.0  
- **Batch Size**: 32  
- **Loss Function**: CachedMultipleNegativesRankingLoss
- **Mini Batch Size**: 16 (for gradient caching)
- **Training Time**: ~3.6 hours on H100

## Usage

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('thivy/embeddinggemma-300m-norwegian-health')

# Encode text
texts = [
    "Hva er symptomene på diabetes?",
    "Diabetes kan gi tørste, hyppig vannlating og tretthet."
]
embeddings = model.encode(texts)

# Calculate similarity
similarity = model.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")
```

### Information Retrieval

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('thivy/embeddinggemma-300m-norwegian-health')

# Example: Medical FAQ retrieval
queries = ["Hva er symptomene på influensa?"]
documents = [
    "Influensa gir feber, hodepine og muskelsmerter.",
    "Diabetes kan føre til økt tørste og hyppig vannlating.",
    "Høyt blodtrykk kan være årsak til hodepine."
]

# Encode
query_embeddings = model.encode(queries)
doc_embeddings = model.encode(documents)

# Find most relevant document
similarities = np.dot(query_embeddings, doc_embeddings.T)
best_match_idx = similarities.argmax()
print(f"Best match: {documents[best_match_idx]}")
```

### RAG Pipeline Integration

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('thivy/embeddinggemma-300m-norwegian-health')

# Create vector database
documents = [...]  # Your Norwegian health documents
doc_embeddings = model.encode(documents)

# Create FAISS index
index = faiss.IndexFlatIP(768)  # Inner product for cosine similarity
index.add(doc_embeddings.astype('float32'))

# Query
query = "Hva er behandling for høyt blodtrykk?"
query_embedding = model.encode([query])

# Search
k = 5  # Top 5 results
scores, indices = index.search(query_embedding.astype('float32'), k)

# Get relevant documents
relevant_docs = [documents[i] for i in indices[0]]
```

## Technical Details

**Architecture Components**:
1. **Transformer Encoder**: Gemma-based language model  
2. **Pooling Layer**: Mean pooling of token representations  
3. **Dense Layers**: Two linear projection layers  
4. **Normalization**: L2 normalization for cosine similarity  

**Training Approach**:
- Domain adaptation rather than full fine-tuning
- Conservative learning rate to prevent catastrophic forgetting  
- CachedMultipleNegativesRankingLoss for efficient contrastive learning
- InformationRetrievalEvaluator for retrieval-specific metrics

## Intended Use

This model is specifically designed for:
- Norwegian health information retrieval
- Medical question-answering systems  
- RAG pipelines for healthcare applications
- Semantic search in Norwegian medical documents
- Clinical decision support tools

## Limitations

- Optimized primarily for Norwegian text (limited multilingual capability)
- Domain-specific to health/medical content
- May not perform optimally on other domains without further adaptation
- Requires sufficient context for optimal performance (not suitable for very short texts)

## Citation

If you use this model, please cite:

```
@misc{embeddinggemma-300m-norwegian-health,
    title={EmbeddingGemma 300M Norwegian Health Domain Adapted},
    author={Thivyesh},
    year={2026},
    url={https://huggingface.co/thivy/embeddinggemma-300m-norwegian-health}
}
```

## License

This model inherits the license from the base EmbeddingGemma model. Please refer to Google's licensing terms for EmbeddingGemma.
"""

def push_model_to_hub():
    """Push the trained model to HuggingFace Hub."""
    
    # Configuration
    MODEL_PATH = "models/embeddinggemma-300m-domain-adapted/final"
    HUB_MODEL_ID = "thivy/embeddinggemma-300m-norwegian-health"
    
    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Load the model
    model = SentenceTransformer(MODEL_PATH)
    
    print("Creating model card...")
    model_card = create_model_card()
    
    print(f"Pushing model to Hub: {HUB_MODEL_ID}")
    
    # Push to hub
    model.push_to_hub(
        repo_id=HUB_MODEL_ID,
        commit_message="Add domain-adapted EmbeddingGemma-300M for Norwegian health information retrieval",
        private=False,
        create_pr=False
    )
    
    # Write model card to README.md
    print("Writing model card...")
    readme_path = f"{MODEL_PATH}/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    # Push the updated README
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=HUB_MODEL_ID,
        commit_message="Add detailed model card"
    )
    
    print(f"✓ Successfully pushed model to: https://huggingface.co/{HUB_MODEL_ID}")
    print("\nModel card preview:")
    print("-" * 60)
    print(model_card[:500] + "...")

if __name__ == "__main__":
    push_model_to_hub()