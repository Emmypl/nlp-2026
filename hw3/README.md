# HW3: Multi-Modal Information Retrieval and Ranking

This project implements a comprehensive information retrieval system combining sparse (BM25), dense (embedding-based), and multi-modal approaches with neural reranking. The system handles text-image retrieval tasks with state-of-the-art ensemble methods.

## Overview

**Task**: Multi-modal information retrieval (text + images)
**Approach**: Hybrid sparse-dense retrieval with neural reranking
**Dataset**: Text queries with image-based documents
**Size**: 2.5 GB (images and cached embeddings)

## Project Structure

```
hw3/
├── notebooks/
│   ├── 00_eda.ipynb                            # Exploratory data analysis
│   ├── 01a_precompute_dense.ipynb              # Dense embedding precomputation
│   ├── 01b_precompute_multimodal_embeddings.ipynb  # Multi-modal embeddings
│   ├── 01c_generate_vlm_captions.ipynb         # Vision-Language Model captions
│   ├── 01d_fuse_dense_cache.ipynb              # Embedding fusion strategies
│   ├── 02_retriever_ablation.ipynb             # Retriever comparison study
│   ├── 03_reranker_ablation.ipynb              # Reranker comparison study
│   ├── 04_rrf_ensemble.ipynb                   # Reciprocal Rank Fusion
│   ├── 05_generate_submission.ipynb            # Final submission generation
│   └── 99_report_figures.ipynb                 # Visualization for report
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── evaluation/             # Evaluation metrics (NDCG, Recall@k)
│   └── retrievers/             # Retriever implementations
│       ├── base.py             # Base retriever interface
│       ├── bm25.py             # BM25 sparse retrieval
│       ├── dense.py            # Dense embedding retrieval
│       ├── cached.py           # Cached retrieval wrapper
│       ├── cross_encoder.py    # Cross-encoder reranker
│       └── llm.py              # LLM-based retrieval
├── data/
│   ├── *.json                  # Query and document datasets
│   └── images/                 # Image data (2.5 GB)
├── outputs/
│   ├── submissions/            # Competition submissions
│   ├── papermill_notebooks/    # Executed notebooks
│   ├── logs/                   # Experiment logs
│   └── figures/                # Performance visualizations
└── artifacts/
    ├── dense_scores/           # Cached dense retrieval scores
    ├── multimodal_scores/      # Cached multi-modal scores
    └── vlm_captions/           # Generated image captions
```

## Pipeline Overview

### Stage 1: Data Exploration and Preprocessing
**Notebook**: `00_eda.ipynb`
- Dataset statistics and analysis
- Query and document distribution
- Image data exploration
- Quality assessment

### Stage 2: Embedding Precomputation
**Dense Embeddings** (`01a_precompute_dense.ipynb`):
- Text embedding models (BGE, MiniLM, etc.)
- Document and query encoding
- Similarity computation and caching

**Multi-modal Embeddings** (`01b_precompute_multimodal_embeddings.ipynb`):
- Vision-Language models (CLIP variants)
- Image-text joint embeddings
- Multi-modal similarity scores

**VLM Captions** (`01c_generate_vlm_captions.ipynb`):
- Vision-Language Model caption generation
- Image-to-text conversion
- Caption quality evaluation

**Embedding Fusion** (`01d_fuse_dense_cache.ipynb`):
- Combining multiple embedding sources
- Weighted fusion strategies
- Cache management

### Stage 3: Retriever Ablation Study
**Notebook**: `02_retriever_ablation.ipynb`

Comprehensive comparison of retrieval methods:

**Sparse Retrievers**:
- **BM25**: Traditional keyword-based retrieval
- TF-IDF variants

**Dense Retrievers**:
- **BGE embeddings**: BAAI General Embedding
- **MiniLM**: Efficient sentence embeddings
- **Multi-modal CLIP**: Vision-language joint embeddings

**Hybrid Approaches**:
- Sparse + Dense fusion
- Late fusion with score normalization

### Stage 4: Reranker Ablation Study
**Notebook**: `03_reranker_ablation.ipynb`

Neural reranking methods:

**Cross-Encoder Rerankers**:
- BERT-based cross-encoders
- Specialized ranking models
- Fine-tuned domain-specific rankers

**LLM-based Reranking**:
- Large language model reranking
- Listwise vs pairwise ranking
- Prompt-based relevance scoring

### Stage 5: Ensemble Methods
**Notebook**: `04_rrf_ensemble.ipynb`

**Reciprocal Rank Fusion (RRF)**:
- Combining rankings from multiple retrievers
- Parameter tuning (k value)
- Weighted vs unweighted fusion

**Multi-stage Pipeline**:
1. First-stage retrieval (BM25 + Dense)
2. Second-stage reranking (Cross-encoder)
3. Final ensemble (RRF)

### Stage 6: Submission Generation
**Notebook**: `05_generate_submission.ipynb`
- Full pipeline execution on test set
- Post-processing and formatting
- Competition submission file creation

## Retriever Implementations

### BM25 Retriever (`src/retrievers/bm25.py`)
- Classic probabilistic retrieval
- Fast and efficient
- Strong baseline performance
- Keyword matching with IDF weighting

### Dense Retriever (`src/retrievers/dense.py`)
- Neural embedding-based similarity
- Semantic understanding beyond keywords
- Pre-trained models:
  - BGE-base/large
  - all-MiniLM-L6-v2
  - Custom fine-tuned models

### Cross-Encoder Reranker (`src/retrievers/cross_encoder.py`)
- Pairwise query-document scoring
- Higher accuracy than bi-encoders
- Computationally expensive (reranking only)
- Models: ms-marco-MiniLM, cross-encoder/ms-marco-electra-base

### LLM Retriever (`src/retrievers/llm.py`)
- Large language model-based ranking
- Context-aware relevance scoring
- Flexible prompt-based approach
- Listwise or pairwise ranking strategies

### Cached Retriever (`src/retrievers/cached.py`)
- Wrapper for precomputed results
- Fast inference by loading cached scores
- Disk space vs compute time tradeoff

## Key Features

### Multi-Modal Understanding
- Text-only retrieval
- Image-only retrieval
- Text + Image fusion
- VLM-generated captions as text proxy

### Comprehensive Evaluation
- Metrics: NDCG@k, Recall@k, MRR
- Ablation studies for all components
- Statistical significance testing
- Per-query performance analysis

### Optimization Techniques
- Efficient caching of embeddings
- Batch processing for large datasets
- GPU acceleration for neural models
- Multi-processing for BM25

### Ensemble Strategies
- Score-level fusion
- Rank-level fusion (RRF)
- Learned fusion weights
- Multi-stage cascading

## Results

### Retriever Performance
| Method | NDCG@10 | Recall@100 |
|--------|---------|------------|
| BM25 | X.XX | X.XX |
| Dense (BGE) | X.XX | X.XX |
| Multi-modal CLIP | X.XX | X.XX |
| Hybrid (BM25 + Dense) | X.XX | X.XX |

### Reranker Improvement
| Method | NDCG@10 | Delta |
|--------|---------|-------|
| Dense baseline | X.XX | - |
| + Cross-encoder | X.XX | +X.XX |
| + LLM reranker | X.XX | +X.XX |

### Final Ensemble
- **RRF Ensemble**: Best overall performance
- Components: BM25, Dense (BGE), Cross-encoder reranking
- Achieves robust results across diverse queries

## Usage

### Setup
```bash
# Install dependencies
pip install torch transformers sentence-transformers rank-bm25 numpy pandas

# For multi-modal models
pip install open-clip-torch pillow
```

### Run Full Pipeline
```bash
# Precompute embeddings
jupyter notebook notebooks/01a_precompute_dense.ipynb

# Run retriever ablation
jupyter notebook notebooks/02_retriever_ablation.ipynb

# Generate final submission
jupyter notebook notebooks/05_generate_submission.ipynb
```

### Use Retrievers Programmatically
```python
from src.retrievers import BM25Retriever, DenseRetriever, CrossEncoderReranker

# BM25 retrieval
bm25 = BM25Retriever()
results = bm25.retrieve(query, top_k=100)

# Dense retrieval
dense = DenseRetriever(model_name='BAAI/bge-base-en-v1.5')
results = dense.retrieve(query, top_k=100)

# Reranking
reranker = CrossEncoderReranker()
reranked = reranker.rerank(query, results, top_k=10)
```

## Technical Details

### Embedding Models
- **BGE-base-en-v1.5**: 768-dim, strong general performance
- **all-MiniLM-L6-v2**: 384-dim, efficient
- **CLIP ViT-B/32**: 512-dim, multi-modal

### Reranking Models
- **ms-marco-MiniLM-L-6-v2**: Fast cross-encoder
- **cross-encoder/ms-marco-electra-base**: Higher accuracy

### Hardware Requirements
- GPU recommended for embedding generation
- 16GB+ RAM for large document collections
- 2.5 GB storage for image data

### Performance Optimization
- Batch size tuning for GPU utilization
- FAISS indexing for dense retrieval
- Caching to avoid recomputation
- Parallel processing for BM25

## Key Learnings

1. **Hybrid is Better**: Sparse + Dense outperforms either alone
2. **Reranking Pays Off**: Cross-encoders significantly improve top-k results
3. **Multi-modal Helps**: VLM captions bridge image-text gap
4. **Ensemble Robustness**: RRF provides consistent gains across query types
5. **Cache Everything**: Precomputation essential for large-scale experiments

## Future Improvements

- Late interaction models (ColBERT)
- Learned sparse retrieval (SPLADE)
- Query expansion techniques
- Pseudo-relevance feedback
- Domain adaptation for embeddings

## References

- BM25: Robertson & Zaragoza (2009)
- Dense Retrieval: Karpukhin et al. (2020) - DPR
- Cross-Encoders: Nogueira & Cho (2019)
- RRF: Cormack et al. (2009)
- CLIP: Radford et al. (2021)

---

**Portfolio Note**: This project demonstrates:
- End-to-end information retrieval pipeline
- Multi-modal learning (text + vision)
- Systematic ablation study methodology
- State-of-the-art ensemble techniques
- Production-ready retrieval system design
