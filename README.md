# NLP Coursework Portfolio

This repository contains coursework from a graduate-level Natural Language Processing course, showcasing various NLP techniques, deep learning architectures, and state-of-the-art language models applied to real-world tasks.

## Projects

### [HW1: Text Classification with Deep Learning](./hw1/)
Sentiment analysis and text classification using various neural architectures including BiLSTM, GRU, BERT variants, RoBERTa, and DeBERTa. Implements ensemble methods for improved performance.

**Key Highlights:**
- Multiple deep learning architectures (RNNs, Transformers)
- Transfer learning with pre-trained models
- 5-fold cross-validation
- Ensemble methods

**Size**: 2.4 GB (model checkpoints)

### [HW2: Model Evaluation and Analysis](./hw2/)
Comprehensive model evaluation framework with accuracy analysis, error analysis, and systematic model comparison.

**Key Highlights:**
- Systematic model evaluation pipeline
- Error analysis and interpretation
- Performance optimization

**Size**: 16 MB

### [HW3: Information Retrieval and Ranking](./hw3/)
Multi-modal information retrieval system combining sparse (BM25), dense (embeddings), and neural approaches with reranking and ensemble methods.

**Key Highlights:**
- Multi-modal embeddings (text + image)
- Vision-Language Model (VLM) captions
- Retriever ablation studies (BM25, dense, hybrid)
- Cross-encoder reranking
- Reciprocal Rank Fusion (RRF) ensemble

**Size**: 2.5 GB (data with images)

### [HW4: Tool Routing with Fine-Tuned LLMs](./hw4/)
Dynamic tool filtering and hierarchical routing using fine-tuned large language models (Qwen 32B, Granite 8B) with LoRA/DoRA adaptation.

**Key Highlights:**
- **Large Model Training**: Qwen 2.5-32B-Instruct (32B parameters)
- Supervised Fine-Tuning (SFT) with LoRA/DoRA
- Dynamic tool filtering
- Hierarchical routing strategies
- Comprehensive inference and evaluation pipeline

**Size**: 3.3 GB (fine-tuned model checkpoints)

### [Final Project: Advanced Routing Systems](./finalProj/)
Advanced routing and classification system combining traditional ML, embedding-based approaches, and neural routers.

**Key Highlights:**
- Feature engineering for router classification
- ML classifiers (tree-based, ensemble)
- Embedding-based KNN routing
- Advanced neural routing architectures

**Size**: 2.1 GB (cached embeddings and models)

## Repository Structure

```
nlp/
├── hw1/                    # Text classification with deep learning
│   ├── notebooks/          # 18 notebooks (BiLSTM, BERT, DeBERTa, etc.)
│   ├── data/               # Training/test datasets
│   ├── checkpoints/        # Model checkpoints (1.2 GB)
│   └── submissions/        # Competition submissions
│
├── hw2/                    # Model evaluation and analysis
│   ├── *.ipynb             # 4 analysis notebooks
│   ├── data/               # Datasets
│   ├── output/             # Validation results
│   └── submissions/        # Submission files
│
├── hw3/                    # Information retrieval
│   ├── notebooks/          # 12 notebooks (retrieval, reranking)
│   ├── data/               # Datasets with images (2.5 GB)
│   ├── src/                # Retriever implementations
│   ├── outputs/            # Results and submissions
│   └── artifacts/          # Cached scores and captions
│
├── hw4/                    # Tool routing with LLMs
│   ├── notebooks/          # 8 notebooks (training, inference)
│   ├── data/               # Training datasets
│   ├── output/models/      # Fine-tuned models (3.1 GB)
│   │   ├── Qwen2.5-32B-Instruct_*/  # 32B parameter models
│   │   └── granite-4.1-8b_*/        # 8B parameter models
│   ├── src/                # Training and inference utilities
│   └── scripts/            # Pipeline scripts
│
├── finalProj/              # Advanced routing systems
│   ├── notebooks/          # 7 notebooks (feature eng, routers)
│   ├── data/               # Datasets
│   ├── src/                # Router implementations
│   ├── output/             # Cached features and results
│   └── validation/         # Validation results
│
└── .gitignore              # Git ignore rules
```

## Technical Stack

### Deep Learning Frameworks
- PyTorch
- Transformers (Hugging Face)
- Unsloth (efficient LLM fine-tuning)

### Pre-trained Models
- BERT, RoBERTa, DeBERTa (text classification)
- BERTweet (Twitter-specific)
- Qwen 2.5-32B-Instruct (LLM fine-tuning)
- Granite 4.1-8B (LLM fine-tuning)
- Various embedding models (BGE, MiniLM, etc.)

### NLP Techniques
- Transfer learning
- Fine-tuning with LoRA/DoRA
- Ensemble methods
- Information retrieval (BM25, dense, hybrid)
- Cross-encoder reranking
- Multi-modal embeddings

### Traditional ML
- Scikit-learn (classifiers, feature engineering)
- XGBoost, LightGBM (gradient boosting)
- KNN routing

## Key Features

### Model Training at Scale
- Fine-tuned 32B parameter models (Qwen 2.5)
- Efficient training with LoRA/DoRA adapters
- Multi-GPU training support
- Checkpoint management and evaluation

### Information Retrieval
- Multi-modal retrieval (text + images)
- Hybrid sparse-dense approaches
- Neural reranking with cross-encoders
- Ensemble methods (RRF)

### Comprehensive Evaluation
- Cross-validation frameworks
- Ablation studies
- Error analysis and interpretation
- Performance visualization

## Setup and Requirements

### Environment
Most projects use conda environments with PyTorch and Transformers. Key dependencies:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Unsloth (for efficient fine-tuning)

### Hardware Requirements
- GPU recommended for model training (hw1, hw4)
- 16GB+ RAM for large model inference
- Storage: ~11 GB for full repository

## Results and Achievements

Each homework folder contains:
- Detailed notebooks with experiments and analysis
- Final submissions to competitions/evaluations
- Comprehensive reports with figures and visualizations
- Ablation studies comparing different approaches

## Portfolio Highlights

### Technical Skills Demonstrated
- Large-scale model fine-tuning (32B parameters)
- Efficient parameter adaptation (LoRA/DoRA)
- Multi-modal learning (text + vision)
- Information retrieval and ranking
- Ensemble methods and optimization
- Systematic evaluation and ablation studies

### Best Practices
- Clean, modular code organization
- Comprehensive documentation
- Reproducible experiments
- Version control with Git
- Efficient checkpoint management

## Usage

Each project folder contains its own README with specific instructions. Generally:

1. Navigate to the project folder
2. Install dependencies (see project-specific README)
3. Explore notebooks in sequential order (00_, 01_, 02_, etc.)
4. Run inference using saved checkpoints

## License

This repository is for educational and portfolio purposes.

---

**Note**: Model checkpoints and cached embeddings are included for reproducibility. The repository has been optimized for storage (58% reduction from original 26 GB) while preserving all essential models (≥8B parameters) and portfolio-worthy content.
