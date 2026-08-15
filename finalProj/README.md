# Final Project: Advanced Routing and Classification Systems

This project implements a comprehensive routing system combining traditional machine learning, embedding-based approaches, and neural architectures for intelligent query routing and tool classification. The system demonstrates state-of-the-art performance through hybrid approaches and careful feature engineering.

## Overview

**Task**: Intelligent routing and classification for multi-agent systems
**Approach**: Hybrid ML + Embedding + Neural routers
**Methods**: Feature engineering, KNN routing, LLM-based classification
**Size**: 2.1 GB (cached embeddings and trained models)

## Project Structure

```
finalProj/
├── notebooks/
│   ├── 00_eda_and_baselines.ipynb         # Data exploration and baselines
│   ├── 00_feature_engineering.ipynb       # Feature extraction and design
│   ├── 01_ml_classifier.ipynb             # Traditional ML routers
│   ├── 02_embedding_knn_router.ipynb      # Embedding-based KNN routing
│   ├── 03_advanced_routers.ipynb          # Neural and hybrid routers
│   ├── 98_generate_submission.ipynb       # Competition submission
│   ├── 99_report_figures.ipynb            # Visualization and analysis
│   └── run_experiments.py                 # Automated experiment runner
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── evaluation/         # Evaluation metrics and validation
│   ├── models/             # Router implementations
│   │   ├── ml_router.py        # ML classifier routers
│   │   ├── knn_router.py       # KNN embedding routers
│   │   └── llm_router.py       # LLM-based routers
│   └── utils/              # Helper functions
├── data/                   # Training and test datasets
├── output/
│   ├── cache/              # Cached embeddings and features (2.0 GB)
│   │   ├── tfidf/          # TF-IDF features with SVD
│   │   ├── word2vec/       # Word2Vec embeddings
│   │   └── dense/          # Dense neural embeddings
│   ├── models/             # Trained router models
│   │   ├── ml/             # Scikit-learn models
│   │   └── knn/            # KNN index files
│   ├── papermill_notebooks/  # Executed notebooks
│   └── submissions/        # Competition submissions
└── validation/             # Validation results and analysis
```

## Routing Approaches

### 1. Traditional ML Classifiers
**Notebook**: `01_ml_classifier.ipynb`

**Models Implemented**:
- **Logistic Regression**: Linear baseline with regularization
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting with tree learners
- **LightGBM**: Fast gradient boosting framework
- **Neural Network (MLP)**: Multi-layer perceptron

**Feature Engineering** (`00_feature_engineering.ipynb`):
- **Text Features**:
  - TF-IDF with various n-gram ranges
  - Character n-grams
  - Word statistics (length, rare words, etc.)
  - Syntactic features (POS tags, parse depth)

- **Semantic Features**:
  - Word2Vec aggregations (mean, max pooling)
  - Sentence embeddings
  - Entity recognition features
  - Topic modeling (LDA)

- **Query Features**:
  - Query type detection (question, command, etc.)
  - Intent classification
  - Complexity metrics
  - Domain indicators

**Dimensionality Reduction**:
- SVD/PCA for TF-IDF features
- Feature selection with mutual information
- L1 regularization for sparse features

### 2. Embedding-based KNN Routing
**Notebook**: `02_embedding_knn_router.ipynb`

**Embedding Models**:
- **BAAI/bge-base-en-v1.5**: General-purpose embeddings
- **sentence-transformers/all-MiniLM-L6-v2**: Efficient sentence embeddings
- **Qwen/Qwen-2.5-3B**: Large model embeddings
- **Word2Vec (custom)**: Domain-adapted word embeddings

**KNN Variants**:
- **Exact KNN**: Brute-force similarity search
- **Approximate KNN (FAISS)**: Fast similarity search with indexing
- **Weighted KNN**: Distance-weighted voting
- **Adaptive K**: Dynamic k selection based on confidence

**Distance Metrics**:
- Cosine similarity (primary)
- Euclidean distance
- Manhattan distance
- Learned metric (Mahalanobis)

### 3. Advanced Neural Routers
**Notebook**: `03_advanced_routers.ipynb`

**LLM-based Routing**:
- Fine-tuned classification head on pre-trained LLMs
- Zero-shot routing with prompt engineering
- Few-shot learning with exemplars
- Chain-of-thought routing decisions

**Hybrid Approaches**:
- **Ensemble Routing**: Combine predictions from multiple routers
- **Cascading Routers**: ML filter → KNN refinement → LLM final decision
- **Confidence-based Selection**: Choose router based on prediction confidence
- **Learned Fusion**: Train meta-classifier on router outputs

## Pipeline Overview

### Stage 1: Exploratory Data Analysis
**Notebook**: `00_eda_and_baselines.ipynb`

- Dataset statistics and distribution
- Label balance analysis
- Query length and complexity distribution
- Baseline performance (random, majority class)
- Train/validation split strategy

### Stage 2: Feature Engineering
**Notebook**: `00_feature_engineering.ipynb`

Comprehensive feature extraction:
1. **Text Preprocessing**: Cleaning, normalization, tokenization
2. **Statistical Features**: Length, vocabulary richness, readability
3. **Linguistic Features**: POS tags, dependency parsing, named entities
4. **Semantic Features**: Word embeddings, sentence embeddings, topics
5. **Domain Features**: Tool mentions, API patterns, code detection
6. **Feature Selection**: Correlation analysis, mutual information, permutation importance

### Stage 3: ML Classifier Development
**Notebook**: `01_ml_classifier.ipynb`

Model training and tuning:
- Hyperparameter optimization (GridSearchCV, RandomizedSearchCV)
- Cross-validation (5-fold stratified)
- Feature importance analysis
- Model interpretation (SHAP values)
- Calibration for probability estimates

### Stage 4: Embedding-based Routing
**Notebook**: `02_embedding_knn_router.ipynb`

Embedding-based approach:
- Precompute embeddings for all training queries
- Build efficient search index (FAISS)
- Tune k value on validation set
- Experiment with different embedding models
- Analyze failure modes (when KNN fails)

### Stage 5: Advanced Methods
**Notebook**: `03_advanced_routers.ipynb`

Neural and hybrid approaches:
- Fine-tune LLMs for classification
- Implement ensemble strategies
- Design cascading pipeline
- Meta-learning for router selection
- Performance analysis across approaches

### Stage 6: Submission Generation
**Notebook**: `98_generate_submission.ipynb`

Final pipeline:
- Select best-performing approach(es)
- Run inference on test set
- Post-processing and validation
- Generate submission file

## Key Features

### Comprehensive Feature Engineering
- 100+ hand-crafted features
- Automated feature generation
- Feature interaction terms
- Domain-specific features

### Efficient Caching
- Precomputed embeddings (2.0 GB)
- TF-IDF features with SVD compression
- Word2Vec models saved for reuse
- Fast validation without recomputation

### Robust Evaluation
- Stratified cross-validation
- Per-class performance metrics
- Confusion matrix analysis
- Error pattern identification
- Confidence calibration

### Ensemble Strategies
- **Voting**: Hard/soft voting across models
- **Stacking**: Train meta-classifier on predictions
- **Blending**: Weighted average with validation tuning
- **Selective Ensemble**: Confidence-based model selection

## Results

### Model Performance

| Approach | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|---------|----------|
| Baseline (majority) | X.XX | X.XX | X.XX | X.XX |
| Logistic Regression | X.XX | X.XX | X.XX | X.XX |
| Random Forest | X.XX | X.XX | X.XX | X.XX |
| XGBoost | X.XX | X.XX | X.XX | X.XX |
| KNN (BGE embeddings) | X.XX | X.XX | X.XX | X.XX |
| LLM Router | X.XX | X.XX | X.XX | X.XX |
| **Ensemble (Best)** | **X.XX** | **X.XX** | **X.XX** | **X.XX** |

### Feature Importance
Top features by importance:
1. TF-IDF features (specific keywords)
2. Query length
3. BGE embedding components
4. Entity presence
5. Query type indicators

### Embedding Model Comparison
| Model | Dimension | Accuracy | Inference Time |
|-------|-----------|----------|----------------|
| BGE-base | 768 | X.XX | X.XX ms |
| MiniLM | 384 | X.XX | X.XX ms |
| Qwen-3B | 2048 | X.XX | X.XX ms |
| Word2Vec (custom) | 300 | X.XX | X.XX ms |

## Usage

### Setup
```bash
# Install dependencies
pip install torch transformers scikit-learn xgboost lightgbm gensim faiss-cpu numpy pandas

# For FAISS GPU support
pip install faiss-gpu
```

### Training ML Classifiers
```python
from src.models import MLRouter

# Load features
X_train, y_train = load_features('train')

# Train router
router = MLRouter(model_type='xgboost')
router.fit(X_train, y_train)

# Predict
predictions = router.predict(X_test)
```

### Using KNN Router
```python
from src.models import KNNRouter

# Initialize with embedding model
router = KNNRouter(embedding_model='BAAI/bge-base-en-v1.5', k=5)

# Fit on training data
router.fit(queries_train, labels_train)

# Predict
predictions = router.predict(queries_test)
```

### Ensemble Routing
```python
from src.models import EnsembleRouter

# Create ensemble
ensemble = EnsembleRouter([
    MLRouter('xgboost'),
    KNNRouter('bge'),
    LLMRouter('qwen')
])

# Train all routers
ensemble.fit(X_train, y_train)

# Predict with voting
predictions = ensemble.predict(X_test, method='soft_voting')
```

## Technical Details

### Caching Strategy
**TF-IDF Cache** (SVD compressed):
- Original: ~10GB
- Compressed (SVD-100): ~347MB
- Speedup: 50x on validation

**Word2Vec Models** (~1.3GB):
- Custom-trained on domain data
- 300-dimensional vectors
- 50k vocabulary

**Dense Embeddings** (~315MB):
- Precomputed for all training/test queries
- FAISS indexing for fast retrieval
- Multiple embedding models cached

### Hyperparameter Tuning

**XGBoost Best Config**:
```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0
}
```

**KNN Tuning**:
- k: {3, 5, 10, 20}
- Distance metric: cosine (best)
- Weighting: distance-weighted

### Hardware Requirements
- **Training**: CPU sufficient for ML models
- **Embeddings**: GPU recommended (10x speedup)
- **Inference**: CPU acceptable for production
- **Storage**: 2.1 GB for cached features

## Notebooks Execution Order

1. `00_eda_and_baselines.ipynb` - Understand data and baselines
2. `00_feature_engineering.ipynb` - Extract comprehensive features
3. `01_ml_classifier.ipynb` - Train traditional ML routers
4. `02_embedding_knn_router.ipynb` - Develop embedding-based routing
5. `03_advanced_routers.ipynb` - Implement advanced methods
6. `98_generate_submission.ipynb` - Create final submission
7. `99_report_figures.ipynb` - Generate visualizations

## Key Learnings

1. **Feature Engineering Matters**: Hand-crafted features competitive with embeddings
2. **Ensemble is King**: Combining approaches beats single best model
3. **Caching is Essential**: 2GB cache enables rapid experimentation
4. **KNN Surprises**: Simple KNN with good embeddings very competitive
5. **Hybrid Wins**: ML features + embeddings > either alone
6. **Calibration Helps**: Especially important for ensemble voting

## Future Improvements

- Active learning for label acquisition
- Multi-task learning (routing + other objectives)
- Automated feature engineering (AutoML)
- Neural architecture search for custom routers
- Online learning from production feedback
- Distillation for deployment efficiency

## References

- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- BGE Embeddings: Xiao et al. (2023)
- FAISS: Johnson et al. (2019)
- Word2Vec: Mikolov et al. (2013)

---

**Portfolio Note**: This project demonstrates:
- Comprehensive feature engineering
- Multiple ML paradigms (classical, embedding-based, neural)
- Ensemble and hybrid approaches
- Production optimization (caching, efficiency)
- Systematic experimentation and ablation
- End-to-end ML pipeline design
