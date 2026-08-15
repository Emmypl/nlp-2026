# HW1: Text Classification with Deep Learning

This project explores various deep learning architectures for text classification, focusing on sentiment analysis. The assignment compares traditional RNN-based models with modern transformer-based approaches, implementing ensemble methods for improved performance.

## Overview

**Task**: Sentiment analysis / text classification
**Dataset**: Twitter sentiment data (train.json, test.json, val.json)
**Models**: 10+ architectures from BiLSTM to DeBERTa
**Size**: 2.4 GB (primarily model checkpoints)

## Project Structure

```
hw1/
├── notebooks/              # Jupyter notebooks for each model
│   ├── 00_data_prep.ipynb              # Data preprocessing
│   ├── 01a_bilstm.ipynb                # BiLSTM model
│   ├── 01a_bilstm_glove_attention.ipynb # BiLSTM + GloVe + Attention
│   ├── 01b_gru.ipynb                   # GRU model
│   ├── 01c_bert_base.ipynb             # BERT base
│   ├── 01d_bertweet.ipynb              # BERTweet (Twitter-specific)
│   ├── 01e_roberta_base.ipynb          # RoBERTa base
│   ├── 01f_deberta_v3.ipynb            # DeBERTa v3
│   ├── 01f_deberta_5fold_cv.ipynb      # DeBERTa with 5-fold CV
│   ├── 02_comparison.ipynb             # Model comparison & analysis
│   └── 03_bertweet_roberta_ens.ipynb   # Ensemble model
├── data/                   # Dataset files
│   ├── train.json          # Training data
│   ├── test.json           # Test data
│   └── val.json            # Validation data
├── checkpoints/            # Saved model weights (1.2 GB)
│   ├── bilstm/
│   ├── gru/
│   ├── bert/
│   ├── roberta/
│   └── deberta/
└── submissions/            # Competition submission files
```

## Models Implemented

### 1. Traditional RNN Architectures

**BiLSTM (Bidirectional LSTM)**
- Notebook: `01a_bilstm.ipynb`
- Architecture: Embedding → BiLSTM → Dense
- Simple baseline model

**BiLSTM + GloVe + Attention**
- Notebook: `01a_bilstm_glove_attention.ipynb`
- Pre-trained GloVe embeddings
- Attention mechanism for interpretability
- Enhanced performance over vanilla BiLSTM

**GRU (Gated Recurrent Unit)**
- Notebook: `01b_gru.ipynb`
- Lightweight alternative to LSTM
- Faster training with comparable performance

### 2. Transformer-Based Models

**BERT Base**
- Notebook: `01c_bert_base.ipynb`
- Google's BERT (110M parameters)
- Transfer learning from general domain

**BERTweet**
- Notebook: `01d_bertweet.ipynb`
- Twitter-specific BERT variant
- Pre-trained on 850M+ tweets
- Domain adaptation advantage

**RoBERTa Base**
- Notebook: `01e_roberta_base.ipynb`, `01e_roberta_base_output.ipynb`
- Optimized BERT training approach
- Improved performance on sentiment tasks

**DeBERTa v3**
- Notebook: `01f_deberta_v3.ipynb`, `01f_deberta_v3_output.ipynb`
- Microsoft's DeBERTa (Decoding-enhanced BERT with disentangled attention)
- State-of-the-art architecture

**DeBERTa with 5-Fold Cross-Validation**
- Notebook: `01f_deberta_5fold_cv.ipynb`, `01f_deberta_5fold_cv_output.ipynb`
- Robust evaluation with cross-validation
- Ensemble of 5 fold models
- Best single-model performance

### 3. Ensemble Methods

**BERTweet + RoBERTa Ensemble**
- Notebook: `03_bertweet_roberta_ens.ipynb`
- Combines predictions from multiple models
- Weighted averaging for improved accuracy
- Production-ready ensemble

## Key Features

### Data Preprocessing
- Text normalization and cleaning
- Tokenization (word-level for RNNs, subword for transformers)
- Padding and batching
- Train/val/test splits

### Training Techniques
- Transfer learning from pre-trained models
- Fine-tuning with task-specific heads
- Learning rate scheduling
- Early stopping and checkpointing
- 5-fold cross-validation

### Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance analysis
- Model comparison (notebook `02_comparison.ipynb`)

## Results

### Model Comparison
The `02_comparison.ipynb` notebook provides comprehensive analysis:
- Performance metrics across all models
- Training curves and convergence analysis
- Error analysis and failure cases
- Computational efficiency comparison

### Best Performing Models
1. **DeBERTa v3 (5-fold CV)**: Highest accuracy with robust cross-validation
2. **BERTweet + RoBERTa Ensemble**: Best generalization
3. **DeBERTa v3**: Best single model without CV

### Key Insights
- Transformer models significantly outperform RNNs
- Domain-specific pre-training (BERTweet) provides substantial gains
- Ensemble methods improve robustness
- 5-fold CV reduces variance in performance estimates

## Usage

### Setup
```bash
# Install dependencies
pip install torch transformers numpy pandas scikit-learn

# For GloVe embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
```

### Training
Each notebook is self-contained and can be run independently:
```bash
jupyter notebook notebooks/01f_deberta_v3.ipynb
```

### Inference
Load pre-trained checkpoints:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./checkpoints/deberta')
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/deberta')

# Predict
inputs = tokenizer("Great movie!", return_tensors="pt")
outputs = model(**inputs)
```

## Notebooks Execution Order

1. `00_data_prep.ipynb` - Prepare and explore data
2. `01a_bilstm.ipynb` - Baseline RNN model
3. `01c_bert_base.ipynb` - First transformer model
4. `01d_bertweet.ipynb` - Domain-specific model
5. `01e_roberta_base.ipynb` - Improved transformer
6. `01f_deberta_v3.ipynb` - State-of-the-art model
7. `01f_deberta_5fold_cv.ipynb` - Robust evaluation
8. `02_comparison.ipynb` - Comprehensive analysis
9. `03_bertweet_roberta_ens.ipynb` - Final ensemble

## Technical Details

### Hyperparameters (Transformers)
- Learning rate: 2e-5 to 5e-5
- Batch size: 16-32
- Max sequence length: 128-256
- Epochs: 3-5
- Warmup steps: 500

### Hardware Requirements
- GPU recommended (CUDA support)
- 8GB+ GPU memory for transformer models
- Training time: 30 min - 2 hours per model

## Files

### Checkpoints (1.2 GB)
Pre-trained model weights for all architectures, enabling:
- Quick inference without retraining
- Ensemble model creation
- Transfer learning to related tasks

### Submissions
Competition submission files with predictions on test set.

## Key Learnings

1. **Transfer Learning**: Pre-trained models provide massive performance gains
2. **Domain Adaptation**: Twitter-specific models (BERTweet) excel on social media text
3. **Ensemble Power**: Combining diverse models improves robustness
4. **Cross-Validation**: Essential for reliable performance estimates
5. **Architecture Evolution**: DeBERTa's disentangled attention provides measurable improvements

## Future Improvements

- Experiment with larger models (XLNet, T5)
- Add data augmentation techniques
- Implement adversarial training
- Multi-task learning with related objectives
- Distillation for deployment efficiency

## References

- BERT: Devlin et al. (2018)
- RoBERTa: Liu et al. (2019)
- DeBERTa: He et al. (2020)
- BERTweet: Nguyen et al. (2020)
- GloVe: Pennington et al. (2014)

---

**Portfolio Note**: This project demonstrates proficiency in:
- Deep learning for NLP
- Transfer learning and fine-tuning
- Model comparison and evaluation
- Ensemble methods
- Production-ready implementation
