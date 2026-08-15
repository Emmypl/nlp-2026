# HW2: LLM-as-a-Judge - Predicting Human Preferences

This project builds an LLM-based judge system to evaluate and compare AI-generated responses, predicting which response better aligns with human preferences. The goal is to maximize agreement with ground-truth human judgments on pairwise comparison tasks.

## Overview

**Task**: Pairwise response comparison and preference prediction
**Dataset**: 4,000 labeled training samples, 1,000 test samples
**Constraint**: Model size ≤ 9B parameters (open-weight models only)
**Evaluation**: Accuracy on predicting human preference verdicts
**Size**: 16 MB

## Task Description

Given two AI-generated responses to the same user instruction, predict the human preference verdict:
- **A**: Response 1 (dialog_1) is better
- **B**: Response 2 (dialog_2) is better
- **tie**: Both responses are equally good
- **neither**: Both responses are equally bad

Each sample includes:
- Full conversation history (multi-turn dialogues supported)
- Role-based messages (user/assistant alternation)
- Number of dialogue turns
- Ground-truth verdict (training data only)

## Project Structure

```
hw2/
├── run_models.ipynb                # Main model evaluation pipeline
├── kaggle_submission.ipynb         # Competition submission generation
├── accuracy_analysis.ipynb         # Performance analysis
├── error_analysis.ipynb            # Failure mode investigation
├── data/
│   ├── train.json                  # 4,000 labeled samples
│   ├── test.json                   # 1,000 unlabeled samples
│   └── sample_submission.csv       # Submission format template
├── output/                         # Validation results and CSVs
├── submissions/                    # Kaggle submission files
├── summary_results.csv             # Aggregated performance metrics
├── run_overnight.sh                # Batch evaluation script
└── hw2_context.md                  # Assignment context
```

## Notebooks

### 1. Main Pipeline (`run_models.ipynb`)
- Model selection and configuration
- Prompt engineering experiments
- Zero-shot vs few-shot approaches
- Chain-of-thought reasoning
- Ensemble methods
- Validation on held-out training data

### 2. Kaggle Submission (`kaggle_submission.ipynb`)
- Test set inference
- Post-processing and calibration
- Submission file generation (CSV format)
- Multiple submission strategies

### 3. Accuracy Analysis (`accuracy_analysis.ipynb`)
- Performance metrics across approaches
- Per-verdict category breakdown (A, B, tie, neither)
- Confidence score analysis
- Calibration curves

### 4. Error Analysis (`error_analysis.ipynb`)
- Comparative failure mode analysis
- Position bias investigation
- Error pattern identification
- Concrete example showcase

## Approaches Explored

### 1. Prompt Engineering
- **Zero-shot prompting**: Direct comparison with clear instructions
- **Few-shot learning**: Example-based guidance
- **Chain-of-thought (CoT)**: Step-by-step reasoning before verdict
- **Self-consistency**: Multiple sampling with voting

### 2. Model Selection
Evaluated open-weight models ≤ 9B parameters:
- Qwen 2.5-7B-Instruct
- LLaMA 3.1-8B-Instruct
- Mistral-7B-Instruct
- Gemma-7B-IT

### 3. Debiasing Strategies
- **Position swapping**: Evaluate both (A, B) and (B, A) orders
- **Calibration**: Adjust prediction thresholds per category
- **Tie/Neither handling**: Special attention to ambiguous cases

### 4. Ensemble Methods
- Model averaging across different prompts
- Voting from multiple approaches
- Weighted ensemble based on validation performance

## Key Challenges Addressed

### 1. Position Bias
**Problem**: LLMs systematically prefer responses in certain positions

**Solution**:
- Experimental measurement on training data
- Swapping response positions and comparing predictions
- Debiasing through dual-inference and majority voting

### 2. Tie and Neither Cases
**Problem**: Models struggle with ambiguous or equally poor responses

**Solution**:
- Specialized prompts for edge cases
- Confidence thresholding
- Post-processing calibration

### 3. Multi-turn Dialogue Evaluation
**Problem**: Longer conversations require holistic understanding

**Solution**:
- Context window optimization
- Turn-by-turn vs full dialogue evaluation
- Attention to conversation coherence

## Analysis and Insights

### Comparative Error Analysis
The `error_analysis.ipynb` notebook provides:
- Confusion matrices for different approaches
- Error concentration by verdict type
- Failure pattern identification (long responses, ambiguous instructions)
- Method-specific weakness analysis

### Position Bias Measurement
Quantitative results on systematic preference:
- A vs B selection frequency
- Impact of response order on predictions
- Debiasing effectiveness evaluation

### Performance Breakdown
Results by verdict category:
- Accuracy on clear winners (A, B)
- Performance on ambiguous cases (tie, neither)
- Confidence score distribution

## Results

### Validation Performance
- **Best single model**: Qwen 2.5-7B with CoT prompting
- **Best ensemble**: Multi-prompt voting system
- **Position bias**: Reduced from X% to Y% with debiasing

### Kaggle Competition
- Public leaderboard: Beat strong baseline
- Private leaderboard: Final ranking and score
- Submission strategy: 3 best-performing approaches

## Usage

### Setup
```bash
# Install dependencies
pip install transformers torch pandas numpy scikit-learn
```

### Running Models
```bash
jupyter notebook run_models.ipynb
```

### Generate Submission
```bash
jupyter notebook kaggle_submission.ipynb
```

### Batch Evaluation
```bash
bash run_overnight.sh
```

## Prompt Template Example

```
You are an expert evaluator comparing two AI assistant responses.

Instruction: {instruction}

Response A: {response_a}

Response B: {response_b}

Think step-by-step:
1. Evaluate helpfulness, accuracy, and relevance
2. Consider response completeness
3. Check for errors or harmful content

Which response is better? Choose: A, B, tie, or neither

Verdict:
```

## Technical Details

### Model Constraints
- Maximum parameters: 9B
- Open-weight models only
- No closed-source API usage (GPT, Claude, Gemini)
- NVIDIA NeMo allowed if underlying model is open-weight

### Submission Format
CSV file with columns:
- `id`: Sample identifier (1-1000)
- `verdict`: Prediction (A, B, tie, neither)

### Evaluation Metric
**Accuracy**: Percentage of predictions matching ground-truth verdicts

## Key Learnings

1. **Position Bias is Real**: Models systematically favor certain positions without debiasing
2. **CoT Helps**: Chain-of-thought reasoning improves judgment quality
3. **Tie Cases are Hard**: Ambiguous cases require specialized handling
4. **Calibration Matters**: Raw model predictions benefit from post-processing
5. **Ensemble Strength**: Combining diverse approaches reduces error

## Future Improvements

- Fine-tuning with DPO/PPO on preference data
- Advanced calibration techniques (temperature scaling)
- Constitutional AI principles for judging
- Multi-stage evaluation pipeline
- Explainable AI for verdict justification

## Files

### Output
- Validation predictions on training data splits
- Per-sample confidence scores
- Aggregated performance metrics

### Submissions
Multiple Kaggle submission files representing different strategies

## Report Questions Addressed

1. **Method Description**: Detailed explanation of approach, model selection, prompts, and techniques
2. **Comparative Error Analysis**: At least 2 approaches compared with failure mode analysis
3. **Position Bias Analysis**: Quantitative measurement and debiasing effectiveness

## References

- Constitutional AI (Bai et al., 2022)
- Self-Consistency (Wang et al., 2022)
- Chain-of-Thought Prompting (Wei et al., 2022)
- Position Bias in LLM Evaluation (Wang et al., 2023)

---

**Portfolio Note**: This project demonstrates:
- LLM-as-a-judge system design
- Systematic bias identification and mitigation
- Prompt engineering expertise
- Comparative evaluation methodology
- Kaggle competition experience
