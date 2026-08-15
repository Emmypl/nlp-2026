# Final Project Report: LLM Routing

## Question 1: Implementation Details
**Describe how you implement your router, including your choice of packages, router framework, loss functions, hyperparameters, etc.**

**Answer:**
Our router is implemented as a classical Machine Learning pipeline using a combination of feature extraction and predictive modeling. We utilize standard, robust data science packages including `pandas` for data manipulation, `scikit-learn` for traditional feature extraction (TF-IDF, SVD, RandomForest, Ridge) and validation (KFold), `lightgbm` and `xgboost` for scalable gradient boosting trees, and `sentence-transformers` / `gensim` for dense semantic feature extraction.

Our router framework is structured into three primary formulation methodologies:
1. **Classification Routing:** The task is framed as a standard multi-class classification problem. We map each training query to a single target label (the model that maximizes the reward formula) and minimize multi-class log-loss.
2. **Regression Routing:** The task is framed as an 11-way multi-target regression problem. We train 11 separate regression models (e.g., LightGBM Regressors) using Mean Squared Error (MSE) to predict the exact reward score for each candidate LLM. The router selects the model with the highest predicted reward.
3. **Cost-Sensitive Routing:** Similar to classification, but we introduce sample weights during training. The weight of each query is set to the margin (difference) between the best model's reward and the second-best model's reward. This forces the multi-class log-loss optimization to penalize mistakes heavily on queries where picking the wrong model results in a massive performance drop or cost spike.

For our features, we explored TF-IDF (e.g., max_features=5000, bigrams), Truncated SVD, Word2Vec, hand-crafted meta-features (query length, math/code symbol detection), and Dense Semantic Embeddings (`all-MiniLM-L6-v2`). Across our tree-based models (LightGBM, XGBoost, Random Forest), we used `n_estimators=100` and evaluated the models locally using 5-Fold Cross Validation.

---

## Question 2: Balancing Performance and Cost
**How do you balance the performance and cost of your routing decision? Describe the detailed design intuition and motivation of your LLM router.**

**Answer:**
To systematically balance performance and cost, our router mathematically integrates the project's target objective function directly into the training data generation phase. We define our local optimization target exactly as the competition metric: `Reward = 0.85 * P - 0.15 * (C / C_max)`. 

Instead of arbitrarily penalizing cost or writing hard-coded heuristic rules (e.g., "always use Model K if the query is short"), we calculate this exact Reward score for every single query-model combination in the training set. 

Our core intuition is that **not all routing mistakes are equally bad**. 
* If two models yield a similar reward for a query, picking the slightly suboptimal one is fine. 
* However, if a query strictly requires a massive model to get a correct answer, routing it to a small model is catastrophic. 
To address this, our best formulation uses **Cost-Sensitive Learning**. By weighing the training samples based on the reward gap between the optimal choice and the runner-up, we explicitly tell the gradient boosting trees to prioritize learning the decision boundaries that have the highest impact on the overall cost-performance tradeoff.

---

## Question 3: Method Comparisons
**Compare all the methods you have tried and use a table to display their respective performances. Which method performed the best, and why?**

**Answer:**
We systematically evaluated a grid of 50 machine learning configurations spanning 5 feature sets, 4 models, and 3 routing formulations, compared against 3 baseline strategies. 

### Respective Performances of Selected Methods
The table below lists representative configurations, including all baselines, the top performers, and comparisons across formulations and features:

| Model / Experiment ID | Formulation | Feature Configuration | CV_Reward_0.85 | CV_Avg_Performance | CV_Avg_Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle_Upper_Bound** | *Upper Bound Baseline* | - | **0.6731** | 0.8083 | 0.0072 |
| **ridge** | **Regression** | **w2v_size100_win5** | **0.4614** | 0.5696 | 0.0117 |
| **ridge** | **Regression** | **dense_all-MiniLM-L6-v2** | **0.4611** | 0.5709 | 0.0124 |
| **ridge** | **Regression** | **svd_comp500_tfidf5000_ng1-1** | **0.4578** | 0.5662 | 0.0121 |
| **ridge** | **Regression** | **meta_len1_code1_math1** | **0.4555** | 0.5675 | 0.0138 |
| **lightgbm** | **Regression** | **tfidf_max5000_ng1-1** | **0.4553** | 0.5634 | 0.0121 |
| **random_forest** | **Regression** | **dense_all-MiniLM-L6-v2** | **0.4542** | 0.5630 | 0.0125 |
| **lightgbm** | **Cost-Sensitive** | **dense_all-MiniLM-L6-v2** | **0.4518** | 0.5485 | 0.0074 |
| **xgboost** | **Cost-Sensitive** | **dense_all-MiniLM-L6-v2** | **0.4511** | 0.5752 | 0.0195 |
| *Static_Model_K* | *Static Baseline* | - | *0.4504* | 0.5324 | 0.0011 |
| **random_forest** | **Cost-Sensitive** | **w2v_size100_win5** | **0.4500** | 0.5331 | 0.0016 |
| **random_forest** | **Classification** | **dense_all-MiniLM-L6-v2** | **0.4488** | 0.5310 | 0.0013 |
| **xgboost** | **Classification** | **svd_comp500_tfidf5000_ng1-1** | **0.4482** | 0.5306 | 0.0015 |
| **xgboost** | **Cost-Sensitive** | **tfidf_max5000_ng1-1** | **0.4478** | 0.5861 | 0.0259 |
| **lightgbm** | **Classification** | **dense_all-MiniLM-L6-v2** | **0.4460** | 0.5283 | 0.0015 |
| **xgboost** | **Regression** | **meta_len1_code1_math1** | **0.4336** | 0.5405 | 0.0133 |
| **random_forest** | **Regression** | **meta_len1_code1_math1** | **0.4215** | 0.5234 | 0.0120 |
| *Random_Baseline* | *Random Baseline* | - | *0.3580* | 0.4561 | 0.0153 |

---

### Analysis & Key Discoveries

#### 1. Why Regression Outperformed Classification & Cost-Sensitive Formulations
In our experimentation, multi-class Classification and Cost-Sensitive Multi-Class Classification struggled to substantially beat the `Static_Model_K` baseline. The primary culprit is **extreme class imbalance**. 
* Because `Model_K` (the most expensive and powerful model) achieves the optimal or near-optimal reward on a vast majority of queries, standard classifiers are heavily incentivized to predict `Model_K` for almost all inputs to minimize multi-class cross-entropy. For example, `random_forest` Classification routed `9,739` out of `10,182` test queries to `Model_K`, functioning essentially as a static router.
* Conversely, **Regression Routing** models each LLM's reward independently as a continuous prediction target. This bypasses class imbalance entirely, as every model receives a continuous feedback signal for every training sample. As a result, Regression models route more dynamically: the best regression router (`ridge` on Word2Vec) only routed `4,630` queries to `Model_K` and distributed the remaining ~5,500 queries across more specialized models like `Model_H`, `Model_B`, `Model_D`, and `Model_F`. This active load-balancing raised the average performance from `0.5324` (static baseline) to `0.5696`, easily offsetting the minor cost increase.

#### 2. Why Ridge Regression Beat Tree-Based Regressors (LightGBM, XGBoost, Random Forest)
While gradient boosted trees are typically the go-to for tabular data, **Ridge Regression (linear regression with L2 regularization)** emerged as the clear winner:
* Tree-based regressors predicted reward values with high variance, causing them to overfit to noise in specific splits. This led to sub-optimal routing decisions on the validation folds (evidenced by higher average costs or lower overall rewards for XGBoost/Random Forest regression).
* Ridge regression is structurally smoother and regularized, preventing it from making extreme prediction errors. Since the reward function itself is linear (`Reward = 0.85 * P - 0.15 * C`), predicting this target with a linear estimator provides a strong inductive bias that generalizes exceptionally well.

#### 3. Semantic Features vs. Hand-Crafted Features
* **Word2Vec (`w2v_size100_win5`)** and **Dense Embeddings (`dense_all-MiniLM-L6-v2`)** yielded the highest scores (`0.4614` and `0.4611` respectively). This indicates that capturing the underlying semantic domain (e.g., recognizing code, mathematics, conversational tones, or factual reasoning) is crucial for predicting model capabilities.
* **Hand-crafted Meta-features** (`meta_len1_code1_math1`) performed surprisingly well with Ridge (`0.4555`), showing that basic indicators like length, code presence, and mathematical expressions are strong, low-dimensional signals for routing. However, when paired with trees, these few features led to severe overfitting and poor performance (e.g., Random Forest regression scoring `0.4215`).
