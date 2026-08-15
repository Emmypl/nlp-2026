# Homework 2: LLM-as-a-Judge: Predicting Human Preferences

## Task Introduction:
- Building your own LLM Judge that evaluates and compares AI-generated responses. Your goal is to maximize agreement with ground-truth human preferences.
- You are given a dataset of pairwise conversations. Your task is to predict the verdict which response is better:
	- A: Response 1 (dialog 1) is better.
	- B: Response 2 (dialog 2) is better.
	- tie: both responses are equally good.
	- neither: both responses are equally bad

## Task Rules & Constraints:
- Model size limit: Your model must have ≤ 9B total parameters (including MoE models — total params, not active params). The model must be open-weight (e.g., Gemma-4-E4B-IT, Qwen3.5-9B, LLaMA-3.1-8B, Mistral-7B).
- Any approach is allowed: prompt engineering, fine-tuning, reinforcement learning (e.g., DPO/PPO), ensembles, post-processing calibration — anything goes, as long as you respect the model size limit.
- API usage policy: You may call inference LLM APIs (e.g., NVIDIA NeMo) only if the underlying model is open-weight and within the 9B parameter limit. You may NOT use LLM APIs backed by closed-source models (e.g. GPT-5, Claude, Gemini, etc.) for generating predictions. Using any API for data augmentation during training is allowed but must be disclosed in your report.

## Dataset Introduction:
The dataset consists of real human preference judgments collected from a platform where users compared two anonymous AI responses to the same instruction and voted for the better one. hw2_dataset contains:
- train.json, 4000 samples: Labeled samples for training and experimentation
- test.json, 1000 samples: Unlabeled samples for submission
- sample_submission.csv, 1000 samples: Example submission file.

Each sample in train.json is a JSON object: 
- dialog_1 / dialog_2: Full conversation history as a list of message objects. Each message has a role ("user" or "assistant") and content. In multi-turn conversations (num_turns > 1), user and assistant messages alternate.
- num_turns: Number of dialogue turns.
- verdict: The ground-truth label. Only present in train.json.
- Note: test.json has the same format but without the verdict field.

## Kaggle Submission:
Submission format
- A 1001*2 .csv file, id starts from 1. 
- Prediction
	- A: Response 1 (dialog 1) is better
	- B: Response 2 (dialog 2) is better
	- tie: Both responses are equally good
	- neither: Both responses are equally bad
- Column name must be id and verdict.
There are one simple baseline and one strong baseline, beat them to get the higher score.
- Public leaderboard is calculated with approximately 30% of the test data, private leaderboard is calculated with the other 70%, so the final standings may be different. You can only view your private leaderboard score after the competition has ended. 
- Your submission is evaluated by Accuracy — the percentage of predictions that exactly match the ground-truth verdicts.
- You can submit at most 5 times each day and choose 3 of the submissions to be considered for the private leaderboard, or will otherwise default to the best public scoring submissions.

## Report Questions:
Question 1: Method Description (10%)
- What model did you use and why?
- Describe your approach and explain the reasoning in detail.
- Show your exact prompt template(s).
- Detail any additional techniques (e.g., calibration, ensembling, fine-tuned, chain-of-thought).

Question 2: Comparative Error Analysis (10%)
- Implement at least 2 different judging approaches (e.g., zero-shot vs. chain-of-thought, different prompt designs, or different models) and compare their failure modes on the training set.
- Report each method's accuracy on a held-out split of train.json.For each method, identify its most common error pattern (e.g., errors concentrated on tie cases, long responses, ambiguous instructions) and provide one concrete example.
- Explain why different methods fail differently — what does each method's error pattern reveal about its underlying weakness?

Question 3: Position Bias Analysis (10%)
- Position bias is a well-known issue in pairwise LLM evaluation — models may systematically prefer the response in a particular position (first or second), regardless of quality.
- Design an experiment using the training data to measure whether your model exhibits position bias.
- Report quantitative results (e.g., how often does your model pick A vs B?). If you implemented any debiasing strategies, describe them and evaluate their effectiveness.

## Grading Policy:
Kaggle (70%) 
- 30% based on the public leaderboard score and 70% based on the private leaderboard score
- Basic score :
	- Over strong baseline : 55
	- Over simple bassline : 40
	- Under simple baseline : 25
- Ranking score:
	- 15-(15/N)*(ranking-1), N=numbers of people in the interval
Report (30%)
- 10% for each question.

