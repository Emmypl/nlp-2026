# HW3 - Multimodal Retrieval-Augmented Generation

## Task Introduction
- Build a retrieval-augmented generation (RAG) retriever that selects the most relevant pieces of evidence from a document to support answering a question.
- The documents are multimodal, containing both text and images. Pre-generated text descriptions are provided for each image, making text-only retrieval a viable option. Users may also opt to use raw images with VLMs.
- The goal is to return a single ranked list of the top-5 most relevant evidence items (text and images combined, not 5 per modality), ordered from most to least relevant, maximizing overlap with ground-truth evidence.
- **Note:** This assignment evaluates only the retrieval stage; no final answer generation is required.

## Task Rules & Constraints
- Model size limit: Your model must be ≤ 80B total parameters (MoE included — count total params, not active). We judge this by the model's advertised size in its official name: anything labeled 80B or below qualifies, regardless of the exact parameter count. Open-weight only (e.g., Qwen3-80B-A3B, Llama-3.1-70B, Gemma-3-27B).
- Any approach is allowed: You are free to use any approach you like — the only constraints are the model size limit and the open-weight requirement.
- API usage policy:  Any API used in your retrieval pipeline (embedding, reranking, query rewriting, etc.) must be backed by an open-weight model within the 80B parameter limit. Closed-source APIs such as GPT-5, Claude are NOT allowed. 

## Dataset
The dataset consists of question-document pairs built from real-world long documents, where relevant evidence may appear in different modalities such as text, tables, and figures.
* `train.jsonl`: 2055 Labeled samples for training and experimentation
* `test.jsonl`: 1798 Unlabeled samples for submission
* `images/`: 14826 Image assets referenced by the samples 
* `sample_submission.csv`: 1798 Example submission file

## Data Introduction
Each sample in `train.jsonl` is a JSON object:
* `q_id`: Unique question ID.
* `doc_name` / `domain`: Source document name and its category.
* `question`: Natural-language question to answer from the document.
* `evidence_modality_type`: Modalities that may contain evidence, such as text, table, figure, or chart.
* `text_quotes`: A list of candidate text snippets, each with a `quote_id` and the original text content.
* `img_quotes`: A list of candidate image evidences (tables / charts / figures), each with a `quote_id`, an `img_path` pointing to the file under `image/`, and an `img_description` a pre-generated natural-language description of the image. You may use this description directly as text evidence, or generate your own with a multimodal model if you prefer. 
* `gold_quotes`: Ground-truth supporting `quote_ids`. Only available in `train.jsonl`.
* `answer_short` / `answer_interleaved`: Reference answers for development only.
Note: `test.jsonl` has the same public fields but excludes gold labels and reference answers.

## Kaggle Submission (70%)
### Submission Format
* A CSV file with 2 columns: `q_id`, `gold_quotes`.
* Each row corresponds to one question in `test.jsonl`.
* `q_id` must match the question ID exactly.
* `gold_quotes` contains your top-5 retrieved quote_ids, separated by a single space.
* Column names must be exactly `q_id`, `gold_quotes`.
* Evaluation metric: Recall@5. Submit at most 5 quote_ids per row; extra ones beyond the 5th will be ignored. Submitting fewer than 5 is allowed but reduces your maximum recall.
* Baselines for scoring:
  * **Simple Baseline:** 0.68258
  * **Strong Baseline:** 0.80046
  * **Kaggle Leaderboard Highest Score (May 21th):** 0.88597

### Evaluation
* Public leaderboard is calculated with approximately 30% of the test data, private leaderboard is calculated with the other 70%, so the final standings may be different. You can only view your private leaderboard score after the competition has ended.
* Your submission is evaluated by Recall@5, which measures how many ground-truth evidence quote_ids are covered by your top-5 retrieved results.
* You can submit at most 5 times each day and choose 3 of the submissions to be considered for the private leaderboard, or will otherwise default to the best public scoring submissions.

## Report (30%)
Please answer the following 4 questions in detail:
### Q1. Method Description (5%)
* Clearly describe your overall retrieval pipeline and the main idea behind your method.
* Explain the retriever, evidence preprocessing, ranking procedure, and any additional techniques you used.
### Q2. Comparison of Retrieval Methods (5%)
* Based on your experimental results, compare the performance of BM25, Dense Retriever, direct LLM selection, and your own method.
* Analyze the strengths and weaknesses of all four methods, and explain what you think causes the performance differences.
### Q3. Multimodal Embedding vs. Text-Description Retrieval for Image Evidence (10%)
* There are two common ways to retrieve image evidence :
    (a) Use a multimodal embedding model to encode the image directly into a vector, then compare it with the question embedding.
    (b) Convert each image into a text description, then retrieve it together with other text evidence using a text-only retriever. You may either use the `img_description` field that is already provided in the dataset, or generate your own description with a VLM / image-captioning model. 
* Briefly state which option you chose and why.
* Compare your experimental results for these two approaches. Discuss which one performs better on this task and explain what you think causes the difference.
### Q4. Modality Preference Analysis (10%)
* Examine whether your retriever shows a preference for a particular modality in its retrieval results.
    * For example, does it tend to retrieve more text evidence, or does it favor image evidence?
    * Based on your actual experimental results, describe this preference and analyze what you think causes it.
    * Even if your retriever shows no clear preference, you are still expected to explain what aspects of your method cause this balanced behavior.

## Grading policy
### Kaggle (70%) 
* 30% based on the public leaderboard score and 70% based on the private leaderboard score
* Basic score :
    * Over strong baseline : 55
    * Over simple bassline : 40
    * Under simple baseline : 25
* Ranking score:
    * 15-(15/N)*(ranking-1), N=numbers of people in the interval
### Report (30%)
* 10% for each question
You will receive 0 points if you do not submit the source code.