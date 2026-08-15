# Intro. to NLP HW1 - Multi-label Tweet Classification

## Objective
Predict anti-vaccine concerns in COVID-19 tweets. This is a multi-label classification task with 12 specific target labels.

## Target Labels (12 Concerns)
- ineffective, unnecessary, pharma, rushed, side-effect, mandatory, country, ingredients, political, none, conspiracy, religious.

## Constraints & Requirements
- **Goal:**  Apply deep learning model to predict the concern in tweets
- **Models:** Implement at least one recurrent neural network model (RNN/LSTM/GRU) to predict concerns in tweets (multi-label classification). Use a Transformer-based model (e.g., BERT) for the same task and compare its performance with your RNN/LSTM/GRU model.
- **Parameters:** Pretrained models (e.g., BERT) are allowed, but the total number of model parameters must be under 1 billion (1B).
- **Prohibited:** Large Language Models (LLMs) are NOT allowed.
- **Metric:** Evaluation is based on Macro-F1 score.
- **Environment:** PyTorch, Pandas, Transformers library. You may use any external packages, such as PyTorch, Keras, scikit-learn.

## Data Structure
- COVID-19 anti-vaccine tweets labelled with various specific anti-vaccine concerns in a multi-label setting.
- `train.json`, `val.json`, `test.json`, `sample_submission.csv`.
- Each tweet has an 'ID', 'tweet' content, and nested 'labels'. Each tweet can have multiple concerns.

## Kaggle Submission Format
- Model is expected to classify the concern of the tweets in 'test.json' file, and then upload your model's predictions to Kaggle. the submission format should be :
- A 1977*13 .csv file, first row for column name and the last 1976 rows for your result ( '1' indicates that the tweet has the concern, '0' signifies that it does not )
- First row must match the one shown in the sample_submission.csv, make sure the order is correct !

## Report Structure
Assignment also contains report to write. Keep these questions in mind when working on the assignment to answer the questions. Contains the following questions:
Question 1: Implementation
- (5%) Describe how you build your model ? 
- (5%) How did you do to preprocess your data from dataset ? The distribution of the concern is imbalanced, what did you do to improve the macro F1 score on those concern which are in small scale ?

Question 2: When implementing your model…
- (5%) Have you tried pretrained word embedding ? (e.g. Glove or Word2vec) What was their influence on the result after using them ?
- (5%) Have you tried attention on your model ? What was its influence on the result? When your model predict the concern, what was it focusing on ? Do some case studies.

Question 3:
= (10%) Compare the differences in performance between your RNN/LSTM/GRU model and the Transformer-based model. Which approach performs better? Explain the possible reasons for the difference.
