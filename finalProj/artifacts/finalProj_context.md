# Final Project: LLM Routing

# Introduction
* In real-world LLM applications, routing queries to the right model can save massive amounts of compute while maintaining high user satisfaction.
* For complex reasoning tasks, you might want to route to a highly capable but expensive model. For simple informational queries, a smaller, cheaper model might suffice.
* Given a user query, your router must dynamically select the most appropriate model from a pool of 11 candidate models.
* There is no limit to the method you can use. Feel free to employ any method you prefer to build the router or process the data.

# Requirements
* Upload your submission to Kaggle.
* Submit a report and your source code to E3.
* Deadline is 06/26 (Friday) 23:59, no late submission.

# Dataset
A variety of queries and the corresponding performance and cost for 11 different LLMs.
*   `data/train.csv`: Contains the ID, the text query, and the ground-truth performance (0.0 to 1.0) and cost for each of the 11 candidate models.
*   `data/test.csv`: Contains only the ID and the text query. You must predict the best model to route to for these queries.
*   `data/sample_submission.csv`: A sample submission file showing the correct format.

# Task
Build a routing mechanism, whether it's a lightweight machine learning classifier, a similarity-based KNN router, or an LLM-as-a-Judge, to predict the optimal pred\_model for each query in the test set. 

The scoring metric is Reward\_{0.85}. The higher reward the better.
* Heavily favors performance but applies a penalty for excessive cost. It is calculated globally across your entire submission as follows:
  * $Reward_{0.85} = 0.85 \times \bar{P} - 0.15 \times \frac{\bar{C}}{\overline{C_{max}}}$
  * $\bar{P}$ is the average performance of the models you selected across all test queries.
  * $\bar{C}$ is the average cost of the models you selected across all test queries.
  * $\overline{C_{max}}$ is the average maximum cost available per query (used as a global normalization factor to scale costs between 0 and 1).

Your goal is to maximize this Reward score.

# Kaggle Submission
* Determine which model is the best choice to answer each query in test set, and then upload your routing decision to Kaggle. Note that your routing decision of each query can be different. The submission format should be:
  * A 2551\*2 .csv file, first row for column name and the last 2550 rows for your routing decision (Model\_A or Model\_B or … or Model\_K)
  * First row must match the one shown in the sample\_submission.csv. Make sure the order is correct!
* There’ll be a simple baseline and a strong baseline. Beat them to get higher score.
* Maximum submissions 3 times per day. Choose 2 of the submissions to be considered for the private leaderboard.

# Report Submission
1. Describe how you implement your router, including your choice of packages, router framework, loss functions, hyperparameters, etc.
2. How do you balance the performance and cost of your routing decision? Describe the detailed design intuition and motivation of your LLM router.
3. Compare all the methods you have tried and use a table to display their respective performances. Which method performed the best, and why?

Please answer the questions in detail to get the full point of each question.

# Grading Policy
* Kaggle (70%)
  * 30% based on the public leaderboard score and 70% based on the private leaderboard score. Both public and private board are the 50% split of test set.
  * Basic score :
    * Over strong baseline : 55
    * Over simple bassline : 40
    * Under simple baseline : 25
  * Ranking score:
    * 15 - (15/N)\*(ranking - 1), where N=numbers of people in the interval
* Report (30%)
  * 10% for each quesiton
You will receive 0 points if you do not submit the source code.
