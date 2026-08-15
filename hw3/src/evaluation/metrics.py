import pandas as pd
from typing import List, Union, Tuple
from tqdm.auto import tqdm

def calculate_recall_at_k(predicted_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
	if not gold_ids:
		return 0.0
	predicted_k = predicted_ids[:k]
	hits = len(set(predicted_k).intersection(set(gold_ids)))
	return hits / len(gold_ids)

def calculate_mrr_at_k(predicted_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
	if not gold_ids:
		return 0.0
	predicted_k = predicted_ids[:k]
	for rank, pred_id in enumerate(predicted_k, 1):
		if pred_id in gold_ids:
			return 1.0 / rank
	return 0.0

def calculate_hit_rate_at_k(predicted_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
	if not gold_ids:
		return 0.0
	predicted_k = predicted_ids[:k]
	for pred_id in predicted_k:
		if pred_id in gold_ids:
			return 1.0
	return 0.0

def calculate_calibration_metrics(df: pd.DataFrame) -> dict:
	"""
	Computes calibration and confidence metrics from evaluation predictions DataFrame.
	The DataFrame must contain 'scores', 'gold_quotes' (predictions), and 'target_quotes' (ground truth).
	"""
	if df.empty or "scores" not in df.columns:
		print("⚠️ Warning: DataFrame is empty or does not contain confidence scores.")
		return {}
		
	gaps = []
	max_scores = []
	
	rank1_scores_correct = []
	rank1_scores_incorrect = []
	
	gap_correct = []
	gap_incorrect = []
	
	for idx, row in df.iterrows():
		# Parse scores
		try:
			scores = [float(s) for s in str(row["scores"]).split()]
		except Exception:
			continue
			
		if not scores:
			continue
			
		max_score = scores[0]
		max_scores.append(max_score)
		
		# Calculate gap between rank 1 and rank 2
		gap = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
		gaps.append(gap)
		
		# Check correctness of Rank-1 prediction
		pred_ids = str(row["gold_quotes"]).split()
		target_ids = str(row["target_quotes"]).split()
		
		if pred_ids and target_ids:
			rank1_correct = pred_ids[0] in target_ids
			if rank1_correct:
				rank1_scores_correct.append(max_score)
				gap_correct.append(gap)
			else:
				rank1_scores_incorrect.append(max_score)
				gap_incorrect.append(gap)
				
	def mean(lst):
		return sum(lst) / len(lst) if lst else 0.0
		
	metrics = {
		"avg_max_score (confidence)": mean(max_scores),
		"avg_score_gap": mean(gaps),
		"mean_rank1_confidence_when_correct": mean(rank1_scores_correct),
		"mean_rank1_confidence_when_incorrect": mean(rank1_scores_incorrect),
		"mean_score_gap_when_correct": mean(gap_correct),
		"mean_score_gap_when_incorrect": mean(gap_incorrect),
	}
	
	print("\n" + "="*50)
	print("🎯 CONFIDENCE & CALIBRATION REPORT")
	print("="*50)
	for k, v in metrics.items():
		print(f"{k:<38}: {v:.4f}")
	print("="*50 + "\n")
	
	return metrics

def evaluate_retriever(retriever, dataset, k: int = 5, return_predictions: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
	total_recall = 0.0
	total_mrr = 0.0
	total_hit_rate_k = 0.0
	total_hit_rate_1 = 0.0
	valid_samples = 0
	predictions = []
	
	# Filter out samples lacking ground-truth evidence to build a clean target iterable
	eval_dataset = [sample for sample in dataset if sample.gold_quotes]
	total_samples = len(eval_dataset)
	
	if total_samples == 0:
		print("⚠️ Warning: Provided evaluation dataset contains zero samples with valid gold_quotes.")
		if return_predictions:
			return 0.0, pd.DataFrame()
		return 0.0
		
	print(f"🏃‍♂️ Starting validation loop across {total_samples} valid samples...")
	
	# Initialize the automated environment-aware progress bar wrapper
	progress_bar = tqdm(
		eval_dataset, 
		desc=f"Pipeline Evaluation (Recall@{k})", 
		total=total_samples,
		leave=True
	)
	
	for sample in progress_bar:
		valid_samples += 1
		
		# Execute model inference sequence
		preds = retriever.retrieve(sample, top_k=k)
		pred_ids = [p[0] for p in preds]
		pred_scores = [p[1] for p in preds]
		
		# Calculate metric score matching current sample iteration
		sample_recall = calculate_recall_at_k(pred_ids, sample.gold_quotes, k=k)
		sample_mrr = calculate_mrr_at_k(pred_ids, sample.gold_quotes, k=k)
		sample_hr_k = calculate_hit_rate_at_k(pred_ids, sample.gold_quotes, k=k)
		sample_hr_1 = calculate_hit_rate_at_k(pred_ids, sample.gold_quotes, k=1)
		
		total_recall += sample_recall
		total_mrr += sample_mrr
		total_hit_rate_k += sample_hr_k
		total_hit_rate_1 += sample_hr_1
		
		predictions.append({
			"q_id": sample.q_id,
			"gold_quotes": " ".join(pred_ids),
			"target_quotes": " ".join(sample.gold_quotes),
			"scores": " ".join(f"{s:.6f}" for s in pred_scores),
			"recall": sample_recall,
			"mrr": sample_mrr,
			"hit_rate_k": sample_hr_k,
			"hit_rate_1": sample_hr_1
		})
		
		# Compute and push the running average score directly onto the tracking bar layout
		running_avg_recall = total_recall / valid_samples
		progress_bar.set_postfix({"Running Recall": f"{running_avg_recall:.4f}"})
		
	final_recall = total_recall / valid_samples if valid_samples > 0 else 0.0
	final_mrr = total_mrr / valid_samples if valid_samples > 0 else 0.0
	final_hr_k = total_hit_rate_k / valid_samples if valid_samples > 0 else 0.0
	final_hr_1 = total_hit_rate_1 / valid_samples if valid_samples > 0 else 0.0
	
	print("\n" + "="*50)
	print(f"📊 EVALUATION REPORT (k={k})")
	print("="*50)
	print(f"Recall@{k}:      {final_recall:.4f}		Measures the percentage of queries where the ground truth is in the top-K candidates. This represents the absolute mathematical upper bound of what your Stage-2 Reranker can achieve.")
	print(f"MRR@{k}:         {final_mrr:.4f}		Evaluates the average rank position of the first correct answer across all queries.")
	print(f"Hit Rate@{k}:    {final_hr_k:.4f}		Similar to Recall@{k}, but specifically checks if the first correct answer appears within the top K positions (binary success for each query).")
	print(f"Hit Rate@1:      {final_hr_1:.4f}		The most stringent metric, measuring how often the single top-ranked candidate is correct.")
	print("="*50 + "\n")
	
	if return_predictions:
		return final_recall, pd.DataFrame(predictions)
	return final_recall