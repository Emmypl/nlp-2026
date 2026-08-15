import re
import json
import time
import os
import pandas as pd
import torch
from jinja2 import UndefinedError
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def parse_letter(text, options=None):
	# Remove thinking block if present
	if "</think>" in text:
		text = text.split("</think>")[-1]
	elif "<think>" in text:
		# Started thinking but got cut off, no answer generated
		return None

	text = text.strip()
	if len(text) == 1 and text in "ABCDEFGH":
		return text

	# Search for isolated letters from A-H
	match = re.search(r'\b[A-H]\b', text)
	if match:
		return match.group(0)

	# Fallback: search for any occurrence of A-H
	match = re.search(r'[A-H]', text)
	if match:
		return match.group(0)

	# Tool-name recovery: model output "The selected tool is [the] <name>" without a letter.
	# Try prefix-matching the partial name against option tool names.
	if options:
		name_match = re.search(r'selected tool is (?:the )?(.+)', text, re.IGNORECASE)
		if name_match:
			raw = name_match.group(1).strip().rstrip('.\n ')
			# Normalize spaces/hyphens to underscores for comparison
			partial = re.sub(r'[\s\-]+', '_', raw.lower())
			if len(partial) >= 3:
				best_letter = None
				best_overlap = 0
				for letter, tool in options.items():
					tool_name = tool.get('name', '').lower()
					if tool_name.startswith(partial):
						overlap = len(partial)
					elif partial.startswith(tool_name):
						overlap = len(tool_name)
					else:
						overlap = 0
					if overlap > best_overlap:
						best_overlap = overlap
						best_letter = letter
				if best_letter and best_overlap >= 3:
					return best_letter

	return None

def compute_semantic_similarity(tool1, tool2, embedder):
	if not embedder:
		return 0.0
	# Create simple string representations
	desc1 = tool1.get('description', '')
	desc2 = tool2.get('description', '')
	text1 = f"{tool1.get('name', '')} {desc1}"
	text2 = f"{tool2.get('name', '')} {desc2}"
	emb1 = embedder.encode(text1, convert_to_tensor=True)
	emb2 = embedder.encode(text2, convert_to_tensor=True)
	return util.cos_sim(emb1, emb2).item()

def run_evaluation(model, tokenizer, dataset, training_strategy, prompt_format, model_name, output_csv):
	try:
		embedder = SentenceTransformer('all-MiniLM-L6-v2', device=model.device)
	except Exception:
		embedder = None
		
	results = []
	correct_count = 0
	format_error_count = 0
	option_counts = {chr(65+i): 0 for i in range(8)}
	semantic_confusion_sum = 0
	semantic_confusion_count = 0
	
	start_time = time.time()
	
	for row in tqdm(dataset, desc=f"Evaluating {training_strategy} - {prompt_format}", miniters=max(1, len(dataset) // 100), mininterval=0, maxinterval=float("inf")):
		messages = row.get("messages", [])
		if not messages:
			from src.utils.prompts import format_prompt
			messages = format_prompt(row, is_test=True)
			
		try:
			prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		except (TypeError, UndefinedError, ValueError):
			# Fallback for models with non-standard dict-based chat templates
			# (e.g. granite-20b-functioncalling expects messages={'query': ..., 'functions_str': [...]})
			# Build a plain-text prompt manually using standard Granite format.
			sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
			user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
			prompt = f"System:\n{sys_msg}\n\nQuestion:\n{user_msg}\n\nAnswer:\n"
		
		# Pre-fill assistant response with a closed think block for reasoning models to bypass slow thinking phase
		try:
			vocab = tokenizer.get_vocab()
			if "<think>" in vocab:
				prompt += "<think>\nDone thinking.\n</think>\n"
		except Exception:
			pass
			
		inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

		generation_kwargs = {
			"max_new_tokens": 10,
			"pad_token_id": tokenizer.eos_token_id,
			"do_sample": False
		}

		with torch.no_grad():
			outputs = model.generate(
				**inputs,
				**generation_kwargs
			)
			
		# Extract only the newly generated text
		generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
		generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
		
		pred_letter = parse_letter(generated_text, options=row.get('options'))
		true_letter = row['answer']

		if pred_letter is None:
			format_error_count += 1
			# Fallback
			pred_letter = 'A'
			
		if pred_letter in option_counts:
			option_counts[pred_letter] += 1
			
		is_correct = (pred_letter == true_letter)
		if is_correct:
			correct_count += 1
		else:
			# Semantic confusion
			if pred_letter in row['options'] and true_letter in row['options']:
				pred_tool = row['options'][pred_letter]
				true_tool = row['options'][true_letter]
				sim = compute_semantic_similarity(pred_tool, true_tool, embedder)
				semantic_confusion_sum += sim
				semantic_confusion_count += 1
				
		results.append({
			'id': row.get('id', ''),
			'pred': pred_letter,
			'true': true_letter,
			'generated_text': generated_text
		})
		
	latency = time.time() - start_time
	acc = correct_count / len(dataset) if dataset else 0
	fmt_err = format_error_count / len(dataset) if dataset else 0
	sem_conf = semantic_confusion_sum / semantic_confusion_count if semantic_confusion_count > 0 else 0
	
	# Save to CSV
	if output_csv:
		import os
		import pandas as pd
		import re

		def get_model_size(name):
			name = name.lower()
			if "1.2b" in name: return "1.2"
			if "2b" in name: return "2.0"
			if "8b" in name: return "8"
			if "12b" in name: return "12.0"
			if "13b" in name: return "13"
			if "20b" in name: return "20"
			if "30b" in name: return "30.0"
			if "32b" in name: return "32.0"
			m = re.search(r'(\d+(?:\.\d+)?)\s*b', name)
			if m:
				val = float(m.group(1))
				return str(val) if val % 1 != 0 else str(int(val))
			return ""

		strategy_map = {
			"baseline": "Baseline",
			"LoRA": "SFT - LoRA",
			"DoRA": "SFT - DoRA"
		}
		training_strategy_mapped = strategy_map.get(training_strategy, training_strategy)

		log_data = {
			'Training Strategy': training_strategy_mapped,
			'Method': 'Prompt_ZeroShot',
			'Prompt Format': prompt_format,
			'Model': model_name,
			'Size (B)': get_model_size(model_name),
			'Internal Validation Accuracy': round(acc, 4),
			'Kaggle Public Score': '-',
			'Epochs': '-' if training_strategy == 'baseline' else '1',
			'LR': '-' if training_strategy == 'baseline' else '2e-4',
			'Format Error Rate': round(fmt_err, 4),
			'Semantic Confusion': round(sem_conf, 4),
			'Option Bias (A)': round(option_counts.get('A', 0) / len(dataset) if dataset else 0, 4),
			'Latency Seconds': round(latency, 4),
			'Worst-Performing Domain': '',
			'Notes': ''
		}

		df_new = pd.DataFrame([log_data])
		
		if os.path.exists(output_csv):
			try:
				df_existing = pd.read_csv(output_csv)
				# Check if this exact run is already logged
				mask = (
					(df_existing['Training Strategy'] == log_data['Training Strategy']) &
					(df_existing['Prompt Format'] == log_data['Prompt Format']) &
					(df_existing['Model'] == log_data['Model'])
				)
				if mask.any():
					idx = df_existing[mask].index[0]
					for col, val in log_data.items():
						# Preserve existing Kaggle score if it exists and the new one is '-'
						if col == 'Kaggle Public Score' and pd.notna(df_existing.at[idx, col]) and str(df_existing.at[idx, col]).strip() not in ['', '-']:
							continue
						df_existing.at[idx, col] = val
					df_existing.to_csv(output_csv, index=False)
					print(f"Updated existing results in {output_csv}")
				else:
					df_combined = pd.concat([df_existing, df_new], ignore_index=True)
					df_combined.to_csv(output_csv, index=False)
					print(f"Logged new results to {output_csv}")
			except Exception as e:
				print(f"Error appending to CSV: {e}")
		else:
			df_new.to_csv(output_csv, index=False)
			print(f"Created and logged results to {output_csv}")
		
	print(f"\\n--- Evaluation Results ---")
	print(f"Training Strategy: {training_strategy}")
	print(f"Prompt Format: {prompt_format}")
	print(f"Model: {model_name}")
	print(f"Accuracy: {acc:.4f}")
	print(f"Format Error Rate: {fmt_err:.4f}")
	print(f"Semantic Confusion: {sem_conf:.4f}")
	print(f"Option Bias (A): {option_counts.get('A', 0) / len(dataset) if dataset else 0:.4f}")
	print(f"Latency: {latency:.2f} seconds")
	
	# === NEW: Save detailed predictions for Q3 Error Analysis ===
	if results:
		import os
		import pandas as pd
		
		# Set output directory to output/validation
		codebase_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
		val_out_dir = os.path.join(codebase_path, "output", "validation")
		os.makedirs(val_out_dir, exist_ok=True)
		
		# Clean the model name for file path
		clean_model_name = model_name.split("/")[-1]
		
		# Format: validation_{model}_{config}_{method}.csv
		# (e.g. validation_Mistral-Nemo-Instruct-2407_FullInfo_Prompt_ZeroShot.csv)
		pred_csv_path = os.path.join(val_out_dir, f"validation_{clean_model_name}_{prompt_format}_{training_strategy}.csv")
		
		# Convert results list to DataFrame and save
		df_results = pd.DataFrame(results)
		df_results['is_correct'] = df_results['pred'] == df_results['true']
		df_results.to_csv(pred_csv_path, index=False)
		print(f"Detailed predictions saved to: {pred_csv_path}")

	return acc, results
