def format_prompt(sample, is_test=False):
	system_prompt = "You are a highly capable AI assistant that selects the correct tool to use based on the user's task context. Given the historical context and the current step, select the appropriate tool from the provided options. You must output ONLY a single uppercase letter corresponding to the correct tool option (e.g., A, B, C)."
	
	# Construct options string
	options_str = ""
	for key, tool in sample.get('options', {}).items():
		options_str += f"Option {key}:\n{tool}\n\n"
		
	user_prompt = f"Historical Context:\n{sample.get('full_context', '')}\n\nCurrent Step:\n{sample.get('current_step', '')}\n\nOptions:\n{options_str}\nSelected Tool:"
	
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt}
	]
	
	if not is_test and 'answer' in sample:
		messages.append({"role": "assistant", "content": str(sample['answer']).strip()})
		
	return messages
