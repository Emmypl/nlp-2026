import json
import random
import os
from sklearn.model_selection import train_test_split

def load_jsonl(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def augment_options(train_data, dev_data):
    all_tools = []
    for d in dev_data:
        if 'tools' in d: all_tools.extend(d['tools'])
    
    unique_tools = {}
    for t in all_tools:
        name = t.get('name') if isinstance(t, dict) else str(t)
        if name and name not in unique_tools:
            unique_tools[name] = t
    
    available_distractors = list(unique_tools.values())
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    augmented_data = []
    for sample in train_data:
        new_sample = dict(sample)
        options = new_sample.get('options', {})
        if len(options) >= 8:
            augmented_data.append(new_sample)
            continue
            
        existing_names = {v.get('name') if isinstance(v, dict) else str(v) for v in options.values()}
        
        num_needed = 8 - len(options)
        distractors = []
        attempts = 0
        while len(distractors) < num_needed and attempts < 100:
            candidate = random.choice(available_distractors)
            name = candidate.get('name') if isinstance(candidate, dict) else str(candidate)
            if name not in existing_names:
                distractors.append(candidate)
                existing_names.add(name)
            attempts += 1
            
        all_option_values = list(options.values()) + distractors
        random.shuffle(all_option_values)
        
        old_answer_key = new_sample.get('answer')
        correct_tool_val = options.get(old_answer_key) if old_answer_key else None
            
        new_options = {}
        new_answer = None
        for i, val in enumerate(all_option_values):
            letter = letters[i]
            new_options[letter] = val
            if correct_tool_val and val == correct_tool_val:
                new_answer = letter
                
        new_sample['options'] = new_options
        if new_answer: new_sample['answer'] = new_answer
            
        augmented_data.append(new_sample)
        
    return augmented_data

if __name__ == "__main__":
    print("Loading data...")
    train_data = load_jsonl("../../data/train.jsonl")
    dev_data = load_jsonl("../../data/addition.jsonl")
    
    if not train_data:
        print("Data not found. Run this script from the src/data directory, or update paths.")
    else:
        print("Augmenting options to A-H...")
        augmented_train = augment_options(train_data, dev_data)
        
        print("Splitting data into train/val...")
        train_split, val_split = train_test_split(augmented_train, test_size=0.2, random_state=42)
        
        save_jsonl(train_split, "../../output/cache/train_full_info.jsonl")
        save_jsonl(val_split, "../../output/cache/val_full_info.jsonl")
        print("Saved augmented and split data to ../../output/cache/")
