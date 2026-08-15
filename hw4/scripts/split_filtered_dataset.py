import os
import json
import copy
from sklearn.model_selection import train_test_split

DATA_DIR = "output/cache"
DEV_PATH = "data/addition.jsonl"
FILTERED_TRAIN_PATH = f"{DATA_DIR}/train_filtered_top3_BAAI-bge-small-en-v1.5.jsonl"

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def strip_descriptions(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == 'description':
                continue
            new_dict[k] = strip_descriptions(v)
        return new_dict
    elif isinstance(obj, list):
        return [strip_descriptions(item) for item in obj]
    else:
        return obj

def make_structural(data):
    structural = []
    for row in data:
        new_row = copy.deepcopy(row)
        new_row['options'] = strip_descriptions(new_row['options'])
        structural.append(new_row)
    return structural

print("Loading dev data to map domains...")
dev_data = load_jsonl(DEV_PATH)
tool_to_domain = {}
for item in dev_data:
    domain = item.get('domain', 'Unknown')
    for tool in item['tools']:
        tool_to_domain[tool['name']] = domain

def get_domain(row):
    correct_tool = row['options'][row['answer']]
    return tool_to_domain.get(correct_tool['name'], 'Unknown')

print("Loading filtered training data...")
train_filtered_expanded_raw = load_jsonl(FILTERED_TRAIN_PATH)

print("Filtering out samples where the true answer was lost during retrieval...")
train_filtered_expanded = []
for row in train_filtered_expanded_raw:
    if row['answer'] in row['options']:
        train_filtered_expanded.append(row)
print(f"Kept {len(train_filtered_expanded)} valid samples out of {len(train_filtered_expanded_raw)}.")

print("Creating structural configuration...")
train_filtered_structural = make_structural(train_filtered_expanded)

print("Computing stratified domains...")
domains = [get_domain(row) for row in train_filtered_expanded]

print("Performing 85/15 train-validation split...")
train_idx, val_idx = train_test_split(
    range(len(train_filtered_expanded)), 
    test_size=0.15, 
    stratify=domains,
    random_state=42
)

# Split Full Info
train_full = [train_filtered_expanded[i] for i in train_idx]
val_full = [train_filtered_expanded[i] for i in val_idx]

# Split Structural
train_struct = [train_filtered_structural[i] for i in train_idx]
val_struct = [train_filtered_structural[i] for i in val_idx]

print(f"Saving splits (Train: {len(train_full)}, Val: {len(val_full)})...")
save_jsonl(train_full, f"{DATA_DIR}/train_filtered_full_info.jsonl")
save_jsonl(val_full, f"{DATA_DIR}/val_filtered_full_info.jsonl")
save_jsonl(train_struct, f"{DATA_DIR}/train_filtered_structural.jsonl")
save_jsonl(val_struct, f"{DATA_DIR}/val_filtered_structural.jsonl")

print("✅ Successfully generated train and validation files for both structOnly and fullInfo configurations.")
