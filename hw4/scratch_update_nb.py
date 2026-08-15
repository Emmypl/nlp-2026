import json
import os

NOTEBOOK_PATH = '/home/emmy/emmy/mlbio/hw4/notebooks/03_inference_eval.ipynb'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

# Update cell 2 to load filtered datasets
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'val_full =' in ''.join(cell['source']):
        source = cell['source']
        source.append('\n# Load filtered validation sets\n')
        source.append('val_filtered_full = load_jsonl(f"{DATA_DIR}/val_filtered_top4_BAAI-bge-small-en-v1.5_fullInfo.jsonl")\n')
        source.append('val_filtered_struct = load_jsonl(f"{DATA_DIR}/val_filtered_top4_BAAI-bge-small-en-v1.5_structOnly.jsonl")\n')
        source.append('print(f"Loaded {len(val_filtered_full)} filtered full-info and {len(val_filtered_struct)} filtered structural samples.")\n')
        cell['source'] = source
        break

# Add cells at the end (before verify_experiment_logging)
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Evaluate Dynamic Filtering (Top 4)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate SFT Model on Filtered Dataset (StructOnly)\n",
            "try:\n",
            "    print(f\"Loading adapter from {SFT_STRUCT_DIR} for Filtered Eval...\")\n",
            "    model_struct_filtered = PeftModel.from_pretrained(base_model, SFT_STRUCT_DIR)\n",
            "    \n",
            "    print(\"Evaluating SFT Model with Dynamic Filtering (Top 4) - StructOnly...\")\n",
            "    run_evaluation(\n",
            "        model=model_struct_filtered,\n",
            "        tokenizer=tokenizer,\n",
            "        dataset=val_filtered_struct,\n",
            "        training_strategy=\"Dynamic Filtering (Top 4)\",\n",
            "        prompt_format=\"structOnly\",\n",
            "        model_name=MODEL_ID,\n",
            "        output_csv=f\"{ARTIFACTS_DIR}/experiment_summary.csv\"\n",
            "    )\n",
            "    model_struct_filtered.unload()\n",
            "except Exception as e:\n",
            "    print(f\"Could not evaluate SFT model on filtered struct dataset: {e}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate SFT Model on Filtered Dataset (FullInfo)\n",
            "try:\n",
            "    print(f\"Loading adapter from {SFT_FULL_DIR} for Filtered Eval...\")\n",
            "    model_full_filtered = PeftModel.from_pretrained(base_model, SFT_FULL_DIR)\n",
            "    \n",
            "    print(\"Evaluating SFT Model with Dynamic Filtering (Top 4) - FullInfo...\")\n",
            "    run_evaluation(\n",
            "        model=model_full_filtered,\n",
            "        tokenizer=tokenizer,\n",
            "        dataset=val_filtered_full,\n",
            "        training_strategy=\"Dynamic Filtering (Top 4)\",\n",
            "        prompt_format=\"fullInfo\",\n",
            "        model_name=MODEL_ID,\n",
            "        output_csv=f\"{ARTIFACTS_DIR}/experiment_summary.csv\"\n",
            "    )\n",
            "    model_full_filtered.unload()\n",
            "except Exception as e:\n",
            "    print(f\"Could not evaluate SFT model on filtered full dataset: {e}\")\n"
        ]
    }
]

# Insert before the last cell (verify_experiment_logging)
nb['cells'] = nb['cells'][:-1] + new_cells + [nb['cells'][-1]]

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
