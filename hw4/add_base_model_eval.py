import json

NOTEBOOK_PATH = '/home/emmy/emmy/mlbio/hw4/notebooks/03_inference_eval.ipynb'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

# Find the index of the first Dynamic Filtering eval cell
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'Evaluate SFT Model on Filtered Dataset (StructOnly)' in ''.join(cell['source']):
        target_idx = i
        break

base_eval_cells = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate Base Model on Filtered Dataset (StructOnly)\n",
            "try:\n",
            "    print(\"Evaluating Base Model with Dynamic Filtering (Top 4) - StructOnly...\")\n",
            "    run_evaluation(\n",
            "        model=base_model,\n",
            "        tokenizer=tokenizer,\n",
            "        dataset=val_filtered_struct,\n",
            "        training_strategy=\"Dynamic Filtering (Top 4)\",\n",
            "        prompt_format=\"structOnly\",\n",
            "        model_name=MODEL_ID,\n",
            "        output_csv=f\"{ARTIFACTS_DIR}/experiment_summary.csv\"\n",
            "    )\n",
            "except Exception as e:\n",
            "    print(f\"Could not evaluate Base model on filtered struct dataset: {e}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate Base Model on Filtered Dataset (FullInfo)\n",
            "try:\n",
            "    print(\"Evaluating Base Model with Dynamic Filtering (Top 4) - FullInfo...\")\n",
            "    run_evaluation(\n",
            "        model=base_model,\n",
            "        tokenizer=tokenizer,\n",
            "        dataset=val_filtered_full,\n",
            "        training_strategy=\"Dynamic Filtering (Top 4)\",\n",
            "        prompt_format=\"fullInfo\",\n",
            "        model_name=MODEL_ID,\n",
            "        output_csv=f\"{ARTIFACTS_DIR}/experiment_summary.csv\"\n",
            "    )\n",
            "except Exception as e:\n",
            "    print(f\"Could not evaluate Base model on filtered full dataset: {e}\")\n"
        ]
    }
]

if target_idx != -1:
    nb['cells'] = nb['cells'][:target_idx] + base_eval_cells + nb['cells'][target_idx:]

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
