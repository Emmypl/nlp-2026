import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
"""# --- 04: WEIGHTED ENSEMBLE MODEL ---
import pickle, numpy as np, pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

TARGET_LABELS = ['ineffective','unnecessary','pharma','rushed','side-effect',
                 'mandatory','country','ingredients','political','none','conspiracy','religious']

# Load validation predictions, labels, and test predictions where available
val_preds, val_labels = {}, None
test_preds = {}
val_f1_scores = {}

# We will ensemble RoBERTa and BERTweet (our two highest performing models)
# that have test_preds saved in their pkl files.
models_to_load = {
    'RoBERTa': 'checkpoints/history_roberta_base.pkl',
    'BERTweet': 'checkpoints/history_bertweet.pkl'
}

for name, fpath in models_to_load.items():
    try:
        with open(fpath, 'rb') as f:
            r = pickle.load(f)
        val_preds[name] = r['val_results']['preds']
        if val_labels is None:
            val_labels = r['val_results']['labels']
        if 'test_preds' in r:
            test_preds[name] = r['test_preds']
        val_f1_scores[name] = r['best_f1']
        print(f"✓ Loaded {name} | Val F1: {val_f1_scores[name]:.4f}")
    except Exception as e:
        print(f"⚠ Could not load {name}: {e}")

print(f"\\nModels available for ensemble: {list(test_preds.keys())}")
""",
"""# --- ENSEMBLE WEIGHTING & VAL EVALUATION ---
# Weight predictions proportional to their validation F1 scores
weights = {name: f1 for name, f1 in val_f1_scores.items() if name in test_preds}
total_weight = sum(weights.values())
normalized_weights = {name: w / total_weight for name, w in weights.items()}

print("Ensemble Weights:")
for name, w in normalized_weights.items():
    print(f"  {name:<10}: {w:.4f}")

# Compute ensemble validation probabilities
val_ensemble_probs = np.zeros_like(val_labels)
for name in normalized_weights:
    val_ensemble_probs += val_preds[name] * normalized_weights[name]

# standard 0.5 threshold evaluation
base_ensemble_f1 = f1_score(val_labels, (val_ensemble_probs > 0.5).astype(int), average='macro', zero_division=0)
print(f"\\nEnsemble Val F1 (threshold=0.5): {base_ensemble_f1:.4f}")
""",
"""# --- PER-LABEL THRESHOLD TUNING ON ENSEMBLE ---
best_thresholds = []
print(f"\\n{'Label':<15} {'Best Thresh':>12} {'F1 at 0.5':>10} {'F1 tuned':>10}")
print("-" * 52)
for j, label in enumerate(TARGET_LABELS):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds_t = (val_ensemble_probs[:, j] > t).astype(int)
        f = f1_score(val_labels[:, j], preds_t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    default_f1 = f1_score(val_labels[:, j], (val_ensemble_probs[:, j] > 0.5).astype(int), zero_division=0)
    best_thresholds.append(best_t)
    print(f"{label:<15} {best_t:>12.2f} {default_f1:>10.4f} {best_f1:>10.4f}")

macro_tuned = np.mean([
    f1_score(val_labels[:, j], (val_ensemble_probs[:, j] > best_thresholds[j]).astype(int), zero_division=0)
    for j in range(12)
])
print(f"\\n✓ Ensemble Val F1 (TUNED): {macro_tuned:.4f}")
""",
"""# --- SUBMISSION ---
test_df = pd.read_pickle('./checkpoints/test_df.pkl')

# Compute ensemble test probabilities
test_ensemble_probs = np.zeros_like(test_preds[list(test_preds.keys())[0]])
for name in normalized_weights:
    test_ensemble_probs += test_preds[name] * normalized_weights[name]

# Apply tuned thresholds
binary_tuned = np.stack([
    (test_ensemble_probs[:, i] > best_thresholds[i]).astype(int)
    for i in range(len(TARGET_LABELS))
], axis=1)

sub_tuned = pd.DataFrame({'index': test_df['ID'].values})
for i, label in enumerate(TARGET_LABELS):
    sub_tuned[label] = binary_tuned[:, i]

sub_tuned.to_csv('./submission_ensemble_tuned.csv', index=False)
print(f"\\n✓ Saved final ensemble submission: submission_ensemble_tuned.csv")
"""
]

for code in code_cells:
    nb.cells.append(nbf.v4.new_code_cell(code))

with open("04_ensemble.ipynb", "w") as f:
    nbf.write(nb, f)
print("04_ensemble.ipynb created successfully.")
