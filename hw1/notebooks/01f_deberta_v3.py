#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- RESOURCE MANAGEMENT (be polite on shared server) ---
import os
import torch

# ── CPU Thread Limits ──────────────────────────────────────────────────────────
# Server has 64 CPUs. Claim at most 8 so others can work.
# PyTorch uses two thread pools: intra-op (math inside one op) and
# inter-op (parallelism across independent ops).
CPU_CORES   = 32   # ← adjust this. 4–8 is polite on a 64-core server.
torch.set_num_threads(CPU_CORES)            # intra-op parallelism
torch.set_num_interop_threads(CPU_CORES)    # inter-op parallelism
os.environ["OMP_NUM_THREADS"]  = str(CPU_CORES)  # OpenMP (used by numpy/scipy)
os.environ["MKL_NUM_THREADS"]  = str(CPU_CORES)  # Intel MKL (used by numpy)

# ── GPU Setup ─────────────────────────────────────────────────────────────────
# If GPU becomes available later, set CUDA_VISIBLE_DEVICES to limit which
# GPU(s) this notebook uses. E.g. "0" = first GPU only, "0,1" = two GPUs.
# Leave blank ("") to use all visible GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # claim 1 GPU when available

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)
import numpy as np
np.random.seed(SEED)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Device          : {DEVICE}")
print(f"CPU threads     : {torch.get_num_threads()} intra / {torch.get_num_interop_threads()} inter")
print(f"OMP/MKL threads : {CPU_CORES}")
if DEVICE == 'cuda':
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"GPU             : Not available (running on CPU)")
    print(f"  → If you have GPU access, run: ssh <gpu-node> or use your cluster's job scheduler")


# In[2]:


# --- IMPORTS & LOAD ---
import numpy as np, pandas as pd, pickle, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score

# Change the model name and batch size globally
MODEL_NAME = 'microsoft/deberta-v3-base'
DEBERTA_BATCH_SIZE = 8   

import transformers.utils.import_utils as _hf
_hf.check_torch_load_is_safe = lambda: None  # bypass torch 2.6 CVE check

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE} | Model: {MODEL_NAME}")

train_df    = pd.read_pickle('./checkpoints/train_df.pkl')
val_df      = pd.read_pickle('./checkpoints/val_df.pkl')
test_df     = pd.read_pickle('./checkpoints/test_df.pkl')
pos_weights = torch.load('./checkpoints/pos_weights.pt')

with open('./checkpoints/settings.pkl', 'rb') as f:
    S = pickle.load(f)

MAX_LEN       = S['MAX_LEN']
TARGET_LABELS = S['TARGET_LABELS']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
print(f"✓ Loaded data | Tokenizer vocab: {tokenizer.vocab_size:,}")


# In[3]:


# --- DATASET CLASS ---
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, target_labels, max_len=64,
                 model_type='bert', word2idx=None):
        self.df            = df
        self.tokenizer     = tokenizer
        self.target_labels = target_labels
        self.max_len       = max_len
        self.model_type    = model_type
        self.word2idx      = word2idx
        self.has_labels    = all(l in df.columns for l in target_labels)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        tweet = row['tweet']

        if self.model_type == 'bert':
            enc = self.tokenizer(tweet, max_length=self.max_len,
                                 padding='max_length', truncation=True,
                                 return_tensors='pt')
            out = {
                'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'token_type_ids': enc.get('token_type_ids', torch.zeros(self.max_len, dtype=torch.long)).squeeze(),
            }
        else:  # rnn
            tokens = tweet.split()[:self.max_len]
            ids    = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
            pad    = self.word2idx['<PAD>']
            ids   += [pad] * (self.max_len - len(ids))
            out    = {
                'input_ids':      torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor([1]*len(tokens) + [0]*(self.max_len - len(tokens)), dtype=torch.long),
            }

        if self.has_labels:
            out['labels'] = torch.tensor([row[l] for l in self.target_labels], dtype=torch.float32)
        return out

# No bert_tokenizer needed — GRU uses model_type='rnn'
print("✓ TweetDataset class ready")


# In[4]:


# --- DATALOADERS ---
train_ds = TweetDataset(train_df, tokenizer, TARGET_LABELS, MAX_LEN, 'bert')
val_ds   = TweetDataset(val_df,   tokenizer, TARGET_LABELS, MAX_LEN, 'bert')
test_ds  = TweetDataset(test_df,  tokenizer, TARGET_LABELS, MAX_LEN, 'bert')

train_loader = DataLoader(train_ds, batch_size=DEBERTA_BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=DEBERTA_BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=DEBERTA_BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✓ Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")


# In[9]:


# --- DEBERTA-V3 CLASSIFIER ---
class DebertaMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels=12, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa lacks a dense pooler, extract [CLS] token representation manually
        hidden = out.last_hidden_state[:, 0, :]
        
        if self.training:
            # MULTI-SAMPLE DROPOUT
            logits = torch.mean(torch.stack([
                self.fc(self.dropout(hidden)) for _ in range(5)
            ]), dim=0)
        else:
            logits = self.fc(self.dropout(hidden))
            
        return logits

model = DebertaMultiLabelClassifier(MODEL_NAME, num_labels=len(TARGET_LABELS)).to(DEVICE).float()
print(f"✓ DeBERTa-v3 params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# In[10]:


# --- 8. TRAINING UTILITIES & HELPER FUNCTIONS ---

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # 1. Compute standard BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        
        # 2. Extract probability of the true class (pt)
        pt = torch.exp(-bce)
        
        # 3. Apply Focal weight: completely suppresses loss for easy, confident predictions
        focal_loss = ((1 - pt) ** self.gamma) * bce
        
        return focal_loss.mean()

# Overwrite your loss function definition
loss_fn = FocalLoss(gamma=2.0, pos_weight=pos_weights.to(DEVICE))
print("✓ Initialized FocalLoss with gamma=2.0")


def compute_macro_f1(y_true, y_pred, threshold=0.5):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    y_pred_binary = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred_binary, average='macro', zero_division=0)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0
        self.best_model = None

    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter    = 0
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None, clip_grad=1.0):
    """
    FIX #3: scheduler is now stepped per BATCH (not per epoch).
    FIX +: gradient clipping prevents exploding gradients.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # gradient clip
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # FIX #3: step per batch, not per epoch

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device, threshold=0.5):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    macro_f1   = compute_macro_f1(all_labels, all_preds, threshold)

    return {'loss': total_loss / len(dataloader), 'macro_f1': macro_f1,
            'preds': all_preds, 'labels': all_labels}


def predict_test(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 loss_fn, device, epochs, patience=5, model_name="Model"):
    """Generic training loop used for any model."""
    early_stopping = EarlyStopping(patience=patience, delta=0.001)
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0.0

    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        train_loss  = train_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_results = evaluate(model, val_loader, loss_fn, device)
        val_loss    = val_results['loss']
        val_f1      = val_results['macro_f1']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        marker = " ★ NEW BEST" if val_f1 > best_val_f1 else ""
        best_val_f1 = max(best_val_f1, val_f1)
        print(f"Epoch {epoch+1:>2}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}{marker}")

        if early_stopping(val_f1, model):
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break

    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
        print(f"\n✓ Loaded best weights (Val F1: {early_stopping.best_score:.4f})")

    return history, early_stopping.best_score


def make_submission(model, test_loader, test_df, device, filename, threshold=0.5):
    """Generate predictions and save Kaggle submission CSV."""
    preds = predict_test(model, test_loader, device)
    binary = (preds > threshold).astype(int)
    sub = pd.DataFrame({'index': test_df['ID'].values})
    for i, label in enumerate(TARGET_LABELS):
        sub[label] = binary[:, i]
    sub.to_csv(filename, index=False)
    print(f"✓ Submission saved: {filename}  shape={sub.shape}")
    return sub


# In[ ]:


# --- TRAIN DEBERTA_V3 ---
EPOCHS = 15
LR     = 2e-5
optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=total_steps // 5,   # 20% warmup (better stability for DeBERTa)
    num_training_steps=total_steps)

history, best_f1 = run_training(model, train_loader, val_loader,
    optimizer, scheduler, loss_fn, DEVICE, EPOCHS,
    patience=5, model_name="DeBERTa-v3-base")
torch.save(model.state_dict(), './checkpoints/deberta_v3_best.pt')


# In[ ]:


# --- SUBMISSION + SAVE ---
test_preds = predict_test(model, test_loader, DEVICE)
binary     = (test_preds > 0.5).astype(int)
sub        = pd.DataFrame({'index': test_df['ID'].values})
for i, label in enumerate(TARGET_LABELS):
    sub[label] = binary[:, i]
sub.to_csv('./submission_deberta_v3.csv', index=False)

val_res = evaluate(model, val_loader, loss_fn, DEVICE)
with open('./checkpoints/history_deberta_v3.pkl', 'wb') as f:
    pickle.dump({
        'history': history, 'best_f1': best_f1,
        'val_results': val_res,
        'test_preds': test_preds    # ← needed for ensemble notebook
    }, f)
print("✓ Saved: submission_deberta_v3.csv + history_deberta_v3.pkl")


# In[ ]:


# --- PER-LABEL THRESHOLD TUNING (free F1 improvement) ---
from sklearn.metrics import f1_score
import numpy as np

val_res = evaluate(model, val_loader, loss_fn, DEVICE)
probs   = val_res['preds']    # shape: (987, 12)
labels  = val_res['labels']

best_thresholds = []
print(f"{'Label':<15} {'Best Thresh':>12} {'F1 at 0.5':>10} {'F1 tuned':>10}")
print("-" * 52)
for i, label in enumerate(TARGET_LABELS):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds_t = (probs[:, i] > t).astype(int)
        f = f1_score(labels[:, i], preds_t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    default_f1 = f1_score(labels[:, i], (probs[:, i] > 0.5).astype(int), zero_division=0)
    best_thresholds.append(best_t)
    print(f"{label:<15} {best_t:>12.2f} {default_f1:>10.4f} {best_f1:>10.4f}")

# Apply tuned thresholds to test predictions and save
test_preds = predict_test(model, test_loader, DEVICE)
binary_tuned = np.stack([
    (test_preds[:, i] > best_thresholds[i]).astype(int)
    for i in range(len(TARGET_LABELS))
], axis=1)
sub_tuned = pd.DataFrame({'index': test_df['ID'].values})
for i, label in enumerate(TARGET_LABELS):
    sub_tuned[label] = binary_tuned[:, i]
sub_tuned.to_csv('./submission_deberta_v3_tuned.csv', index=False)
print(f"\n✓ Tuned submission saved: submission_deberta_v3_tuned.csv")

import pickle
with open('./checkpoints/thresholds_deberta_v3.pkl', 'wb') as f:
    pickle.dump(best_thresholds, f)

