#!/usr/bin/env python3
"""
SFT Training Script: Qwen3-30B-A3B with Unsloth
================================================
Trains LoRA adapters on structOnly and fullInfo prompt formats.

VRAM profile (RTX 4090, 24GB):
  - Base model (4-bit NF4): ~16.74 GB
  - After LoRA (r=16):      ~20.12 GB
  - Headroom for training:  ~3.4 GB

Run with:
    nohup conda run -n mlbio_unsloth python scripts/run_sft_training.py \
        > output/logs/run_sft_training_Qwen3-30B-A3B_both_LoRA.log 2>&1 &
"""

import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# IMPORTANT: import unsloth FIRST so all kernel patches are applied
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
import json as _json
from datasets import load_dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# ── Path setup ────────────────────────────────────────────────
codebase_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, codebase_path)
from src.utils.prompts import format_prompt  # noqa: E402

# ── Configuration ─────────────────────────────────────────────
MODEL_ID    = "unsloth/Qwen3-30B-A3B"
CACHE_DIR   = os.path.join(codebase_path, "output", "cache")
MODELS_DIR  = os.path.join(codebase_path, "output", "models")
LOG_DIR     = os.path.join(codebase_path, "output", "logs")

STRUCTONLY_OUTPUT_DIR = os.path.join(MODELS_DIR, "sft_Qwen3-30B-A3B_structOnly_LoRA")
FULLINFO_OUTPUT_DIR   = os.path.join(MODELS_DIR, "sft_Qwen3-30B-A3B_fullInfo_LoRA")

COMPUTE_DTYPE   = torch.bfloat16 if is_bfloat16_supported() else torch.float16
MAX_SEQ_LENGTH  = 1500

LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

TRAIN_BATCH     = 1
EVAL_BATCH      = 1
GRAD_ACCUM      = 4
LR              = 2e-4
NUM_EPOCHS      = 1
EVAL_STEPS      = 100
SAVE_STEPS      = 100
EARLY_STOP_PAT  = 3

# ── Dataset formatting ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def apply_chat_template(example, tokenizer=tokenizer):
    data     = _json.loads(example["text"])
    messages = format_prompt(data)
    prompt   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": prompt}

# ── Training configs ──────────────────────────────────────────
configs = [
    {
        "prompt_format": "structOnly",
        "train_path": os.path.join(CACHE_DIR, "train_structural.jsonl"),
        "val_path":   os.path.join(CACHE_DIR, "val_structural.jsonl"),
        "output_dir": STRUCTONLY_OUTPUT_DIR,
    },
    {
        "prompt_format": "fullInfo",
        "train_path": os.path.join(CACHE_DIR, "train_full_info.jsonl"),
        "val_path":   os.path.join(CACHE_DIR, "val_full_info.jsonl"),
        "output_dir": FULLINFO_OUTPUT_DIR,
    },
]

use_bf16 = is_bfloat16_supported()
use_fp16 = not use_bf16

# ── Main training loop ────────────────────────────────────────
for config in configs:
    fmt  = config["prompt_format"]
    print(f"\n{'='*60}")
    print(f"  TRAINING: {fmt}")
    print(f"  Model:    {MODEL_ID}")
    print(f"  Output:   {config['output_dir']}")
    print(f"{'='*60}")

    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. Load dataset
    print(f"Loading dataset: {config['train_path']}")
    dataset = load_dataset("text", data_files={
        "train": config["train_path"],
        "val":   config["val_path"],
    })
    processed = dataset.map(apply_chat_template)

    # 2. Load model with Unsloth 4-bit
    print(f"Loading {MODEL_ID} with Unsloth 4-bit NF4...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_ID,
        max_seq_length  = MAX_SEQ_LENGTH,
        dtype           = COMPUTE_DTYPE,
        load_in_4bit    = True,
    )
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after base load: {alloc:.2f} GB")

    # 3. Attach LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = LORA_R,
        target_modules           = TARGET_MODULES,
        lora_alpha               = LORA_ALPHA,
        lora_dropout             = LORA_DROPOUT,
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",
        random_state             = 3407,
    )
    model.print_trainable_parameters()
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after LoRA:      {alloc:.2f} GB")

    # 4. SFT Config
    training_args = SFTConfig(
        output_dir                  = config["output_dir"],
        per_device_train_batch_size = TRAIN_BATCH,
        per_device_eval_batch_size  = EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        logging_steps               = 10,
        logging_dir                 = os.path.join(config["output_dir"], "logs"),
        num_train_epochs            = NUM_EPOCHS,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        optim                       = "paged_adamw_8bit",
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model             = model,
        train_dataset     = processed["train"],
        eval_dataset      = processed["val"],
        processing_class  = tokenizer,
        args              = training_args,
        callbacks         = [EarlyStoppingCallback(
                                early_stopping_patience=EARLY_STOP_PAT)],
    )

    # 6. Train
    ckpt = get_last_checkpoint(config["output_dir"])
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    trainer.save_model(config["output_dir"])
    print(f"✅ Saved model to {config['output_dir']}")

    # 7. Cleanup
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🧹 Cleared VRAM after {fmt}.\n")

print("🎉 All training runs complete.")
