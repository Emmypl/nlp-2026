import json

with open('notebooks/02_sft_training.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    if "USE_UNSLOTH = False" in source:
        # We will keep USE_UNSLOTH dynamically detected, but let's just remove the hardcoded one or comment it
        cell['source'] = [line.replace("USE_UNSLOTH = False", "USE_UNSLOTH = False # Auto-detected later") for line in cell['source']]
        
    if "import os\nimport torch\nfrom unsloth" in source:
        new_source = """# === IMPORT LIBRARIES ===
# IMPORTANT: Unsloth must be imported FIRST to apply all kernel patches
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from src.utils.prompts import format_prompt

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    USE_UNSLOTH = True
    print("✅ Unsloth available! Using Unsloth.")
except ImportError:
    USE_UNSLOTH = False
    print("⚠️ Unsloth NOT available! Falling back to standard Transformers + PEFT.")
    def is_bfloat16_supported():
        return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
"""
        cell['source'] = [line + "\n" for line in new_source.split('\n')[:-1]]
        
    if "if USE_UNSLOTH == False:" in source and "peft_config = LoraConfig(" in source:
        new_source = """# === LoRA CONFIG ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
"""
        cell['source'] = [line + "\n" for line in new_source.split('\n')[:-1]]
        
    if "model, tokenizer = FastLanguageModel.from_pretrained(" in source:
        # The training loop cell
        new_source = """# === MULTI-CONFIG SFT TRAINING LOOP ===
import gc
from transformers.trainer_utils import get_last_checkpoint

max_seq_length = 1500

configs = [
    {
        "prompt_format": "structOnly",
        "train_path": f"{CACHE_DIR}/train_structural.jsonl",
        "val_path":   f"{CACHE_DIR}/val_structural.jsonl",
        "output_dir": STRUCTONLY_OUTPUT_DIR
    },
    {
        "prompt_format": "fullInfo",
        "train_path": f"{CACHE_DIR}/train_full_info.jsonl",
        "val_path":   f"{CACHE_DIR}/val_full_info.jsonl",
        "output_dir": FULLINFO_OUTPUT_DIR
    }
]

use_bf16 = is_bfloat16_supported()
use_fp16 = not use_bf16

for config in configs:
    format_name = config["prompt_format"]
    print(f"\\n================ STARTING TRAINING FOR {format_name} ==================")

    # 1. Load and process datasets
    print(f"Loading dataset from {config['train_path']}...")
    dataset = load_dataset("text", data_files={
        "train": config["train_path"],
        "val":   config["val_path"]
    })
    processed_dataset = dataset.map(lambda x: apply_chat_template(x, tokenizer))

    # 2 & 3. Load model and attach LoRA
    if USE_UNSLOTH:
        print(f"Loading {MODEL_ID} with Unsloth 4-bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=max_seq_length,
            dtype=COMPUTE_DTYPE,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print(f"Loading {MODEL_ID} with Transformers/PEFT 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=COMPUTE_DTYPE,
        )
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # 4. SFT Configuration
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        logging_dir=f"{config['output_dir']}/logs",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["val"],
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 6. Train (resume if checkpoint exists)
    last_checkpoint = get_last_checkpoint(config["output_dir"])
    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model(config["output_dir"])

    # 7. Memory cleanup
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🧹 Cleared GPU cache after {format_name}.")
    print("=================================================================\\n")
"""
        cell['source'] = [line + "\n" for line in new_source.split('\n')[:-1]]

with open('notebooks/02_sft_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
