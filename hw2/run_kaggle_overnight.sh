#!/bin/bash

# Overnight Kaggle Submission Runner (v2 - The Strong Baseline Crusher)
# Usage: nohup ./run_kaggle_overnight.sh > run_kaggle_overnight.log 2>&1 &

# Environment Setup
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/emmy/miniconda3/envs/mlbio_gpu/lib/python3.10/site-packages/nvidia/cu13/lib

PM_BIN="/home/emmy/miniconda3/envs/mlbio_gpu/bin/papermill"
KERNEL_NAME="mlbio_gpu"
INPUT_NB="kaggle_submission.ipynb"

# Ensure output directory exists
mkdir -p submissions

# --- CONFIGURATION 1: GLM-Thinking CoT + Position Debiasing ---
# This is our best shot at breaking 0.56. CoT usually adds 3-5% over Zero-Shot.
echo "🚀 Starting GLM-Thinking CoT + PD..."
$PM_BIN $INPUT_NB submissions/glm_cot_pd.ipynb \
    -k $KERNEL_NAME \
    -p SELECTED_INDEX 3 \
    -p PROMPT_TYPE "cot" \
    -p USE_POSITION_DEBIASING True \
    -p USE_SELF_CONSISTENCY False

# # --- CONFIGURATION 2: GLM-Thinking Zero-Shot + PD + Self-Consistency (SC=3) ---
# # Reliability boost. If the baseline is exactly 0.56, SC will help us cross it.
# echo "🚀 Starting GLM-Thinking Zero-Shot + PD + SC(3)..."
# $PM_BIN $INPUT_NB submissions/glm_zs_pd_sc.ipynb \
#     -k $KERNEL_NAME \
#     -p SELECTED_INDEX 3 \
#     -p PROMPT_TYPE "zero_shot" \
#     -p USE_POSITION_DEBIASING True \
#     -p USE_SELF_CONSISTENCY True

echo "✅ All runs finished! Check submissions/ directory for .csv and .ipynb outputs."
