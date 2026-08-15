#!/bin/bash

# Indices match your MODEL_OPTIONS list: 
# [0]=Qwen, [1]=Llama, [2]=Ministral, [3]=Phi, [4]=Gemma, [5]=GLM
for idx in 4 5
do
    echo "=================================================="
    echo "🔄 STARTING MODEL INDEX: $idx"
    echo "=================================================="
    
    # Run the notebook. 
    # -p SELECTED_INDEX sets the model. 
    # -p VAL_SPLIT_SIZE 250 ensures the sample size is correct.
    papermill run_models.ipynb "run_models_$idx.ipynb" \
        -k mlbio_gpu \
        -p SELECTED_INDEX $idx \
        -p VAL_SPLIT_SIZE 250 \
        --log-output
        
    echo "✅ Finished model index $idx"
done