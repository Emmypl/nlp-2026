import os
from pathlib import Path

def generate_experiment_name(config: dict) -> str:
    """
    Dynamically generates the canonical experiment name based on the 
    'vlm_retriever_reranker_fusion' naming convention.
    """
    # 1. VLM Part
    if os.environ.get("DISABLE_VLM", "0") == "1":
        vlm_part = "none"
    else:
        vlm_file = os.environ.get("VLM_CAPTIONS_FILE", "")
        # Extracts "qwen2-vl-7b-instruct" from "vlm_image_captions/qwen2-vl-7b-instruct_image_captions.json"
        vlm_part = Path(vlm_file).name.split("_")[0] if vlm_file else "none"

    # 2. Retriever Part (Stage 1)
    retriever_part = "none"
    if config.get("dense_scores_path"):
        # We used pre-computed scores. Determine if it was a hybrid or single model.
        path_str = str(config["dense_scores_path"])
        if "w0." in path_str:
            # E.g. "qwen3-embedding-8b_w0.6_bge-m3_w0.4..."
            retriever_part = "hybrid(qwen3-embedding-8b + bge-m3)" 
        else:
            # Single dense model cache
            retriever_part = Path(path_str).name.split('_dense')[0]
    else:
        # No pre-computed scores; we ran Stage 1 raw
        if config.get("retriever") == "bm25":
            retriever_part = "bm25"
        elif config.get("retriever") == "dense":
            retriever_part = config.get("dense_model_name", "unknown").split("/")[-1].lower()
        elif config.get("retriever") == "cached":
            # For pure cached evaluation
            retriever_part = "hybrid(qwen3-embedding-8b + bge-m3)"

    # 3. Reranker Part (Stage 2)
    reranker_part = "none"
    if config.get("retriever") == "llm":
        reranker_part = config.get("llm_model_name", "unknown").split("/")[-1].lower()
    elif config.get("retriever") == "cross_encoder":
        reranker_part = config.get("cross_encoder_model_name", "unknown").split("/")[-1].lower()

    # 4. Fusion Part
    # Standalone runs are always "none". `04_rrf_ensemble.ipynb` handles creating the "rrf" postfix natively.
    fusion_part = "none"

    return f"{vlm_part}_{retriever_part}_{reranker_part}_{fusion_part}"
