import papermill as pm
import sys
from pathlib import Path

def main():
    # =========================================================================
    # 🎛️ MASTER CONTROL RUNTIME CONFIGURATION
    # =========================================================================
    # Set ONLY_VALIDATION to True if you only want to check your local Recall@5 score.
    # Set it to False when you are ready to generate a Kaggle submission file.
    ONLY_VALIDATION = False 
    
    EXPERIMENT_NAME = "experiment_13"
    DENSE_SCORES_TRAIN_PATH = "../outputs/cache/ensemble_dense_scores/qwen3-embedding-8b_w0.6_bge-m3_w0.4_dense_scores_train.json"
    DENSE_SCORES_TEST_PATH = "../outputs/cache/ensemble_dense_scores/qwen3-embedding-8b_w0.6_bge-m3_w0.4_dense_scores_test.json"
    
    # Core model configurations matching our high-performance architecture
    CONFIG_TEMPLATE = {
        "retriever": "llm", # Options: bm25, dense, llm
        "dense_model_name": "Qwen/Qwen3-Embedding-8B", 
        "llm_model_name": "Qwen/Qwen2.5-32B-Instruct",
        "top_k": 5,
        "max_candidates": 12,  # Window size for Stage-1 filtering
        "max_chars": None     # None = Un-truncated rich visual context
    }
    # =========================================================================

    root_dir = Path(__file__).resolve().parents[1]
    
    print("=== 🚀 STARTING RUN ===")
    print(f"Project Root Directory Verified At: {root_dir}")
    print(f"Mode Matrix: ONLY_VALIDATION = {ONLY_VALIDATION}")
    
    # --- STEP 1: LOCAL VALIDATION RUN ---
    print("\n==================================================")
    print("[Step 1] Starting Local Validation Process...")
    print("==================================================")
    
    # Deep-copy and inject validation-specific score paths
    val_config = CONFIG_TEMPLATE.copy()
    val_config["dense_scores_path"] = DENSE_SCORES_TRAIN_PATH
    
    try:
        pm.execute_notebook(
            input_path=str(root_dir / "notebooks/04_reranker.ipynb"),
            output_path = str(root_dir/"outputs"/"papermill_notebooks"/f"validation_{EXPERIMENT_NAME}.ipynb"),
            parameters={"CONFIG": val_config},
            cwd=str(root_dir / "notebooks"),
            kernel_name="mlbio_gpu",
            log_output=True
        )
        print("✅ Step 1: Hybrid Local Validation Completed Successfully!")
    except Exception as e:
        print(f"❌ Error during Validation execution loop: {e}")
        sys.exit(1)
        
    # --- STEP 2: KAGGLE SUBMISSION RUN ---
    if ONLY_VALIDATION:
        print("\n⏭️ ONLY_VALIDATION is True. Skipping Kaggle submission inference pass.")
        print("=== 🏁 INTERNAL VALIDATION COMPLETED! ===")
        return

    print("\n==================================================")
    print("[Step 2] Starting Kaggle Submission Generation...")
    print("==================================================")
    
    # Deep-copy and inject submission specific score paths
    sub_config = CONFIG_TEMPLATE.copy()
    sub_config["dense_scores_path"] = DENSE_SCORES_TEST_PATH
    
    try:
        pm.execute_notebook(
            input_path=str(root_dir / "notebooks/06_generate_submission.ipynb"),
            output_path = str(root_dir/"outputs"/"papermill_notebooks"/f"submission_{EXPERIMENT_NAME}.ipynb"),
            parameters={"CONFIG": sub_config},
            cwd=str(root_dir / "notebooks"),
            kernel_name="mlbio_gpu",
            log_output=True
        )
        print("✅ Step 2: Hybrid Submission CSV Exported Successfully!")
    except Exception as e:
        print(f"❌ Error during Kaggle submission generation loop: {e}")
        sys.exit(1)
        
    print("\n=== 🏁 INTERNAL VALIDATION AND KAGGLE SUBMISSION COMPLETED! ===")

if __name__ == "__main__":
    main()