import os
import json
import subprocess
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
NOTEBOOK_PATH = "01_ml_classifier.ipynb"
CACHE_DIR = "../output/cache"
ARTIFACTS_DIR = "../artifacts"
CSV_PATH = os.path.join(ARTIFACTS_DIR, "experiment_summary.csv")

# Thread lock for CSV file writing
csv_lock = threading.Lock()

# ANCHOR Feature configurations to verify
FEATURE_FILES = {
    "tfidf_max5000_ng1-1": [
        os.path.join(CACHE_DIR, "tfidf_max5000_ng1-1_features.npz"),
        os.path.join(CACHE_DIR, "tfidf_max5000_ng1-1_vectorizer.joblib")
    ],
    "svd_comp500_tfidf5000_ng1-1": [
        os.path.join(CACHE_DIR, "svd_comp500_tfidf5000_ng1-1_features.npy"),
        os.path.join(CACHE_DIR, "svd_comp500_tfidf5000_ng1-1_model.joblib")
    ],
    "dense_all-MiniLM-L6-v2": [
        os.path.join(CACHE_DIR, "dense_all-MiniLM-L6-v2_features.npy")
    ],
    "meta_len1_code1_math1": [
        os.path.join(CACHE_DIR, "meta_len1_code1_math1_features.npy")
    ],
    "w2v_size100_win5": [
        os.path.join(CACHE_DIR, "w2v_size100_win5_features.npy"),
        os.path.join(CACHE_DIR, "w2v_size100_win5_model.joblib")
    ],
    # NONE BASELINE EXPERIMENTS ARE BELOW: 
	"dense_Qwen-Qwen3-Embedding-8B": [
	    os.path.join(CACHE_DIR, "dense_Qwen-Qwen3-Embedding-8B_features.npy")
    ],
	"w2v_size500_win10": [
    	os.path.join(CACHE_DIR, "w2v_size500_win10_features.npy"),
    	os.path.join(CACHE_DIR, "w2v_size500_win10_model.joblib")
    ],
    "tfidf_max30000_ng1-3": [
        os.path.join(CACHE_DIR, "tfidf_max30000_ng1-3_features.npz"),
        os.path.join(CACHE_DIR, "tfidf_max30000_ng1-3_vectorizer.joblib")
    ],
    "svd_comp1000_tfidf30000_ng1-3": [
        os.path.join(CACHE_DIR, "svd_comp1000_tfidf30000_ng1-3_features.npy"),
        os.path.join(CACHE_DIR, "svd_comp1000_tfidf30000_ng1-3_model.joblib")
    ],
}

# ANCHOR Generation instructions to verify
GENERATION_INSTRUCTIONS = {
    "tfidf_max5000_ng1-1": "extract_tfidf(train, max_features=5000, ngram_range=(1,1))",
    "svd_comp500_tfidf5000_ng1-1": "extract_svd(train, n_components=500, tfidf_max_features=5000, tfidf_ngram=(1,1))",
    "dense_all-MiniLM-L6-v2": "extract_dense_embeddings(train, model_name='all-MiniLM-L6-v2')",
    "meta_len1_code1_math1": "extract_meta_features(train)",
    "w2v_size100_win5": "extract_word2vec(train)",
    # NONE BASELINE EXPERIMENTS ARE BELOW: 
    "dense_Qwen-Qwen3-Embedding-8B": "Run the appropriate extraction cell in 00_feature_engineering.ipynb",
    "w2v_size500_win10": "extract_word2vec(train, vector_size=500, window=10)",
    "tfidf_max30000_ng1-3": "extract_tfidf(train, max_features=30000, ngram_range=(1,3))",
    "svd_comp1000_tfidf30000_ng1-3": "extract_svd(train, n_components=1000, tfidf_max_features=30000, tfidf_ngram=(1,3))",
}

def check_cached_features():
    missing = []
    for config, paths in FEATURE_FILES.items():
        for p in paths:
            if not os.path.exists(p):
                missing.append((config, p))
                break
    return missing

# 1. Check features
missing_features = check_cached_features()
if missing_features:
    print("[-] Error: Missing cached features. Please generate them in 00_feature_engineering.ipynb:")
    for config, path in missing_features:
        print(f"  * Config '{config}' is missing. Run this command in 00_feature_engineering.ipynb:")
        print(f"    -> {GENERATION_INSTRUCTIONS[config]}")
    exit(1)
else:
    print("[+] All feature engineering configurations are cached and ready.")

# Experiment Combinations
feature_configs = [
    # "tfidf_max5000_ng1-1",
    # "svd_comp500_tfidf5000_ng1-1",
    # "dense_all-MiniLM-L6-v2",
    # "meta_len1_code1_math1",
    # "w2v_size100_win5",
    "dense_Qwen-Qwen3-Embedding-8B",
    "w2v_size300_win10",
    "svd_comp1000_tfidf30000_ng1-3",
    "tfidf_max30000_ng1-3"  
]

model_types = ["lightgbm", "xgboost", "random_forest", "ridge"]
routing_types = ["classification", "regression", "cost_sensitive"]

# Generate all tasks
tasks = []
for feat in feature_configs:
    for model in model_types:
        for routing in routing_types:
            # Skip invalid Ridge options (Ridge only works for Regression)
            if model == "ridge" and routing != "regression":
                continue
            tasks.append((feat, model, routing))

def run_single_experiment(feat, model, routing):
    output_nb = f"executed_{feat}_{model}_{routing}.ipynb"
    
    cmd = [
        "papermill",
        NOTEBOOK_PATH,
        output_nb,
        "-p", "FEATURE_CONFIG", feat,
        "-p", "ML_MODEL_TYPE", model,
        "-p", "ROUTING_ML_MODEL_TYPE", routing,
        "-p", "RUNNING_IN_PIPELINE", "True",
        "-k", "mlbio"
    ]
    
    print(f"[+] Started: Features={feat}, Model={model}, Routing={routing}")
    
    try:
        # Run notebook execution
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check and process the temp JSON results
        temp_json_path = os.path.join(ARTIFACTS_DIR, f"results_{feat}_{model}_{routing}.json")
        if os.path.exists(temp_json_path):
            with open(temp_json_path, 'r') as f:
                results = json.load(f)
            
            # Fill details
            results['Experiment_ID'] = model
            results['Method_Category'] = routing
            results['Features'] = feat
            
            # Thread-safe writing to CSV
            with csv_lock:
                if os.path.exists(CSV_PATH):
                    df = pd.read_csv(CSV_PATH)
                else:
                    df = pd.DataFrame(columns=[
                        'Experiment_ID', 'Method_Category', 'Features', 'K-Fold', 
                        'CV_Reward_0.85', 'CV_Avg_Performance', 'CV_Avg_Cost', 
                        'Model_Distribution', 'Public_Kaggle_Score'
                    ])
                
                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
                df.to_csv(CSV_PATH, index=False)
                
            # Clean up JSON
            os.remove(temp_json_path)
            print(f"[+] Finished & Logged: Features={feat}, Model={model}, Routing={routing}")
        else:
            print(f"[-] Warning: Run succeeded but no results JSON found for {feat}_{model}_{routing}")
            
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed: Features={feat}, Model={model}, Routing={routing} (Error code: {e.returncode})")
    except Exception as e:
        print(f"[-] Unexpected error during {feat}_{model}_{routing}: {e}")
    finally:
        # Clean up output notebook
        if os.path.exists(output_nb):
            os.remove(output_nb)

# 24 core machine -> run 5 parallel experiments.
# Each experiment spawns tree training which uses 4 cores (n_jobs=4).
# Total cores used: 5 * 4 = 20 (perfect fit for 24 cores without overloading).
MAX_WORKERS = 5

print(f"[+] Starting orchestration pipeline with {MAX_WORKERS} parallel workers for {len(tasks)} tasks...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(run_single_experiment, feat, model, routing) for feat, model, routing in tasks]
    for future in as_completed(futures):
        pass # just iterate to capture completion

print("[+] All experiments are completed.")
