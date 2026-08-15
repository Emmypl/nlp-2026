import torch
import json
from sentence_transformers import CrossEncoder
from typing import List, Tuple
from src.data.loader import Sample
from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever

class CrossEncoderRetriever(BaseRetriever):
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-v2-m3", 
        load_in_4bit: bool = False, 
        device_map: str = "auto", 
        disable_filtering: bool = False, 
        dense_scores_path: str = None, 
        max_candidates: int = 12, 
        max_chars: int = None
    ):
        self.model_name = model_name
        self.disable_filtering = disable_filtering
        self.max_candidates = max_candidates
        self.max_chars = max_chars
        self.dense_scores = None
        
        if dense_scores_path:
            print(f"Loading pre-computed dense scores from {dense_scores_path}...")
            with open(dense_scores_path, "r") as f:
                self.dense_scores = json.load(f)
                
        print(f"Initializing CrossEncoder Retriever with model '{model_name}' (4-bit={load_in_4bit}, disable_filtering={disable_filtering}, max_candidates={max_candidates}, max_chars={max_chars}, hybrid_mode={bool(dense_scores_path)})...")
        
        model_kwargs = {
            "device_map": device_map, 
            "torch_dtype": torch.bfloat16,
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            
        # Using SentenceTransformers' CrossEncoder automatically maps CausalLMs (like Qwen) 
        # and standard cross-encoders (like BGE) to their correct internal inference logic.
        self.model = CrossEncoder(
            model_name,
            model_kwargs=model_kwargs
        )
        
    def retrieve(self, sample: Sample, top_k: int = 5) -> List[Tuple[str, float]]:
        # Get candidates similar to LLMRetriever
        valid_ids = set(tq.quote_id for tq in sample.text_quotes) | set(iq.quote_id for iq in sample.img_quotes)
        if not valid_ids:
            return []
            
        if self.dense_scores and str(sample.q_id) in self.dense_scores:
            cached_list = self.dense_scores[str(sample.q_id)]
            ordered_ids = [qid for qid in cached_list if qid in valid_ids]
            
            if len(ordered_ids) > self.max_candidates and not self.disable_filtering:
                ordered_ids = ordered_ids[:self.max_candidates]
        else:
            ordered_ids = sorted(list(valid_ids))
            MAX_CANDIDATES = 25
            if len(ordered_ids) > MAX_CANDIDATES and not self.disable_filtering:
                bm25_results = BM25Retriever().retrieve(sample, top_k=MAX_CANDIDATES)
                ordered_ids = [r[0] for r in bm25_results]
                
        candidates_map = {}
        for tq in sample.text_quotes:
            candidates_map[tq.quote_id] = tq.text
        for iq in sample.img_quotes:
            candidates_map[iq.quote_id] = iq.img_description if iq.img_description else "[No image description available]"
            
        # Formulate query-candidate pairs
        pairs = []
        for qid in ordered_ids:
            if qid in candidates_map:
                content = candidates_map[qid]
                if self.max_chars and len(content) > self.max_chars:
                    processed_text = content[:self.max_chars]
                else:
                    processed_text = content
                pairs.append((sample.question, processed_text))
        
        if not pairs:
            return []
            
        # Run inference using SentenceTransformers interface
        scores = self.model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        
        # Match scores back to candidate IDs
        qid_score_pairs = []
        for idx, qid in enumerate(ordered_ids):
            if qid in candidates_map:
                qid_score_pairs.append((qid, float(scores[len(qid_score_pairs)])))
                
        # Sort by score descending
        qid_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Flush CUDA cache
        torch.cuda.empty_cache()
        
        return qid_score_pairs[:top_k]
