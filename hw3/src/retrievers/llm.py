import torch
import re
import ctypes
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Tuple
from src.data.loader import Sample
from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever

# 🔥 DYNAMIC SYSTEM HOTFIX: Force-inject the absolute path to the CUDA 13.0 linker library 
# directly into global process memory space. This bypasses all fragile LD_LIBRARY_PATH 
# environment issues and guarantees bitsandbytes and 4-bit quantization load flawlessly!
try:
    ctypes.CDLL("/home/emmy/miniconda3/envs/mlbio_gpu/lib/python3.10/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")
except Exception:
    pass


class LLMRetriever(BaseRetriever):
    def __init__(
        self, 
        model_name: str = "google/gemma-2-9b-it", 
        load_in_4bit: bool = True, 
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
                
        print(f"Initializing LLM Retriever with model '{model_name}' (4-bit={load_in_4bit}, disable_filtering={disable_filtering}, max_candidates={max_candidates}, max_chars={max_chars}, hybrid_mode={bool(dense_scores_path)})...")
        
        # Set up precise model loading configurations
        model_kwargs = {
            "device_map": device_map, 
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa"
        }
        
        # Dynamic quantization based on flag
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure tokenizer has a pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        
        # Build the generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
    def _format_prompt(self, sample: Sample, ordered_ids: List[str]) -> str:
        # Create map to lookup candidate content instantly while maintaining target sorting order
        candidates_map = {}
        for tq in sample.text_quotes:
            candidates_map[tq.quote_id] = tq.text
        for iq in sample.img_quotes:
            candidates_map[iq.quote_id] = iq.img_description if iq.img_description else "[No image description available]"
            
        candidates = []
        for qid in ordered_ids:
            if qid in candidates_map:
                content = candidates_map[qid]
                
                # Dynamic text handling: Truncate only if max_chars restriction is explicitly passed
                if self.max_chars and len(content) > self.max_chars:
                    processed_text = content[:self.max_chars] + "..."
                else:
                    processed_text = content
                    
                candidates.append(f"- [{qid}]: {processed_text}")
            
        candidates_str = "\n".join(candidates)
        
        system_msg = (
            "You are an expert information retrieval assistant. Given a question and a set of numbered candidates, "
            "identify exactly the 5 most relevant candidates containing supporting evidence to answer the question."
        )
        
        user_msg = (
            f"Question: {sample.question}\n\n"
            f"Candidates:\n{candidates_str}\n\n"
            "Instructions: Output exactly the 5 supporting quote IDs, ordered from most relevant to least relevant. "
            "Output ONLY space-separated IDs (e.g., 'text3 image2 text5 text1 image8'). Do NOT output prefixes, greetings, conversational filler, or explanations."
        )
        
        # Use standard formatting for prompt construction
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for models that don't support system roles in templates
            fallback_msg = [{"role": "user", "content": f"{system_msg}\n\n{user_msg}"}]
            return self.tokenizer.apply_chat_template(fallback_msg, tokenize=False, add_generation_prompt=True)
        
    def retrieve(self, sample: Sample, top_k: int = 5) -> List[Tuple[str, float]]:
        # Construct the list of actual valid candidate IDs for verification and parsing
        valid_ids = set(tq.quote_id for tq in sample.text_quotes) | set(iq.quote_id for iq in sample.img_quotes)
        if not valid_ids:
            return []
            
        # --- HYBRID RETRIEVAL STAGE ---
        if self.dense_scores and str(sample.q_id) in self.dense_scores:
            # Path 2: Sort candidates by pre-computed Dense Retriever ranking
            cached_list = self.dense_scores[str(sample.q_id)]
            ordered_ids = [qid for qid in cached_list if qid in valid_ids]
            
            # Path 3: Selective Pre-Filtering (Eliminate noise, keep Top self.max_candidates)
            if len(ordered_ids) > self.max_candidates and not self.disable_filtering:
                ordered_ids = ordered_ids[:self.max_candidates]
        else:
            # Legacy/Fallback flow if no dense scores are available
            ordered_ids = sorted(list(valid_ids))
            MAX_CANDIDATES = 25
            if len(ordered_ids) > MAX_CANDIDATES and not self.disable_filtering:
                bm25_results = BM25Retriever().retrieve(sample, top_k=MAX_CANDIDATES)
                ordered_ids = [r[0] for r in bm25_results]
            
        active_id_set = set(ordered_ids)
        prompt = self._format_prompt(sample, ordered_ids)
        
        # Restrict generating long texts to save compute and enforce formatting
        with torch.no_grad():
            outputs = self.pipe(
                prompt,
                max_new_tokens=40,
                do_sample=False,  # Greedy decoding for maximum deterministic reliability
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        generated_text = outputs[0]['generated_text']
        
        # Separate input prompt from newly generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        else:
            # Fallback in case pipeline behaves differently
            generated_text = generated_text.strip()
            
        # Extract quote IDs using robust Regex patterns
        predicted_tokens = re.findall(r'\b(?:text\d+|image\d+)\b', generated_text.lower())
        
        # Ensure IDs are valid (were presented in the prompt) and order is preserved
        matched_ids = []
        for pid in predicted_tokens:
            if pid in active_id_set and pid not in matched_ids:
                matched_ids.append(pid)
                
        # Handle catastrophic output failure (backfill if model failed to select enough valid ones)
        if len(matched_ids) < top_k:
            # Backfill using the exact Dense-ordered priority!
            remaining = [vid for vid in ordered_ids if vid not in matched_ids]
            matched_ids.extend(remaining[:top_k - len(matched_ids)])
            
        # Convert selected IDs into scores ordered by descending ranking placement
        results = [(quote_id, float(top_k - i)) for i, quote_id in enumerate(matched_ids)]
        
        # Flush VRAM cache to prevent memory fragmentation buildup during long test set loop
        torch.cuda.empty_cache()
        
        return results[:top_k]