import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
from src.data.loader import Sample
from src.retrievers.base import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None, load_in_4bit: bool = False):
        self.model_name = model_name.lower()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Dense Retriever Model '{model_name}' on device: {self.device} (4-bit={load_in_4bit})...")
        model_kwargs = {}
        if load_in_4bit and "cuda" in self.device:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = SentenceTransformer(model_name, device=self.device, model_kwargs=model_kwargs)
        self.model.eval()
        
    def retrieve(self, sample: Sample, top_k: int = 5) -> List[Tuple[str, float]]:
        candidates = []
        for tq in sample.text_quotes:
            candidates.append({"id": tq.quote_id, "text": tq.text})
        for iq in sample.img_quotes:
            desc = iq.img_description if iq.img_description else ""
            candidates.append({"id": iq.quote_id, "text": desc})

        if not candidates:
            return []

        candidate_texts = [c["text"] for c in candidates]
        
        # Add model-specific prompts/prefixes
        query_text = sample.question
        query_kwargs = {"convert_to_tensor": True, "show_progress_bar": False}
        
        if "bge-" in self.model_name:
            query_text = "Represent this sentence for searching relevant passages: " + query_text
        elif "qwen3-embedding" in self.model_name:
            # Qwen3-Embedding uses native sentence-transformers prompt_name configuration
            query_kwargs["prompt_name"] = "query"
        
        with torch.no_grad():
            query_embedding = self.model.encode(query_text, **query_kwargs)
            doc_embeddings = self.model.encode(
                candidate_texts, 
                batch_size=4, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
            
            # Compute cosine similarity
            cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            
        scores = cos_scores.cpu().tolist()
        
        # Flush PyTorch's reserved-but-unused VRAM cache to prevent fragmentation
        # buildup over hundreds of retrieve() calls in the submission loop.
        torch.cuda.empty_cache()
        
        results = [(c["id"], score) for c, score in zip(candidates, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
