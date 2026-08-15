import json
from typing import List, Tuple
from src.data.loader import Sample
from src.retrievers.base import BaseRetriever

class CachedRetriever(BaseRetriever):
    """
    A dummy retriever that reads straight from a pre-computed dense scores JSON cache.
    Useful for evaluating hybrid pipelines or fused scores without re-running models.
    """
    def __init__(self, cache_path: str):
        print(f"Loading cached dense scores from: {cache_path}...")
        with open(cache_path, "r") as f:
            self.cache = json.load(f)
            
    def retrieve(self, sample: Sample, top_k: int = 5) -> List[Tuple[str, float]]:
        # Find all valid quote IDs belonging to this specific query
        valid_ids = set(tq.quote_id for tq in sample.text_quotes) | set(iq.quote_id for iq in sample.img_quotes)
        
        # Get the sorted list of IDs from the cache for this query
        cached_list = self.cache.get(str(sample.q_id), [])
        
        # Filter down to candidates that actually belong to this query
        # (This is important because the cache might contain candidates from other queries if they were ranked globally)
        ordered_ids = [qid for qid in cached_list if qid in valid_ids]
        
        # Return mock scores (e.g., 5.0, 4.0, 3.0...) to satisfy the evaluation loop
        return [(qid, float(top_k - i)) for i, qid in enumerate(ordered_ids[:top_k])]
