from rank_bm25 import BM25Okapi
from typing import List, Tuple
from src.data.loader import Sample
from src.retrievers.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def retrieve(self, sample: Sample, top_k: int = 5) -> List[Tuple[str, float]]:
        candidates = []
        for tq in sample.text_quotes:
            candidates.append({"id": tq.quote_id, "text": tq.text})
        for iq in sample.img_quotes:
            desc = iq.img_description if iq.img_description else ""
            candidates.append({"id": iq.quote_id, "text": desc})

        if not candidates:
            return []

        tokenized_corpus = [c["text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = sample.question.lower().split()
        scores = bm25.get_scores(tokenized_query)

        results = [(c["id"], score) for c, score in zip(candidates, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
