from abc import ABC, abstractmethod
from typing import List, Tuple
from src.data.loader import Sample

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, sample: Sample, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return a list of (quote_id, score) sorted by relevance."""
        pass
