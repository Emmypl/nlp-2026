import json
from dataclasses import dataclass

@dataclass
class TextQuote:
    quote_id: str
    type: str
    text: str
    page_id: int
    layout_id: int

@dataclass
class ImageQuote:
    quote_id: str
    type: str
    img_path: str
    img_description: str
    page_id: int
    layout_id: int

@dataclass
class Sample:
    q_id: int
    doc_name: str
    domain: str
    question: str
    evidence_modality_type: list[str]
    question_type: str
    text_quotes: list[TextQuote]
    img_quotes: list[ImageQuote]
    gold_quotes: list[str] | None = None

def load_jsonl(filepath: str) -> list[Sample]:
    """
    Loads jsonl file into a list of Sample objects.
    Dynamically detects and injects VLM recaptions if found in cache.
    """
    import os
    from pathlib import Path
    
    # 1. Dynamic Zero-Config Detection of Rich VLM Captions
    vlm_captions = {}
    if os.environ.get("DISABLE_VLM", "0") == "0":
        vlm_filename = os.environ.get("VLM_CAPTIONS_FILE", "vlm_image_captions.json")
        possible_paths = [
            Path("outputs/cache") / vlm_filename,
            Path("../outputs/cache") / vlm_filename,
            Path("outputs/cache/vlm_image_captions.json"),
            Path("../outputs/cache/vlm_image_captions.json")
        ]
        for p in possible_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        vlm_captions = json.load(f)
                    print(f"ℹ️ Multi-Modal Upgrade: Auto-loaded {len(vlm_captions)} VLM captions from {p}")
                    break
                except Exception as e:
                    print(f"⚠️ Found VLM captions but failed to load: {e}")
                    
    # 2. Load Dataset Samples
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            
            t_quotes = [TextQuote(**t) for t in d.get("text_quotes", [])]
            
            # Parse image quotes and perform rich caption override
            i_quotes = []
            for i in d.get("img_quotes", []):
                iq = ImageQuote(**i)
                # Inject rich caption if present
                if iq.img_path in vlm_captions:
                    iq.img_description = vlm_captions[iq.img_path]
                i_quotes.append(iq)
            
            samples.append(Sample(
                q_id=d["q_id"],
                doc_name=d["doc_name"],
                domain=d["domain"],
                question=d["question"],
                evidence_modality_type=d["evidence_modality_type"],
                question_type=d.get("question_type", ""),
                text_quotes=t_quotes,
                img_quotes=i_quotes,
                gold_quotes=d.get("gold_quotes")
            ))
    return samples
