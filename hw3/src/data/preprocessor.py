from src.data.loader import Sample

def get_candidate_texts(sample: Sample) -> list[tuple[str, str]]:
    """Returns a list of (quote_id, text) tuples from the sample."""
    candidates = []
    for tq in sample.text_quotes:
        candidates.append((tq.quote_id, tq.text))
    for iq in sample.img_quotes:
        desc = iq.img_description if iq.img_description else ""
        candidates.append((iq.quote_id, desc))
    return candidates
