import re


def normalize_text(text: str) -> str:
    """
    Basic text normalization for augmentation and evaluation.

    This is intentionally simple and close in spirit to the paper's
    normalization step: lowercase and remove punctuation.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text
