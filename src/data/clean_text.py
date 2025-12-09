# Importing Regular Expressions (re), html, emoji, and string for string manipulations

import re
import html
import string
from typing import Optional

class TextCleaner:
    """
    Advanced text cleaner for climate fallacy detection.
    Preserves semantic content while removing noise.
    Does NOT over-normalize, because subtle cues matter.
    """

    def __init__(self,
                 remove_urls=True,
                 remove_emails=True,
                 remove_hashtags=False,
                 remove_mentions=False,
                 normalize_whitespace=True,
                 lowercase=False):
        
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_hashtags = remove_hashtags
        self.remove_mentions = remove_mentions
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase

    def clean(self, text: Optional[str]) -> str:
        """Cleans a single text string."""
        if text is None or not isinstance(text, str):
            return ""

        # ---- Unescape HTML ----
        text = html.unescape(text)

        # ---- Remove URLs ----
        if self.remove_urls:
            text = re.sub(r"http\S+|www\.\S+", "", text)

        # ---- Remove emails ----
        if self.remove_emails:
            text = re.sub(r"\S+@\S+", "", text)

        # ---- Optional: remove hashtags ----
        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)

        # -- remove mentions @username ----
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # ---- Remove stray control characters ----
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

        # ---- Soft-clean punctuation but keep structure ----
        allowed = string.ascii_letters + string.digits + string.punctuation + " "
        text = "".join(ch if ch in allowed else " " for ch in text)

        # ---- Normalize whitespace ----
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        # ---- Lowercase if needed ----
        if self.lowercase:
            text = text.lower()

        return text


# Quick helper version for notebooks:
def basic_clean(text: Optional[str]) -> str:
    """Default cleaner for quick use."""
    return TextCleaner(
        remove_urls=True,
        remove_emails=True,
        remove_hashtags=False,     
        remove_mentions=False,     
        normalize_whitespace=True,
        lowercase=False            
    ).clean(text)
