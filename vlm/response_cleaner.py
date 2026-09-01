"""
GeoChat / VLM response cleaner.

Strips tokenizer artifacts, malformed bounding-box markup, special tokens,
HTML fragments, and normalizes whitespace to produce clean human-readable
English without altering factual content.
"""

from __future__ import annotations

import re


# Compiled patterns (order matters — run broader patterns first)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 1. Bounding-box coordinate blocks: { < 4 8 >< 5 3 > | < 9 0 > }
    (re.compile(r'\{\s*(?:<\s*[\d\s]+\s*>\s*)+(?:\|\s*(?:<\s*[\d\s]+\s*>\s*)+)?\s*\}'), ''),
    # 2. Delimiter tokens: <delim>, <del im>, < del im >, ▁delim
    (re.compile(r'<\s*del\s*im\s*>', re.IGNORECASE), ''),
    (re.compile(r'<\s*delim\s*>', re.IGNORECASE), ''),
    (re.compile(r'▁delim', re.IGNORECASE), ''),
    # 3. HTML-like tags from tokenizer: <p>, </p>, <s>, </s>, <unk>, <image>, <pad>, <br>, etc.
    (re.compile(r'<\s*/?\s*(?:p|s|unk|image|pad|br|div|span|del|delim)\s*>', re.IGNORECASE), ''),
    # 4. Standalone angle-bracket numbers: < 4 8 > or <48>
    (re.compile(r'<\s*[\d\s]+\s*>'), ''),
    # 5. Sentencepiece underscore artifacts: ▁
    (re.compile(r'▁'), ' '),
    # 6. Any leftover braces/brackets containing only digits, spaces, pipes
    (re.compile(r'\{\s*[\d\s|]*\s*\}'), ''),
    (re.compile(r'\[\s*[\d\s|]*\s*\]'), ''),
    (re.compile(r'\(\s*[\d\s|]*\s*\)'), ''),
    # 7. Repeated whitespace
    (re.compile(r'[ \t]+'), ' '),
    # 8. Space before punctuation: " ," → ","
    (re.compile(r'\s+([,.:;?!])'), r'\1'),
    # 9. Multiple consecutive periods/commas (but preserve "...")
    (re.compile(r'\.{4,}'), '...'),
    (re.compile(r',{2,}'), ','),
    # 10. Leading punctuation at sentence start (after newline or start)
    (re.compile(r'(?:^|(?<=\n))\s*[,;]+\s*'), ''),
]


def clean_vlm_response(text: str) -> str:
    """
    Clean a raw VLM/GeoChat response by removing tokenizer artifacts
    and normalizing to readable English.

    This function:
    1. Removes known special tokens (<s>, </s>, <unk>, <image>, etc.)
    2. Removes malformed bounding-box tokenizer markup ({ < 48 >... })
    3. Removes delimiter tokens (<delim>, ▁delim)
    4. Normalizes Unicode whitespace (▁ → space)
    5. Collapses repeated whitespace
    6. Repairs obvious punctuation spacing
    7. Removes empty braces/brackets created by token removal
    8. Strips leading/trailing artifacts

    Does NOT invent facts or add claims not present in the original text.
    """
    if not text:
        return text

    result = text

    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)

    # Final strip and whitespace normalization
    result = result.strip()

    # Remove trailing isolated punctuation
    result = re.sub(r'\s+$', '', result)

    # If cleaning removed everything, return a placeholder
    if not result.strip():
        return "The model returned a response that could not be parsed."

    return result
