"""
Layer 6 — Sentence Segmentation

Splits normalized Amharic text into individual sentences.

Sentence boundary markers handled:
  ።   (U+1362) Ethiopic full stop — the standard modern marker
  ፡፡  (two U+1361) Ethiopic wordspace doubled — historical/alternate
       full stop pattern, normalized to ። before splitting
  ?   (U+003F) question mark
  !   (U+0021) exclamation mark

This layer assumes Layer 3 (normalize) has already run, meaning:
  - Latin periods that should be sentence boundaries have already
    been converted to ።
  - Punctuation is already space-padded

This layer does NOT do punctuation space-padding itself — that is
Layer 3's responsibility. This module focuses purely on splitting.
"""

import re

# ── Boundary normalization ─────────────────────────────────────────────────

# ፡፡ (double Ethiopic wordspace) used as a sentence-final marker
# must be normalized to the standard ። before splitting, so that
# downstream consumers only ever see one full-stop convention.
_DOUBLE_WORDSPACE_PATTERN = re.compile(r"፡፡")


def _normalize_sentence_boundaries(text: str) -> str:
    """
    Convert ፡፡ (double Ethiopic wordspace used as full stop)
    to the standard ።.

    Note: a single ፡ is NOT touched here — per Phase 0 policy,
    single ፡ is a clarification mark, not a sentence boundary.
    Only the doubled ፡፡ pattern is treated as equivalent to ።.
    """
    return _DOUBLE_WORDSPACE_PATTERN.sub("።", text)


# ── Segmentation ────────────────────────────────────────────────────────────

# Sentence-final punctuation: ። ? !
# We split AFTER any one-or-more run of these characters,
# so "ምን ሆነ?!" stays together as one boundary, not two.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"([።?!]+)")


def segment(text: str) -> list[str]:
    """
    Split a block of text into a list of sentences.

    Args:
        text: Normalized text, potentially containing multiple sentences.
              May contain ። ፡፡ ? ! as sentence-final markers.

    Returns:
        List of sentences, each retaining its terminal punctuation,
        with leading/trailing whitespace stripped. Empty results
        (e.g. from trailing whitespace after the last sentence) are
        dropped.

    Examples:
        >>> segment("ሰላም ነው። እንዴት ነህ?")
        ['ሰላም ነው።', 'እንዴት ነህ?']
        >>> segment("ናት፡፡ ሄደ፡፡")
        ['ናት።', 'ሄደ።']
        >>> segment("ምን ሆነ?!")
        ['ምን ሆነ?!']
        >>> segment("")
        []
        >>> segment("ምንም ምልክት የለውም")
        ['ምንም ምልክት የለውም']
    """
    if not text or not text.strip():
        return []

    text = _normalize_sentence_boundaries(text)

    # Split, keeping the delimiter attached to the preceding sentence.
    # re.split with a capturing group returns:
    #   [text1, delim1, text2, delim2, ..., textN]
    # We recombine each text+delim pair.
    parts = _SENTENCE_BOUNDARY_PATTERN.split(text)

    sentences: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        sentence = (chunk + delim).strip()
        if sentence:
            sentences.append(sentence)
        i += 2

    return sentences


def segment_lines(lines: list[str]) -> list[str]:
    """
    Segment multiple input lines (e.g. paragraphs) into a flat list
    of individual sentences.

    Args:
        lines: List of text blocks, each potentially containing
               multiple sentences.

    Returns:
        Flat list of sentences across all input lines, in order.
    """
    result: list[str] = []
    for line in lines:
        result.extend(segment(line))
    return result


def segment_file(input_path: str, output_path: str) -> int:
    """
    Segment an entire text file into one-sentence-per-line output.

    Args:
        input_path:  Path to input text file (UTF-8), one paragraph
                     or document per line.
        output_path: Path to write segmented output, one sentence
                     per line.

    Returns:
        Number of sentences written.

    Raises:
        FileNotFoundError: If input_path does not exist.
        UnicodeDecodeError: If file is not valid UTF-8.
    """
    count = 0
    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            for sentence in segment(line):
                outfile.write(sentence + "\n")
                count += 1
    return count
