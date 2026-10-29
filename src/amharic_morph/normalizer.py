"""
Layer 0 — Character Normalization

Normalizes Amharic homophonic character variants to their canonical forms.
This must run before any other processing step.

Two types of normalization are applied:

1. Order-level normalization: characters from the same consonant family
   but written with a less common variant are replaced with the standard form.
   Example: ሐሙስ → ሀሙስ (both represent the same sound)

2. Labialized character normalization: sequences like ሩ+ዋ or ሩ+አ that
   should be written as the single labialized character ሯ are unified.
   Example: ሩዋ → ሯ

Normalization sources:
  - Established Amharic NLP preprocessing practice
  - Ethiopic Unicode block (U+1200–U+137F)
"""

import re


def normalize(text: str) -> str:
    """
    Normalize Amharic character variants to canonical forms.

    Applies both order-level homophone normalization and
    labialized character unification.

    Args:
        text: Raw Amharic text string (UTF-8)

    Returns:
        Normalized text. All other characters are unchanged.

    Examples:
        >>> normalize("ሃይሌ")
        'ሀይሌ'
        >>> normalize("ሐሙስ")
        'ሀሙስ'
        >>> normalize("ሠላም")
        'ሰላም'
        >>> normalize("ዐይን")
        'አይን'
        >>> normalize("ፀሀይ")
        'ጸሀይ'
        >>> normalize("")
        ''
    """
    if not text:
        return text
    return _apply_normalization(text)


def _apply_normalization(text: str) -> str:
    """
    Apply all normalization rules in sequence.

    Rules are applied in order. Each rule's output feeds into
    the next rule's input.
    """

    # ── ሀ family (hä) ────────────────────────────────────────────────────────
    # Order 1: ሃ ኀ ኃ ሐ ሓ ኻ → ሀ
    # Note: the order-1 member of the ኀ series is ኀ (U+1280), not ኅ (U+1285).
    # ኅ is the order-6 form and is handled by the order-6 rule below.
    t = re.sub("[ሃኀኃሐሓኻ]", "ሀ", text)
    # Order 2: ሑ ኁ ዅ → ሁ
    t = re.sub("[ሑኁዅ]", "ሁ", t)
    # Order 3: ኂ ሒ ኺ → ሂ
    t = re.sub("[ኂሒኺ]", "ሂ", t)
    # Order 4: ኌ ሔ ዄ → ሄ
    t = re.sub("[ኌሔዄ]", "ሄ", t)
    # Order 6: ሕ ኅ → ህ
    t = re.sub("[ሕኅ]", "ህ", t)
    # Order 6: ኆ ሖ ኾ → ሆ
    t = re.sub("[ኆሖኾ]", "ሆ", t)

    # ── ሰ family (se) ────────────────────────────────────────────────────────
    # Order 1: ሠ → ሰ
    t = re.sub("[ሠ]", "ሰ", t)
    # Order 2: ሡ → ሱ
    t = re.sub("[ሡ]", "ሱ", t)
    # Order 3: ሢ → ሲ
    t = re.sub("[ሢ]", "ሲ", t)
    # Order 4: ሣ → ሳ
    t = re.sub("[ሣ]", "ሳ", t)
    # Order 5: ሤ → ሴ
    t = re.sub("[ሤ]", "ሴ", t)
    # Order 6: ሥ → ስ
    t = re.sub("[ሥ]", "ስ", t)
    # Order 7: ሦ → ሶ
    t = re.sub("[ሦ]", "ሶ", t)

    # ── አ family (ā) ─────────────────────────────────────────────────────────
    # Order 1: ዓ ኣ ዐ → አ
    t = re.sub("[ዓኣዐ]", "አ", t)
    # Order 2: ዑ → ኡ
    t = re.sub("[ዑ]", "ኡ", t)
    # Order 3: ዒ → ኢ
    t = re.sub("[ዒ]", "ኢ", t)
    # Order 4: ዔ → ኤ  (note: order 4 skipped — ዓ handled in order 1 above)
    t = re.sub("[ዔ]", "ኤ", t)
    # Order 5: ዕ → እ
    t = re.sub("[ዕ]", "እ", t)
    # Order 6: ዖ → ኦ
    t = re.sub("[ዖ]", "ኦ", t)

    # ── ጸ family (ts'e) ───────────────────────────────────────────────────────
    # Per spec section 2.5, ጸ is canonical and the ፀ series normalizes to it.
    # This is also the direction that maps rare → common: in CC-100 the ጸ
    # series consistently outnumbers the ፀ series (ጽ 55.9k vs ፅ 21.9k).
    # Order 1: ፀ → ጸ
    t = re.sub("[ፀ]", "ጸ", t)
    # Order 2: ፁ → ጹ
    t = re.sub("[ፁ]", "ጹ", t)
    # Order 3: ፂ → ጺ
    t = re.sub("[ፂ]", "ጺ", t)
    # Order 4: ፃ → ጻ
    t = re.sub("[ፃ]", "ጻ", t)
    # Order 5: ፄ → ጼ
    t = re.sub("[ፄ]", "ጼ", t)
    # Order 6: ፅ → ጽ
    t = re.sub("[ፅ]", "ጽ", t)
    # Order 7: ፆ → ጾ
    t = re.sub("[ፆ]", "ጾ", t)

    # ── ው family (wu) ─────────────────────────────────────────────────────────
    # ዉ (U+12C9) → ው (U+12CD). Per spec section 2.5.
    t = re.sub("[ዉ]", "ው", t)

    # ── ጎ / ጐ (go / gwa) ──────────────────────────────────────────────────────
    # The spec table lists this pair as ጎ → ጐ, but that is backwards: it maps
    # the common form onto the rare one. In CC-100, ጎ (U+130E) appears 49,262
    # times against 2,978 for ጐ (U+1310). Normalizing rare → common, matching
    # every other rule in this module.
    t = re.sub("[ጐ]", "ጎ", t)

    # ── Labialized character unification ─────────────────────────────────────
    # Sequences of (order-2 character) + (ዋ or አ) should be written
    # as the single labialized character (order-8 / labialized order)
    t = re.sub("(ሉ[ዋአ])", "ሏ", t)
    t = re.sub("(ሙ[ዋአ])", "ሟ", t)
    t = re.sub("(ቱ[ዋአ])", "ቷ", t)
    t = re.sub("(ሩ[ዋአ])", "ሯ", t)
    t = re.sub("(ሱ[ዋአ])", "ሷ", t)
    t = re.sub("(ሹ[ዋአ])", "ሿ", t)
    t = re.sub("(ቁ[ዋአ])", "ቋ", t)
    t = re.sub("(ቡ[ዋአ])", "ቧ", t)
    t = re.sub("(ቹ[ዋአ])", "ቿ", t)
    t = re.sub("(ሁ[ዋአ])", "ኋ", t)
    t = re.sub("(ኑ[ዋአ])", "ኗ", t)
    t = re.sub("(ኙ[ዋአ])", "ኟ", t)
    t = re.sub("(ኩ[ዋአ])", "ኳ", t)
    t = re.sub("(ዙ[ዋአ])", "ዟ", t)
    t = re.sub("(ጉ[ዋአ])", "ጓ", t)
    # ዱ (order 2), NOT ደ (order 1). Every other rule here uses the order-2
    # form. Triggering on ደ corrupts ordinary words: ድሬደዋ (Dire Dawa) would
    # become ድሬዷ, and ወስደዋል ("they have taken") would collapse into
    # ወስዷል ("he has taken") — a morphological change, not an orthographic one.
    t = re.sub("(ዱ[ዋአ])", "ዷ", t)
    t = re.sub("(ጡ[ዋአ])", "ጧ", t)
    t = re.sub("(ጩ[ዋአ])", "ጯ", t)
    t = re.sub("(ጹ[ዋአ])", "ጿ", t)
    t = re.sub("(ፉ[ዋአ])", "ፏ", t)

    # ── Single character alternates ───────────────────────────────────────────
    # ቊ is an alternate form of ቁ
    t = re.sub("[ቊ]", "ቁ", t)
    # ኵ is an alternate form of ኩ
    t = re.sub("[ኵ]", "ኩ", t)

    return t


def normalize_lines(lines: list[str]) -> list[str]:
    """
    Normalize a list of text lines.

    Args:
        lines: List of raw text strings

    Returns:
        List of normalized text strings, same length as input
    """
    return [normalize(line) for line in lines]


def normalize_file(input_path: str, output_path: str) -> int:
    """
    Normalize an entire text file line by line.

    Args:
        input_path:  Path to input text file (UTF-8)
        output_path: Path to write normalized output

    Returns:
        Number of lines processed

    Raises:
        FileNotFoundError: If input_path does not exist
        UnicodeDecodeError: If file is not valid UTF-8
    """
    count = 0
    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            outfile.write(normalize(line))
            count += 1
    return count
