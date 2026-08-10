"""
Layer 5 — Filter

Applies quality filters to segmented, classified sentences before they
enter the corpus. Runs after Layer 4 (script classification), so input
sentences may already contain <LATIN>...</LATIN> and <URL>...</URL>
tags.

Filters applied:
  1. Ethiopic content threshold: Ethiopic letters / total letters must
     be >= 0.90. Tagged <LATIN>/<URL> spans count as non-Ethiopic; the
     tags themselves are excluded (markup, not content). Digits and
     punctuation are script-neutral and excluded from both sides.
  2. Minimum sentence length: at least 5 Ethiopic word tokens.
  3. Maximum sentence length: at most 512 characters.
  4. Boilerplate removal: drop sentences matching known boilerplate
     patterns (copyright notices, navigation text, share prompts).
  5. Tigrinya detection: OFF by default, measured rather than dropped.
     See the Tigrinya section below.
  6. Deduplication: exact match and near-duplicate (differs only in
     whitespace/punctuation) are both treated as duplicates.
"""

import re

# ── Ethiopic content ratio ───────────────────────────────────────────────────

_ETHIOPIC_RANGE = (0x1200, 0x137F)

# Tags inserted by Layer 4 — must be stripped before computing the
# Ethiopic ratio so the tag markup itself doesn't skew the count.
_TAG_PATTERN = re.compile(r"</?(?:LATIN|URL)>")

# Matches the punctuation set defined in the Phase 0 specification.
# These are excluded from both numerator and denominator of the ratio
# so that punctuation-heavy sentences aren't penalized or rewarded.
_PUNCTUATION_CHARS = set("።፣፤፥፦፡?!#‹›")


def _is_ethiopic_char(ch: str) -> bool:
    return _ETHIOPIC_RANGE[0] <= ord(ch) <= _ETHIOPIC_RANGE[1]


def ethiopic_ratio(sentence: str) -> float:
    """
    Compute the proportion of Ethiopic letters among all letters.

    The question this filter exists to answer is "is this Amharic text,
    or is it text in another script?" Only letters can answer that.
    Digits, punctuation, and symbols are script-neutral and are excluded
    from both numerator and denominator.

    Counting them was previously discarding ordinary Amharic: Arabic
    numerals are Amharic by settled decision, so a sentence like
    "የ 161 ነዋሪዎች እዚህ ቋሚ በሆነ ሁኔታ ይኖራሉ ።" was penalized for content that
    is not foreign at all. On a 60k-line sample, moving to a letters-only
    denominator recovered 12,161 sentences and newly dropped none.

    <LATIN>...</LATIN> and <URL>...</URL> tag markers are stripped
    before counting, but the letters inside them still count as
    non-Ethiopic.

    Args:
        sentence: A classified sentence (Layer 4 output).

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 for a sentence with
        no letters at all.

    Examples:
        >>> ethiopic_ratio("ሰላም ነው")
        1.0
        >>> ethiopic_ratio("ሰላም <LATIN>Hello</LATIN>")  # 3 Ethiopic of 8 letters
        0.375
        >>> ethiopic_ratio("የ 161 ነዋሪዎች")
        1.0
    """
    stripped = _TAG_PATTERN.sub("", sentence)

    ethiopic_count = 0
    total_count = 0

    for ch in stripped:
        if not ch.isalpha():
            continue
        total_count += 1
        if _is_ethiopic_char(ch):
            ethiopic_count += 1

    if total_count == 0:
        return 0.0

    return ethiopic_count / total_count


def passes_ethiopic_threshold(sentence: str, threshold: float = 0.90) -> bool:
    """
    Check whether a sentence meets the minimum Ethiopic content ratio.

    Args:
        sentence: A classified sentence.
        threshold: Minimum required ratio (default 0.90 per spec).

    Returns:
        True if ethiopic_ratio(sentence) >= threshold.
    """
    return ethiopic_ratio(sentence) >= threshold


# ── Sentence length bounds ───────────────────────────────────────────────────


def _count_ethiopic_word_tokens(sentence: str) -> int:
    """
    Count the number of space-separated tokens that contain at least
    one Ethiopic character. <LATIN>/<URL> tagged tokens and pure
    punctuation tokens do not count.
    """
    stripped = _TAG_PATTERN.sub("", sentence)
    count = 0
    for token in stripped.split():
        if token in _PUNCTUATION_CHARS:
            continue
        if any(_is_ethiopic_char(ch) for ch in token):
            count += 1
    return count


def passes_min_length(sentence: str, min_tokens: int = 5) -> bool:
    """
    Check whether a sentence has at least min_tokens Ethiopic word tokens.

    Args:
        sentence: A classified sentence.
        min_tokens: Minimum required Ethiopic token count (default 5).

    Returns:
        True if the sentence meets the minimum length.
    """
    return _count_ethiopic_word_tokens(sentence) >= min_tokens


def passes_max_length(sentence: str, max_chars: int = 512) -> bool:
    """
    Check whether a sentence is at or below the maximum character length.

    Args:
        sentence: A classified sentence.
        max_chars: Maximum allowed character count (default 512).

    Returns:
        True if len(sentence) <= max_chars.
    """
    return len(sentence) <= max_chars


# ── Boilerplate detection ────────────────────────────────────────────────────

# Known boilerplate patterns commonly found in scraped news/web text.
# Matched case-insensitively against the raw sentence (tags included,
# since boilerplate often appears as plain Latin text wrapped by
# Layer 4, e.g. "<LATIN>Copyright</LATIN> 2024").
_BOILERPLATE_PATTERNS = [
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"copyright", re.IGNORECASE),
    re.compile(r"click here", re.IGNORECASE),
    re.compile(r"subscribe", re.IGNORECASE),
    re.compile(r"share this", re.IGNORECASE),
    re.compile(r"read more", re.IGNORECASE),
    re.compile(r"^\s*follow us\s*$", re.IGNORECASE),
]


def is_boilerplate(sentence: str) -> bool:
    """
    Check whether a sentence matches a known boilerplate pattern.

    Args:
        sentence: A classified sentence.

    Returns:
        True if the sentence matches any boilerplate pattern.
    """
    return any(pattern.search(sentence) for pattern in _BOILERPLATE_PATTERNS)


# ── Deduplication ─────────────────────────────────────────────────────────────

# For near-duplicate detection: strip all whitespace and the defined
# punctuation set, then compare. Two sentences that are identical once
# whitespace/punctuation is removed are treated as the same sentence.
_DEDUP_STRIP_PATTERN = re.compile(r"[\s።፣፤፥፦፡?!#‹›]+")


def _dedup_key(sentence: str) -> str:
    """Normalize a sentence to a comparison key for duplicate detection."""
    return _DEDUP_STRIP_PATTERN.sub("", sentence)


def deduplicate(sentences: list[str]) -> list[str]:
    """
    Remove exact and near-duplicate sentences, keeping the first
    occurrence of each.

    Args:
        sentences: List of classified sentences, in original order.

    Returns:
        List with duplicates removed, preserving original order of
        first occurrence.

    Examples:
        >>> deduplicate(["ሰላም ነው።", "ሰላም ነው።", "እንዴት ነህ።"])
        ['ሰላም ነው።', 'እንዴት ነህ።']
        >>> deduplicate(["ሰላም ነው።", "ሰላም  ነው!"])
        ['ሰላም ነው።']
    """
    seen: set[str] = set()
    result: list[str] = []
    for sentence in sentences:
        key = _dedup_key(sentence)
        if key in seen:
            continue
        seen.add(key)
        result.append(sentence)
    return result


# ── Combined filter pass ─────────────────────────────────────────────────────


# ── Tigrinya detection ──────────────────────────────────────────────────────
#
# The Ethiopic-content threshold cannot catch Tigrinya or Ge'ez: they use
# the same script, so they score 1.0 and pass cleanly. CC-100's language
# identification is imperfect and lets them through, which matters because
# they pollute the frequency table that the lexicon is bootstrapped from.
#
# Layer 0 normalization erases most orthographic signals — ኣ, ዓ and ሓ all
# fold into Amharic forms — so detection has to be lexical. Two signals
# survive normalization:
#
#   1. The ቐ series (U+1250-U+1256), which Amharic orthography does not use.
#      On its own it is unreliable: Amharic text mentions Tigray place names
#      like መቐሌ, so presence alone would discard valid Amharic.
#   2. Tigrinya function words, which are highly distinctive and rarely
#      co-occur in Amharic sentences.
#
# The rule requires either two function words, or one function word plus a
# ቐ-series character. On a 60k-line sample this flags ~0.4% of kept
# sentences. Recall is incomplete — it is a heuristic word list, not a
# language model — so this is a MEASUREMENT tool by default. Layer 7 counts
# what it finds; dropping is opt-in via filter_sentences(drop_tigrinya=True).
#
# The word list needs review by someone who reads Tigrinya before it is
# used to discard anything.

_TIGRINYA_ONLY_PATTERN = re.compile(r"[ቐ-ቖ]")

# Markers with no Amharic reading. One is enough.
_TIGRINYA_STRONG_WORDS = frozenset({
    "እዩ", "እያ", "እዮም", "ኢዩ", "ኢና", "ናይ", "ናብ", "ካብ", "እቲ", "እታ",
    "እዞም", "ዘሎ", "ዘለዉ", "ንሱ", "ንሳ", "ከምኡ", "ምዃኑ", "ኣይኮነን",
    "ዘይኮነ", "ኣለዎ",
})

# Markers that also have an Amharic reading, so one alone proves nothing.
#
# አብ is the clearest case: it is Tigrinya "in", but also ordinary Amharic
# "father" (እግዚአብሄር አብ). Layer 0 normalizes the Tigrinya spelling ኣብ into
# አብ, erasing the only orthographic difference — so on normalized text the
# two are indistinguishable. Counting it as a strong marker flagged 1,214
# sentences in a 400k sample, most of them Amharic scripture.
_TIGRINYA_WEAK_WORDS = frozenset({
    "አብ", "ኣብ", "ድማ", "እንተ", "ነዚ", "ይኹን", "ብዛዕባ", "ዝኾነ", "ክኸውን",
})

_TIGRINYA_FUNCTION_WORDS = _TIGRINYA_STRONG_WORDS | _TIGRINYA_WEAK_WORDS


def tigrinya_markers(sentence: str) -> int:
    """
    Count distinct Tigrinya function words in a sentence.

    Args:
        sentence: A classified sentence.

    Returns:
        Number of distinct markers found.

    Examples:
        >>> tigrinya_markers("እቲ ቤት ትምህርቲ ናይ ግምቢ እዩ")
        3
        >>> tigrinya_markers("ሰላም ነው እንዴት ነህ")
        0
    """
    return len(set(sentence.split()) & _TIGRINYA_FUNCTION_WORDS)


def is_likely_tigrinya(sentence: str) -> bool:
    """
    Heuristic test for Tigrinya text that the Ethiopic threshold misses.

    Fires on any of:
      - one strong marker (no Amharic reading)
      - two markers of any strength
      - one marker plus a ቐ-series character

    ቐ alone is not enough: Amharic text names Tigray places like መቐሌ,
    so presence of the letter says nothing on its own.

    Recall is incomplete — this is a word list, not a language model.
    Tigrinya sentences whose vocabulary happens to avoid these markers
    pass through.

    Args:
        sentence: A classified sentence.

    Returns:
        True if the sentence looks like Tigrinya.

    Examples:
        >>> is_likely_tigrinya("እቲ ቤት ትምህርቲ ናይ ግምቢ እዩ")
        True
        >>> is_likely_tigrinya("የመቐለ ከተማ የንጹህ መጠጥ ውሀ ችግር")
        False
        >>> is_likely_tigrinya("ሰውን ከእግዚአብሄር አብ ጋር አስታርቆ")
        False
    """
    tokens = set(sentence.split())
    strong = len(tokens & _TIGRINYA_STRONG_WORDS)
    if strong >= 1:
        return True

    markers = strong + len(tokens & _TIGRINYA_WEAK_WORDS)
    if markers >= 2:
        return True
    return markers >= 1 and bool(_TIGRINYA_ONLY_PATTERN.search(sentence))


# Drop reasons, in the order the filters are applied. Layer 7 reports
# counts per reason so that a run which discards most of the corpus can
# be diagnosed rather than just observed.
DROP_EMPTY = "empty"
DROP_BOILERPLATE = "boilerplate"
DROP_ETHIOPIC_THRESHOLD = "ethiopic_threshold"
DROP_TOO_SHORT = "too_short"
DROP_TOO_LONG = "too_long"
DROP_DUPLICATE = "duplicate"
DROP_TIGRINYA = "tigrinya"

DROP_REASONS = (
    DROP_EMPTY,
    DROP_BOILERPLATE,
    DROP_ETHIOPIC_THRESHOLD,
    DROP_TOO_SHORT,
    DROP_TOO_LONG,
    DROP_TIGRINYA,
    DROP_DUPLICATE,
)


def drop_reason(
    sentence: str,
    ethiopic_threshold: float = 0.90,
    min_tokens: int = 5,
    max_chars: int = 512,
    drop_tigrinya: bool = False,
) -> str | None:
    """
    Return why a sentence would be dropped, or None if it survives.

    Same checks and same order as passes_all_filters, but reporting
    which filter rejected the sentence instead of a bare boolean.
    Deduplication is not included — it needs the full sentence list.

    Args:
        sentence: A classified sentence.
        ethiopic_threshold: Minimum Ethiopic content ratio.
        min_tokens: Minimum Ethiopic word token count.
        max_chars: Maximum character length.
        drop_tigrinya: Also reject sentences that look like Tigrinya.
            Off by default — the detector is a heuristic word list, so
            the full run measures contamination rather than silently
            discarding text on its say-so.

    Returns:
        One of the DROP_* constants, or None if the sentence passes.

    Examples:
        >>> drop_reason("")
        'empty'
        >>> drop_reason("ሰላም ነው ።")
        'too_short'
    """
    if not sentence or not sentence.strip():
        return DROP_EMPTY
    if is_boilerplate(sentence):
        return DROP_BOILERPLATE
    if not passes_ethiopic_threshold(sentence, ethiopic_threshold):
        return DROP_ETHIOPIC_THRESHOLD
    if not passes_min_length(sentence, min_tokens):
        return DROP_TOO_SHORT
    if not passes_max_length(sentence, max_chars):
        return DROP_TOO_LONG
    if drop_tigrinya and is_likely_tigrinya(sentence):
        return DROP_TIGRINYA
    return None


def passes_all_filters(
    sentence: str,
    ethiopic_threshold: float = 0.90,
    min_tokens: int = 5,
    max_chars: int = 512,
    drop_tigrinya: bool = False,
) -> bool:
    """
    Run a single sentence through all non-deduplication filters.

    Deduplication is handled separately via deduplicate() because it
    requires the full list of sentences, not a single sentence.

    Args:
        sentence: A classified sentence.
        ethiopic_threshold: Minimum Ethiopic content ratio.
        min_tokens: Minimum Ethiopic word token count.
        max_chars: Maximum character length.

    Returns:
        True if the sentence passes every filter.
    """
    return (
        drop_reason(
            sentence, ethiopic_threshold, min_tokens, max_chars, drop_tigrinya
        )
        is None
    )


def filter_sentences(
    sentences: list[str],
    ethiopic_threshold: float = 0.90,
    min_tokens: int = 5,
    max_chars: int = 512,
    drop_tigrinya: bool = False,
) -> list[str]:
    """
    Apply the full Layer 5 filter pipeline to a list of sentences:
    per-sentence filters first, then deduplication.

    Args:
        sentences: List of classified sentences.
        ethiopic_threshold: Minimum Ethiopic content ratio.
        min_tokens: Minimum Ethiopic word token count.
        max_chars: Maximum character length.

    Returns:
        Filtered, deduplicated list of sentences.
    """
    survivors = [
        s
        for s in sentences
        if passes_all_filters(
            s, ethiopic_threshold, min_tokens, max_chars, drop_tigrinya
        )
    ]
    return deduplicate(survivors)
