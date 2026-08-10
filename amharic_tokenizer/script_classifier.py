"""
Layer 4 — Script Classification

Detects the script composition of each space-separated token and applies
the appropriate tagging or splitting rule. Runs after Layer 3 (normalize)
has already space-padded punctuation, so input tokens here are simple
space-separated units.

Rules (per Phase 0 specification, Section 4.3):

  ALL ETHIOPIC         -> keep as-is
  ALL LATIN            -> wrap: <LATIN>token</LATIN>
  ALL NUMERIC          -> keep as-is
  URL                  -> wrap: <URL>token</URL>
  MIXED ETHIOPIC+LATIN -> split at script boundary, wrap Latin spans
  MIXED NUMERIC+ETHIOPIC -> split at boundary, keep both spans as-is
  MIXED LATIN+NUMERIC  -> treat as single Latin unit (e.g. COVID-19)
  HASHTAG (#)          -> # kept separate, remainder classified normally

Design note: classification operates on a single space-separated token
at a time. Sentence-level orchestration (splitting a sentence into
tokens, running classify_token on each, rejoining) is handled by
classify_text().
"""

import re

from amharic_tokenizer.punctuation import is_abbreviation

# ── Character class detection ──────────────────────────────────────────────

_ETHIOPIC_RANGE = (0x1200, 0x137F)


def _is_ethiopic_char(ch: str) -> bool:
    return _ETHIOPIC_RANGE[0] <= ord(ch) <= _ETHIOPIC_RANGE[1]


def _is_latin_char(ch: str) -> bool:
    return ch.isalpha() and ch.isascii()


def _is_digit_char(ch: str) -> bool:
    return ch.isdigit() and ch.isascii()


# Characters that are allowed inside a Latin "word" span without breaking
# it apart — covers things like COVID-19, U.S.A, don't, etc.
# These are only consumed when adjacent to a Latin letter; a hyphen or
# period surrounded by Ethiopic is not affected because classification
# scans per maximal same-class span first, then merges allowed connectors.
_LATIN_CONNECTOR_CHARS = set("-.'_")


# ── URL detection ───────────────────────────────────────────────────────────

_URL_PATTERN = re.compile(
    r"^(https?://|www\.)\S+$|^\S+\.(com|org|net|et|io|co)(/\S*)?$",
    re.IGNORECASE,
)


def _is_url(token: str) -> bool:
    """
    Detect whether a token (with surrounding punctuation already
    space-padded away by Layer 3) looks like a URL.
    """
    return bool(_URL_PATTERN.match(token))


# ── Hashtag detection ───────────────────────────────────────────────────────


def _is_hashtag(token: str) -> bool:
    return token.startswith("#") and len(token) > 1


# ── Script span splitting ──────────────────────────────────────────────────


def _split_into_spans(token: str) -> list[tuple[str, str]]:
    """
    Split a token into maximal contiguous spans of the same script class.

    Returns a list of (span_text, span_type) tuples where span_type is
    one of: "ethiopic", "latin", "numeric", "other".

    Latin spans absorb connector characters (- . ' _) when they sit
    between two Latin-letter characters, so "COVID-19" stays
    together as far as the Latin/numeric merge step is concerned —
    that merge happens in classify_token(), not here. This function
    only separates Ethiopic from non-Ethiopic.
    """
    if not token:
        return []

    spans: list[tuple[str, str]] = []
    current_chars: list[str] = []
    current_type: str | None = None

    def char_type(ch: str) -> str:
        if _is_ethiopic_char(ch):
            return "ethiopic"
        if _is_latin_char(ch):
            return "latin"
        if _is_digit_char(ch):
            return "numeric"
        return "other"

    for ch in token:
        ch_type = char_type(ch)

        # Connector characters attach to whichever span they're inside;
        # they never start a new span on their own and never force a
        # split between latin/numeric (that merge is handled later).
        if (
            ch_type == "other"
            and ch in _LATIN_CONNECTOR_CHARS
            and current_type in ("latin", "numeric")
        ):
            current_chars.append(ch)
            continue

        if ch_type != current_type:
            if current_chars:
                spans.append(("".join(current_chars), current_type))
            current_chars = [ch]
            current_type = ch_type
        else:
            current_chars.append(ch)

    if current_chars:
        spans.append(("".join(current_chars), current_type))

    return spans


def _merge_latin_numeric_spans(spans: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Merge adjacent latin/numeric spans into a single latin span.
    This handles cases like "COVID" + "-19" -> "COVID-19" as one
    <LATIN> unit, per the Phase 0 rule that Latin+numeric combos
    are a single Latin unit.

    Pure numeric spans that are NOT adjacent to a latin span are
    left untouched (e.g. plain "2024" stays "numeric", not merged).
    """
    if not spans:
        return spans

    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(spans):
        text, span_type = spans[i]

        if span_type == "latin":
            # Absorb any immediately following numeric span(s),
            # and any further latin spans connected via connector chars
            # (already merged in _split_into_spans for the connector case,
            # but a digit run after a hyphen arrives as a separate
            # numeric span, so merge it here).
            combined = text
            j = i + 1
            while j < len(spans) and spans[j][1] in ("latin", "numeric"):
                combined += spans[j][0]
                j += 1
            merged.append((combined, "latin"))
            i = j
        else:
            merged.append((text, span_type))
            i += 1

    return merged


# ── Token-level classification ──────────────────────────────────────────────


def classify_token(token: str) -> str:
    """
    Classify and tag a single space-separated token.

    Args:
        token: A single token (no internal whitespace), already
               isolated by Layer 3 punctuation space-padding.

    Returns:
        The token transformed according to script classification rules.
        May contain internal spaces if a script-boundary split occurred
        (e.g. "የEthiopia" -> "የ <LATIN>Ethiopia</LATIN>").

    Examples:
        >>> classify_token("ሰላም")
        'ሰላም'
        >>> classify_token("Hello")
        '<LATIN>Hello</LATIN>'
        >>> classify_token("2024")
        '2024'
        >>> classify_token("የEthiopia")
        'የ <LATIN>Ethiopia</LATIN>'
        >>> classify_token("100ኛ")
        '100 ኛ'
        >>> classify_token("COVID-19")
        '<LATIN>COVID-19</LATIN>'
        >>> classify_token("#ኢትዮጵያ")
        '# ኢትዮጵያ'
        >>> classify_token("#Ethiopia")
        '# <LATIN>Ethiopia</LATIN>'
        >>> classify_token("https://example.com")
        '<URL>https://example.com</URL>'
        >>> classify_token("")
        ''
    """
    if not token:
        return token

    # URL check first — URLs contain characters (/, :, .) that would
    # otherwise be misclassified by span splitting.
    if _is_url(token):
        return f"<URL>{token}</URL>"

    # Amharic abbreviations (ዓ.ም, እ.ኤ.አ, ኃ.የተ.የግ.ማ) have the same problem:
    # the internal periods are script boundaries, so span splitting would
    # shred ዓ.ም into "ዓ . ም". They are atomic tokens and stay whole.
    if is_abbreviation(token):
        return token

    # Hashtag: split off the # and classify the remainder independently.
    if _is_hashtag(token):
        remainder = token[1:]
        classified_remainder = classify_token(remainder)
        return f"# {classified_remainder}"

    spans = _split_into_spans(token)
    spans = _merge_latin_numeric_spans(spans)

    parts: list[str] = []
    for text, span_type in spans:
        if span_type == "latin":
            parts.append(f"<LATIN>{text}</LATIN>")
        else:
            # ethiopic, numeric, and other (bare punctuation fragments
            # that slipped through) are kept as-is
            parts.append(text)

    return " ".join(parts)


# ── Sentence-level classification ────────────────────────────────────────────


def classify_text(text: str) -> str:
    """
    Apply script classification to every token in a space-separated
    string of text.

    Args:
        text: A sentence or text block, already punctuation-padded
              by Layer 3.

    Returns:
        The text with every token classified/tagged, joined by single
        spaces.

    Examples:
        >>> classify_text("ፕሬዚዳንቱ Biden ተናገሩ")
        'ፕሬዚዳንቱ <LATIN>Biden</LATIN> ተናገሩ'
    """
    if not text or not text.strip():
        return text

    tokens = text.split(" ")
    classified = [classify_token(t) for t in tokens if t != ""]
    return " ".join(classified)
