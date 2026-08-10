"""
Layer 3 — Punctuation and Whitespace Normalization

Runs after character normalization (Layer 0) and before script
classification (Layer 4). Every downstream layer assumes this one has
already run: the script classifier and the segmenter both expect
punctuation to be space-separated from the words it attaches to.

Three jobs, applied in this order:

1. Doubled wordspace conversion: ፡፡ → ።
2. Latin period conversion: . → ። where the period genuinely ends a
   sentence (see the guards below)
3. Punctuation space-padding: ። ፣ ፤ ፥ ፦ ፡ ? ! #
4. Whitespace normalization

Order matters. ፡፡ must be converted before padding, otherwise padding
splits it into "፡ ፡" and the doubled form is lost — and ፡፡ is the
sentence-final marker on roughly 18% of CC-100 lines.

── On ፡ (U+1361, Ethiopic wordspace) ────────────────────────────────────

In older Amharic orthography ፡ separated words. Modern Amharic uses the
ASCII space for that and treats ፡ as a punctuation mark. CC-100 matches
the modern usage: of the single ፡ occurrences sampled, the overwhelming
majority already sit between spaces, and only a small residue appear
between two Ethiopic characters in the old word-separating position.

Padding handles both readings correctly. Where ፡ is already isolated,
padding is a no-op. Where it is attached to a word (ቤት፡) padding detaches
it as punctuation. Where it genuinely separates two words (ቤት፡ሰው) padding
splits them, which is also what we want.

── On the Latin period ──────────────────────────────────────────────────

The period is not always a sentence boundary in Amharic text. It has four
other uses, and each needs a guard:

  1. Numbers      Amharic uses Arabic numerals, so decimals: 3.5
  2. Abbreviations  ዓ.ም, እ.ኤ.አ, ኃ.የተ.የግ.ማ, ኪ.ሜ
  3. Ellipsis     "..." marking continuation
  4. URLs         used as in English

The spec's rule — convert when the preceding character is Ethiopic —
is not sufficient on its own, because abbreviations also have Ethiopic
characters before the period. Abbreviations are recognized structurally
instead: a token whose period-separated segments are all one or two
Ethiopic characters is an abbreviation, not a sentence end.
"""

import re

# Punctuation that gets space-padded, per spec section 4.3 Layer 3.
_PAD_CHARS = "።፣፤፥፦፡?!#"
_PAD_PATTERN = re.compile(f"([{re.escape(_PAD_CHARS)}])")

# Doubled Ethiopic wordspace used as a full stop.
_DOUBLE_WORDSPACE_PATTERN = re.compile("፡፡")

_ETHIOPIC_RANGE = re.compile(r"[ሀ-፿]")

# Characters that may legitimately precede a sentence-final period.
_CLOSING_PUNCTUATION = set(")]}»”’\"'›")

# Brackets and quotes stripped before a token is analysed for
# abbreviation structure, so that "(ኤፍ.ቢ.ሲ)" is recognized.
_WRAPPING_CHARS = "()[]{}«»“”\"'‹›"

# Abbreviations whose segments are longer than two characters, so the
# structural rule below cannot catch them. Extend as the corpus reveals more.
_KNOWN_ABBREVIATIONS = frozenset({"ወዘተ"})

_URL_HINT_PATTERN = re.compile(
    r"://|^www\.|\.(com|org|net|edu|gov|io|co|et)(/|$|\?)", re.IGNORECASE
)

# Maximum Ethiopic characters per segment for a token to read as an
# abbreviation. ኃ.የተ.የግ.ማ has segments of length 1 and 2; ordinary words
# that end a sentence (ነው. ናቸው. ይችላል.) are longer.
_MAX_ABBREVIATION_SEGMENT = 2


def _is_ethiopic(char: str) -> bool:
    """True if char is in the Ethiopic Unicode block (U+1200-U+137F)."""
    return bool(char) and bool(_ETHIOPIC_RANGE.match(char))


def _is_url_like(token: str) -> bool:
    """
    True if the token looks like a URL or domain name.

    Periods inside URLs are never sentence boundaries.
    """
    return bool(_URL_HINT_PATTERN.search(token))


def is_abbreviation(token: str) -> bool:
    """
    True if the token is an Amharic abbreviation such as ዓ.ም or እ.ኤ.አ.

    Recognized two ways:

    - By name, for abbreviations with segments too long for the
      structural rule (ወዘተ).
    - Structurally, when every period-separated segment is one or two
      Ethiopic characters. This covers ዓ.ም, እ.ኤ.አ, ኪ.ሜ, ኤፍ.ቢ.ሲ and
      ኃ.የተ.የግ.ማ while excluding ordinary sentence-final words, whose
      stems are longer.

    A single Ethiopic character followed by a period (ቁ. ሚ. ም.) is also
    an abbreviation — a one-character sentence is not plausible.

    Examples:
        >>> is_abbreviation("ዓ.ም")
        True
        >>> is_abbreviation("ዓ.ም.")
        True
        >>> is_abbreviation("ነው.")
        False
    """
    core = token.strip(_WRAPPING_CHARS).rstrip(".")
    if not core:
        return False

    if core in _KNOWN_ABBREVIATIONS:
        return True

    if "." not in core:
        # A lone Ethiopic character with a trailing period, e.g. ቁ.
        return len(core) == 1 and _is_ethiopic(core)

    segments = core.split(".")
    return all(
        seg
        and len(seg) <= _MAX_ABBREVIATION_SEGMENT
        and all(_is_ethiopic(c) for c in seg)
        for seg in segments
    )


def _convert_periods_in_token(token: str) -> str:
    """
    Convert sentence-ending Latin periods to ። within a single token.

    Leaves periods alone when they belong to a URL, an ellipsis, a
    decimal number, or an abbreviation.
    """
    if "." not in token:
        return token

    if _is_url_like(token) or "…" in token:
        return token

    # Two or more consecutive periods is an ellipsis, not a sentence end.
    if ".." in token:
        return token

    if is_abbreviation(token):
        return token

    def replace(match: re.Match) -> str:
        i = match.start()
        before = token[i - 1] if i > 0 else ""
        after = token[i + 1] if i + 1 < len(token) else ""

        # Decimal number: 3.5
        if before.isdigit() and after.isdigit():
            return "."

        # Spec rule: convert only after Ethiopic or closing punctuation.
        # This also leaves list numbering ("1." "2.") untouched.
        if _is_ethiopic(before) or before in _CLOSING_PUNCTUATION:
            return "።"

        return "."

    return re.sub(r"\.", replace, token)


def convert_latin_periods(text: str) -> str:
    """
    Convert Latin periods that end sentences into ። (U+1362).

    Args:
        text: Text with Latin periods, already character-normalized.

    Returns:
        Text with sentence-final periods replaced by ።. Periods in
        numbers, abbreviations, ellipses, and URLs are left as-is.

    Examples:
        >>> convert_latin_periods("ኢትዮጵያ ሀገር ናት.")
        'ኢትዮጵያ ሀገር ናት።'
        >>> convert_latin_periods("በ2010 ዓ.ም ተጀመረ.")
        'በ2010 ዓ.ም ተጀመረ።'
        >>> convert_latin_periods("ከ3.5 ቢሊዮን በላይ")
        'ከ3.5 ቢሊዮን በላይ'
    """
    if not text or "." not in text:
        return text

    return " ".join(_convert_periods_in_token(t) for t in text.split(" "))


def pad_punctuation(text: str) -> str:
    """
    Insert a space before and after each of ። ፣ ፤ ፥ ፦ ፡ ? ! #

    This is what makes the token stream space-separated, which every
    later layer depends on.

    Args:
        text: Text to pad.

    Returns:
        Text with padded punctuation. Whitespace is not collapsed here;
        normalize_whitespace() handles that.

    Examples:
        >>> pad_punctuation("ቤት፣ ሰው")
        'ቤት ፣  ሰው'
    """
    if not text:
        return text
    return _PAD_PATTERN.sub(r" \1 ", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse whitespace runs, normalize line endings, and trim.

    Args:
        text: Text to normalize.

    Returns:
        Text with \\r\\n and \\r converted to \\n, runs of horizontal
        whitespace collapsed to a single space, no spaces hugging a
        newline, and no leading or trailing whitespace.

    Examples:
        >>> normalize_whitespace("ሰላም    ነው")
        'ሰላም ነው'
    """
    if not text:
        return text

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace only, so line structure survives.
    t = re.sub(r"[^\S\n]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


def normalize_punctuation(text: str) -> str:
    """
    Run the full Layer 3 transformation.

    Applies, in order: ፡፡ → ።, Latin period conversion, punctuation
    space-padding, and whitespace normalization.

    Args:
        text: Character-normalized text (Layer 0 output).

    Returns:
        Text ready for script classification (Layer 4).

    Examples:
        >>> normalize_punctuation("ኢትዮጵያ ሀገር ናት.")
        'ኢትዮጵያ ሀገር ናት ።'
        >>> normalize_punctuation("ሰላም ነው፡፡")
        'ሰላም ነው ።'
        >>> normalize_punctuation("ቤት፣ሰው")
        'ቤት ፣ ሰው'
        >>> normalize_punctuation("")
        ''
    """
    if not text or not text.strip():
        return ""

    # Must run before padding, or ፡፡ is split into two separate ፡ tokens.
    t = _DOUBLE_WORDSPACE_PATTERN.sub("።", text)
    t = convert_latin_periods(t)
    t = pad_punctuation(t)
    return normalize_whitespace(t)


def normalize_punctuation_lines(lines: list[str]) -> list[str]:
    """
    Apply Layer 3 to each line independently.

    Args:
        lines: List of character-normalized text lines.

    Returns:
        List of the same length, each line transformed.
    """
    return [normalize_punctuation(line) for line in lines]
