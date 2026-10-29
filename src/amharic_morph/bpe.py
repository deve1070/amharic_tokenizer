"""
The BPE level: SentencePiece over lossless sound text, cut at Level 4's
morpheme boundaries.

── Why sound text ───────────────────────────────────────────────────────

At the fidel level a fused suffix cannot be cut off: in ቤቶች the ቶ is
the stem's ት and the plural's ኦ in one letter. In sound text the fusion
comes apart, so every morpheme boundary falls between two symbols:

    ቤቶች     ብኤት | ኦች
    ይሆናል    ይ | ህኦን | ኣል
    ሄድኩ     ህኤድ | ክኡ

The segments are the written forms (ሆን, not the lexicon's ሆነ), so the
pieces join back into exactly the text they came from. Level 4 and Level
1 still report the lexicon form and the root.
"""

from amharic_morph.levels import fidel_order, sound_text, to_order6
from amharic_morph.morphology import Segmentation

# A fused suffix rewrites its carrier to this order and adds no vowel of
# its own: ሄደ + ኩ -> ሄድኩ. Every other carrier order contributes one vowel
# symbol to the suffix.
_BARE = 6
_INFIX_ORDER = 4


def _suffix_length(shape) -> int:
    """How many sound-text symbols a suffix occupies at the end of a word."""
    vowel = 0 if shape.carrier_order in (None, _BARE) else 1
    return vowel + len(sound_text(shape.tail))


def _infix_index(stem_letters: str) -> int | None:
    """Where the infix sits among the stem's written letters (ትላልቅ -> 1)."""
    for i in range(1, len(stem_letters) - 1):
        here, after = stem_letters[i], stem_letters[i + 1]
        if fidel_order(here) == _INFIX_ORDER and to_order6(here) == to_order6(after):
            return i
    return None


def sound_segments(segmentation: Segmentation) -> list[str]:
    """
    The word's sound text, cut at its morpheme boundaries.

    Prefixes, the stem (in two pieces around an infix) and suffixes, in
    written order. They always join back into sound_text(word). An unknown
    word is one segment.

    Examples:
        >>> from amharic_morph import AmharicTokenizer
        >>> sound_segments(AmharicTokenizer().segment("ቤቶች"))
        ['ብኤት', 'ኦች']
    """
    word = segmentation.word
    whole = sound_text(word)
    if not segmentation.known:
        return [whole]

    prefixes = [sound_text(p) for p in segmentation.prefixes]
    suffix_lengths = [_suffix_length(s) for s in segmentation.suffix_shapes]
    start = sum(len(p) for p in prefixes)
    end = len(whole) - sum(suffix_lengths)
    stem = [whole[start:end]]

    if segmentation.infix:
        # The stem's written letters, between the prefixes and the plain
        # tails of the suffixes; a fused carrier stays with the stem.
        first = sum(len(p) for p in segmentation.prefixes)
        last = len(word) - sum(len(s.tail) for s in segmentation.suffix_shapes)
        index = _infix_index(word[first:last])
        if index is not None:
            before = len(sound_text(word[first:first + index]))
            width = len(sound_text(word[first + index]))
            text = stem[0]
            stem = [text[:before], text[before:before + width], text[before + width:]]

    suffixes = []
    position = end
    for length in suffix_lengths:
        suffixes.append(whole[position:position + length])
        position += length
    return [s for s in prefixes + stem + suffixes if s]
