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

from collections import Counter

from amharic_morph.levels import fidel_order, from_sound_text, sound_text, to_order6
from amharic_morph.morphology import Segmentation

# A fused suffix rewrites its carrier to this order and adds no vowel of
# its own: ሄደ + ኩ -> ሄድኩ. Every other carrier order contributes one vowel
# symbol to the suffix.
_BARE = 6
_INFIX_ORDER = 4
# A fused prefix takes the next morpheme's initial አ as this vowel.
_GLOTTAL_A = "አ"
_GLOTTAL_E = "እ"
_ABSORBED = "ኣ"


def _suffix_length(shape) -> int:
    """How many sound-text symbols a suffix occupies at the end of a word."""
    vowel = 0 if shape.carrier_order in (None, _BARE) else 1
    return vowel + len(sound_text(shape.tail))


def _prefix_sound(prefix: str, fusion: str, absorbed: str) -> str:
    """
    A prefix's sound text as written.

    A prefix fused with an አ stops at its consonant; the ኣ after it stands
    for the absorbed አ of what follows: ያለ = ይ | ኣልኧ. When what follows is
    itself a prefix (ካልሆነም = ከ + አል + ...), that prefix starts with this ኣ
    in place of its own አ (እኧ). A preposition that dropped an እ is written
    whole: በዚህ = ብኧ | ዝኢህ; when what follows is a prefix, that prefix
    has lost its እ (በንደዚህ = ብኧ | ንድኧ | ዝኢህ).

    Args:
        prefix:   The prefix morpheme.
        fusion:   The letter this prefix absorbed from what follows, or "".
        absorbed: The letter the previous prefix took from this one, or "".
    """
    text = prefix[:-1] + to_order6(prefix[-1]) if fusion == _GLOTTAL_A else prefix
    sound = sound_text(text)
    if absorbed == _GLOTTAL_A:
        sound = _ABSORBED + sound[len(sound_text(_GLOTTAL_A)):]
    elif absorbed == _GLOTTAL_E:
        sound = sound[len(sound_text(_GLOTTAL_E)):]
    return sound


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

    fused = segmentation.prefix_fused or ("",) * len(segmentation.prefixes)
    prefixes = []
    absorbed = ""
    for prefix, fusion in zip(segmentation.prefixes, fused):
        prefixes.append(_prefix_sound(prefix, fusion, absorbed))
        absorbed = fusion
    suffix_lengths = [_suffix_length(s) for s in segmentation.suffix_shapes]
    start = sum(len(p) for p in prefixes)
    end = len(whole) - sum(suffix_lengths)
    stem = [whole[start:end]]

    if segmentation.infix:
        # The stem's written letters, between the prefixes and the plain
        # tails of the suffixes; a fused carrier stays with the stem.
        # Fusion leaves a prefix's letter count unchanged (የ -> ያ).
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


# ── Training input ───────────────────────────────────────────────────────────
#
# Training sees exactly what encoding encodes: each segment on its own, the
# word's first one with a leading space, which becomes the word mark ▁. The
# corpus is reduced to a table of distinct segments and their counts, and
# SentencePiece never forms a piece across two rows of it.
#
# Running text with the segments marked by a delimiter does not work: its
# pretokenization_delimiter still lets pieces form across the delimiter
# (measured, sentencepiece 0.2.1: ▁ይኣልኧው from ይ|ኣልኧ|ው). Those pieces can
# never be used, since encoding never sees the segments joined, and they
# crowd out the segments it does see.

# The corpus tags (Layer 4) are kept as single pieces.
TAGS = ("<LATIN>", "</LATIN>", "<URL>", "</URL>")


def word_segments(word: str, tokenizer, morphemes: bool) -> list[str]:
    """
    The strings a word is encoded as: its sound text, cut at its morpheme
    boundaries for the morpheme-aware model, the first with a leading space.

    Examples:
        >>> from amharic_morph import AmharicTokenizer
        >>> word_segments("ቤቶች", AmharicTokenizer(), morphemes=True)
        [' ብኤት', 'ኦች']
    """
    segments = sound_segments(tokenizer.segment(word)) if morphemes else [sound_text(word)]
    return [" " + segments[0]] + segments[1:]


def count_segments(lines, tokenizer, morphemes: bool) -> Counter:
    """
    How often each training segment occurs in the lines.

    Args:
        lines:     Corpus lines, words separated by spaces.
        tokenizer: The AmharicTokenizer whose Level 4 cuts the words.
        morphemes: True for the morpheme-aware model, False for plain.
    """
    counts: Counter = Counter()
    for line in lines:
        for word in line.split():
            counts.update(word_segments(word, tokenizer, morphemes))
    return counts


def write_segment_counts(counts: Counter, path) -> None:
    """Write segment counts as SentencePiece's tab-separated input."""
    with open(path, "w", encoding="utf-8") as f:
        for segment, count in counts.most_common():
            f.write(f"{segment}\t{count}\n")


# ── Training ─────────────────────────────────────────────────────────────────


def _sentencepiece():
    try:
        import sentencepiece
    except ImportError as error:
        raise ImportError(
            "the BPE level needs sentencepiece: pip install 'amharic-morph[bpe]'"
        ) from error
    return sentencepiece


def train(counts_path, model_prefix, vocab_size: int, num_threads: int = 4) -> None:
    """
    Train a unigram SentencePiece model on segment counts.

    The settings that make the model lossless and boundary-respecting:
    each row is one segment, so no piece crosses a morpheme boundary; the
    leading space of a word's first segment is the word mark, so no dummy
    prefix is added and spaces are kept as they are; every character seen
    is kept (character_coverage 1.0), and an unseen one is written as its
    UTF-8 bytes rather than <unk>. The text is not normalized again: Layer 0
    already has, and SentencePiece's default NFKC would turn the
    labiovelar mark ʷ into w.

    Args:
        counts_path:  Segment counts written by write_segment_counts().
        model_prefix: Writes model_prefix.model and .vocab.
        vocab_size:   Pieces in the vocabulary.
        num_threads:  Training threads.
    """
    _sentencepiece().SentencePieceTrainer.train(
        input=str(counts_path),
        input_format="tsv",
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        user_defined_symbols=list(TAGS),
        hard_vocab_limit=False,
        num_threads=num_threads,
        minloglevel=2,
    )


# ── Encoding ─────────────────────────────────────────────────────────────────

_WORD_MARK = "▁"


class SubwordModel:
    """
    A trained model, encoding words through Level 4.

    The morpheme-aware model encodes each segment of a word on its own,
    so no piece spans a morpheme boundary. The word's first piece carries
    the word mark ▁. decode() restores the text exactly.

    Args:
        model_file: A .model file written by train().
        tokenizer:  The AmharicTokenizer that cuts words. Built from the
                    default lexicon when None.
        morphemes:  True if the model was trained morpheme-aware.
    """

    def __init__(self, model_file, tokenizer=None, morphemes: bool = True):
        self._sp = _sentencepiece().SentencePieceProcessor(model_file=str(model_file))
        if tokenizer is None:
            from amharic_morph.tokenizer import AmharicTokenizer
            tokenizer = AmharicTokenizer()
        self._tokenizer = tokenizer
        self.morphemes = morphemes

    def encode_word(self, word: str) -> list[str]:
        """The pieces of one word, in sound text."""
        return [
            piece
            for segment in word_segments(word, self._tokenizer, self.morphemes)
            for piece in self._sp.encode(segment, out_type=str)
        ]

    def encode(self, text: str) -> list[str]:
        """The pieces of a text whose words are separated by spaces."""
        return [piece for word in text.split() for piece in self.encode_word(word)]

    def encode_ids(self, text: str) -> list[int]:
        """As encode(), as vocabulary ids."""
        return [self._sp.piece_to_id(piece) for piece in self.encode(text)]

    def decode(self, pieces: list[str]) -> str:
        """The text the pieces came from."""
        return self._restore(self._sp.decode_pieces(pieces))

    def decode_ids(self, ids: list[int]) -> str:
        """As decode(), from vocabulary ids."""
        return self._restore(self._sp.decode_ids(ids))

    @staticmethod
    def _restore(sound: str) -> str:
        if sound.startswith(" "):
            sound = sound[1:]
        return from_sound_text(sound)

    @staticmethod
    def readable(pieces: list[str]) -> list[str]:
        """The pieces written as fidel for display: ▁ብኤት -> ▁ቤት."""
        return [from_sound_text(piece) for piece in pieces]

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()
