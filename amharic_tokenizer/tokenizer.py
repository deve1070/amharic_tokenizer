"""
The AmharicTokenizer API (spec section 5.4).

Exposes all four representation levels of a word through one call. The
levels are different views of the same text, not stages of a transform,
so a caller asks for the one it wants and never assembles them by hand.

── What each level runs on ──────────────────────────────────────────────

Levels 3 and 2 describe the SURFACE word. Level 3 is the script as
written and Level 2 is its full fidel decomposition, so both are
computed straight from the word:

    ቤቶች  fidel  ['ቤ', 'ቶ', 'ች']
    ቤቶች  sound  ['ብ', 'ኤ', 'ት', 'ኦ', 'ች']

Level 1 is different. It is the root of the INDEPENDENT morpheme alone,
so it cannot be computed until Level 4 has found the morpheme
boundaries. Run on the surface form, ቤቶች gives ብ-ት-ች, with the plural
suffix's consonant sitting inside the root. Run on the stem Level 4
returns, it gives ብ-ት. That dependency is the whole reason the
computational order is 4 -> 3 -> 2 -> 1.

── Text, not just words ─────────────────────────────────────────────────

tokenize() accepts a word or a whole sentence. Text goes through the
corpus pipeline first — normalize, punctuation, script classification —
because every level below assumes normalized input. Without Layer 0 the
same word in two spellings gives two different roots.

── Unknown words ────────────────────────────────────────────────────────

A word with no lexicon anchor is returned whole at the morpheme level
and its root is None. Levels 3 and 2 still work on it, since they need
no lexicon. See morphology.segment_word() for why nothing is guessed.
"""

from pathlib import Path

from amharic_tokenizer.levels import level1_root, level2_sound, level3_fidel
from amharic_tokenizer.lexicon_builder import lexicon_index, load_lexicon
from amharic_tokenizer.morphology import Segmentation, segment_word
from amharic_tokenizer.normalizer import normalize
from amharic_tokenizer.punctuation import normalize_punctuation
from amharic_tokenizer.script_classifier import classify_text

MORPHEME = "morpheme"
FIDEL = "fidel"
SOUND = "sound"
ROOT = "root"
BPE = "bpe"

LEVELS = (MORPHEME, FIDEL, SOUND, ROOT, BPE)

DEFAULT_LEXICON = Path(__file__).parent / "lexicon" / "lexicon.json"


class AmharicTokenizer:
    """
    Morphologically-aware Amharic tokenizer.

    Examples:
        >>> tok = AmharicTokenizer()
        >>> tok.tokenize("ቤቶች", level="fidel")
        ['ቤ', 'ቶ', 'ች']
        >>> tok.tokenize("ቤቶች", level="sound")
        ['ብ', 'ኤ', 'ት', 'ኦ', 'ች']
    """

    def __init__(
        self,
        lexicon_path: str | None = None,
        validated_only: bool = True,
    ):
        """
        Args:
            lexicon_path:   Lexicon JSON to load. Defaults to the one
                            packaged with the library.
            validated_only: Anchor Level 4 on validated entries only.
                            Default true — an unvalidated entry is a
                            corpus statistic, and segmenting against one
                            propagates its errors.
        """
        path = str(lexicon_path or DEFAULT_LEXICON)
        self.lexicon = lexicon_index(load_lexicon(path), validated_only)

    def words(self, text: str) -> list[str]:
        """
        Pipeline-normalized word tokens.

        Runs Layer 0 normalization, Layer 3 punctuation and Layer 4
        script classification, then splits on whitespace. Punctuation
        arrives already space-separated from its word, so splitting is
        enough.

        Args:
            text: A word, a sentence, or a passage.

        Returns:
            List of tokens, punctuation among them.

        Examples:
            >>> AmharicTokenizer().words("ቤቶች ናቸው።")
            ['ቤቶች', 'ናቸው', '።']
        """
        return classify_text(normalize_punctuation(normalize(text))).split()

    def segment(self, word: str) -> Segmentation:
        """
        Level 4: the morpheme analysis of one word.

        Args:
            word: A single normalized word.

        Returns:
            A Segmentation. Its known flag is false when the word has no
            lexicon anchor.

        Examples:
            >>> AmharicTokenizer().segment("ቤቶች").morphemes
            ['ቤት', 'ኦች']
        """
        return segment_word(word, self.lexicon)

    def root(self, word: str) -> list[str] | None:
        """
        Level 1: the consonantal root of the word's independent morpheme.

        Args:
            word: A single normalized word.

        Returns:
            The root consonants, or None when the word has no lexicon
            anchor or no root can be extracted. Per spec section 2.2,
            None is documented behaviour for grammatical morphemes and
            loanwords, not an error.

        Examples:
            >>> AmharicTokenizer().root("ቤቶች")
            ['ብ', 'ት']
        """
        analysis = self.segment(word)
        if not analysis.known:
            return None
        return level1_root(analysis.stem)

    def analyze(self, word: str) -> dict:
        """
        All four levels of one word at once.

        Args:
            word: A single normalized word.

        Returns:
            Dict with word, morpheme, fidel, sound, root and known.

        Examples:
            >>> AmharicTokenizer().analyze("ቤቶች")["morpheme"]
            ['ቤት', 'ኦች']
        """
        analysis = self.segment(word)
        return {
            "word": word,
            "morpheme": analysis.morphemes,
            "fidel": level3_fidel(word),
            "sound": level2_sound(word),
            "root": level1_root(analysis.stem) if analysis.known else None,
            "known": analysis.known,
        }

    def tokenize(self, text: str, level: str = MORPHEME) -> list[str]:
        """
        Tokenize text at one representation level.

        Args:
            text:  A word, a sentence, or a passage.
            level: One of morpheme, fidel, sound, root, bpe.

        Returns:
            Flat list of tokens across every word in the text. At the
            root level, words with no root contribute nothing, since a
            root of None is not a token.

        Raises:
            ValueError:          If level is not a known level.
            NotImplementedError: For bpe, until a SentencePiece model
                                 is trained.

        Examples:
            >>> tok = AmharicTokenizer()
            >>> tok.tokenize("ቤቶች", level="morpheme")
            ['ቤት', 'ኦች']
            >>> tok.tokenize("ቤቶች", level="root")
            ['ብ', 'ት']
        """
        if level == BPE:
            raise NotImplementedError(
                "The SentencePiece model is not trained yet. Train it on "
                "morpheme-segmented output first — see spec section 5.2."
            )
        if level not in LEVELS:
            raise ValueError(
                f"unknown level: {level!r}. Expected one of {', '.join(LEVELS)}"
            )

        tokens: list[str] = []
        for word in self.words(text):
            if level == MORPHEME:
                tokens.extend(self.segment(word).morphemes)
            elif level == FIDEL:
                tokens.extend(level3_fidel(word))
            elif level == SOUND:
                tokens.extend(level2_sound(word))
            elif level == ROOT:
                tokens.extend(self.root(word) or ())
        return tokens
