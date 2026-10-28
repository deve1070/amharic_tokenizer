"""
The IRIT morphologically annotated lexicons, as a reference for Levels 4
and 1.

Yeshambel, Mothe and Assabie (SIGIR 2021) annotated about 171K Amharic
word types by hand, once stem-based and once root-based. The files
are kept in resource/irit/, which is gitignored. The download links are
in resource/resouce.txt.

── The annotation format ────────────────────────────────────────────────

Each row is `surface,annotation`. In the annotation:

    _      separates morphemes          ልጆቻችን   ልጅ_ኦች_አችን
    -      separates a root's radicals  ተደረገ    ተ_ድ-ር-ግ_ኧ
    [...]  encloses an alternative      በላይ     በ_ላይ[በ_በላይ]

A morpheme containing a hyphen is a root. Roots appear only in the
root-based file.

── Why it is a reference and not a gold standard ───────────────────────

Three properties make it a reference rather than a gold standard:

- It marks derivation as well as inflection. መንግስት is መ_ንግስ_ት there,
  while Level 4 keeps መንግስት whole as a lexicon entry. A full-sequence
  comparison therefore charges Level 4 for a split it does not make by
  design. outer_affixes() peels off only the inflectional layer.
- It has typos. ከተማ is annotated ክ-ት-ምከተማ in one row and ህዝቦች as
  ህዝብ_ኦች_ኡ_ች in another. is_well_formed_root() catches the root
  typos. The others survive, so agreement figures are an estimate.
- It is not normalized. It writes ሐሳብ and ሀሳብ, ኣችን and አችን.
  Everything read here goes through Layer 0, the same normalization the
  corpus went through, or nothing would match.

About 150 rows do not follow the format at all: punctuation attached to
the surface word (ለማ፥), unbalanced brackets, or spaces inside the
annotation. They are skipped rather than repaired, because a repair
would be a guess about what the annotator meant.
"""

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from amharic_morph.morphology import DEFAULT_PLAIN_SUFFIXES, DEFAULT_PREFIXES
from amharic_morph.normalizer import normalize

REFERENCE_DIR = Path(__file__).parent.parent / "resource" / "irit"
STEM_LEXICON = REFERENCE_DIR / "Stem-based-lexicon.txt"
ROOT_LEXICON = REFERENCE_DIR / "Root-based-lexicon.txt"

_HEADER = "Surface word"

MORPHEME_SEPARATOR = "_"
RADICAL_SEPARATOR = "-"

# Ethiopic syllables only, U+1200-U+135A. The block's punctuation
# (U+1360 onward) and numerals are excluded.
_LETTERS = "ሀ-ፚ"
_WORD = re.compile(f"[{_LETTERS}]+")
_MORPHEME = f"[{_LETTERS}-]+"
_ANALYSIS = f"{_MORPHEME}(?:_{_MORPHEME})*"
_ANNOTATION = re.compile(f"{_ANALYSIS}(?:\\[{_ANALYSIS}\\])*")
_ALTERNATIVE = re.compile(f"\\[({_ANALYSIS})\\]")

# Inflectional suffixes as the reference writes them after Layer 0. The
# plain suffixes and the definite ኡ are what Level 4 produces today. The
# plural possessives are not yet in the Level 4 inventory, so a word
# carrying one counts as a disagreement until they are added.
#
# The bare vowels ኣ (normalized to አ), ኤ and ኢ are left out. The reference
# uses them for derivation as well (ከተም_ኣ, ጥያቅ_ኤ, መር_ኢ), so peeling them
# off would charge Level 4 for derivational splits.
INFLECTIONAL_SUFFIXES = tuple(
    dict.fromkeys(
        normalize(s) for s in (*DEFAULT_PLAIN_SUFFIXES, "ኡ", "ኣችን", "ኣችሁ", "ኣቸው")
    )
)


@dataclass(frozen=True)
class ReferenceAnalysis:
    """
    One annotated reading of a surface word.

    Attributes:
        morphemes: The annotation's morphemes in order. A root appears as
                   its hyphenated form, e.g. "ድ-ር-ግ".
        roots:     The radicals of every root among the morphemes.
    """

    morphemes: tuple[str, ...]
    roots: tuple[tuple[str, ...], ...] = field(default=())

    @property
    def root(self) -> tuple[str, ...] | None:
        """
        The word's root, when the annotation gives exactly one.

        A compound gives two roots, and neither of them is the root of the
        word as a whole.

        Examples:
            >>> parse_annotation("ተ_ድ-ር-ግ_ኧ")[0].root
            ('ድ', 'ር', 'ግ')
            >>> parse_annotation("የ_ኦነግ")[0].root is None
            True
        """
        return self.roots[0] if len(self.roots) == 1 else None


def is_well_formed_root(radicals: tuple[str, ...]) -> bool:
    """
    True if every radical is exactly one Ethiopic syllable.

    A typo fuses a radical with neighbouring text (ክ-ት-ምከተማ) or loses
    it to a stray space (ን-ብ- ብ-ር). Neither is a root.

    Examples:
        >>> is_well_formed_root(("ክ", "ት", "ም"))
        True
        >>> is_well_formed_root(("ክ", "ት", "ምከተማ"))
        False
    """
    return bool(radicals) and all(
        len(r) == 1 and _WORD.fullmatch(r) for r in radicals
    )


def _analysis(text: str) -> ReferenceAnalysis:
    morphemes = tuple(text.split(MORPHEME_SEPARATOR))
    roots = tuple(
        tuple(m.split(RADICAL_SEPARATOR)) for m in morphemes if RADICAL_SEPARATOR in m
    )
    return ReferenceAnalysis(morphemes, roots)


def parse_annotation(annotation: str) -> tuple[ReferenceAnalysis, ...]:
    """
    Parse one annotation into its readings, the primary one first.

    Args:
        annotation: The annotated form of a word, as in the files.

    Returns:
        The readings, or () when the annotation does not follow the
        format (unbalanced brackets, spaces, punctuation, an empty
        morpheme).

    Examples:
        >>> [a.morphemes for a in parse_annotation("በ_ላይ[በ_በላይ]")]
        [('በ', 'ላይ'), ('በ', 'በላይ')]
    """
    text = normalize(annotation.strip())
    if not _ANNOTATION.fullmatch(text):
        return ()
    primary = text.split("[", 1)[0]
    return (_analysis(primary), *(_analysis(a) for a in _ALTERNATIVE.findall(text)))


def load_reference(path: str | Path) -> dict[str, tuple[ReferenceAnalysis, ...]]:
    """
    Read a reference lexicon file.

    Surface words are normalized, so two spellings that Layer 0 merges
    (ሐሳብ, ሀሳብ) become one key, and their readings are pooled. Readings
    repeated across duplicate rows are kept once, in first-seen order.

    Args:
        path: A stem-based or root-based lexicon file.

    Returns:
        Dict mapping each normalized surface word to its readings. Rows
        with a malformed surface word or annotation are left out.
    """
    readings: dict[str, dict[ReferenceAnalysis, None]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) != 2 or row[0] == _HEADER:
                continue
            word = normalize(row[0].strip())
            analyses = parse_annotation(row[1])
            if not _WORD.fullmatch(word) or not analyses:
                continue
            readings.setdefault(word, {}).update(dict.fromkeys(analyses))
    return {word: tuple(found) for word, found in readings.items()}


def outer_affixes(
    analysis: ReferenceAnalysis,
    prefixes=DEFAULT_PREFIXES,
    suffixes=INFLECTIONAL_SUFFIXES,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Split a reading into its inflectional prefixes, core, and
    inflectional suffixes.

    Prefixes are peeled from the left and suffixes from the right for as
    long as they are in the given inventories. At least one morpheme
    always stays in the core, because a word has exactly one independent
    morpheme. Whatever derivation the annotation marks stays inside the
    core, which is what makes the result comparable with Level 4.

    Args:
        analysis: One reading of a word.
        prefixes: Inflectional prefixes, in normalized spelling.
        suffixes: Inflectional suffixes, in normalized spelling.

    Returns:
        (prefixes, core, suffixes), each in surface order.

    Examples:
        >>> outer_affixes(parse_annotation("የ_ልጅ_ኦች_አችን")[0])
        (('የ',), ('ልጅ',), ('ኦች', 'አችን'))
        >>> outer_affixes(parse_annotation("መ_ንግስ_ት")[0])
        ((), ('መ', 'ንግስ', 'ት'), ())
    """
    m = analysis.morphemes
    start, end = 0, len(m)
    while start < end - 1 and m[start] in prefixes:
        start += 1
    while end - 1 > start and m[end - 1] in suffixes:
        end -= 1
    return m[:start], m[start:end], m[end:]


@dataclass(frozen=True)
class Agreement:
    """
    How often Level 4's inflectional affixes match the reference.

    Attributes:
        compared_tokens: Corpus tokens of words Level 4 found in its
                         lexicon and the reference annotates.
        agreeing_tokens: Those whose prefixes and suffixes match some
                         reading of the reference.
        disagreements:   Word -> corpus count, for the rest.
    """

    compared_tokens: int
    agreeing_tokens: int
    disagreements: Counter

    @property
    def rate(self) -> float | None:
        """Agreeing share of compared tokens, None if nothing was compared."""
        if not self.compared_tokens:
            return None
        return self.agreeing_tokens / self.compared_tokens


def inflection_agreement(
    segment,
    reference: dict[str, tuple[ReferenceAnalysis, ...]],
    counts: Counter,
    prefixes=DEFAULT_PREFIXES,
    suffixes=INFLECTIONAL_SUFFIXES,
) -> Agreement:
    """
    Score Level 4's inflectional affixes against the reference.

    Only words that Level 4 found in its lexicon are compared. An unknown
    word is returned whole on purpose, and coverage is measured
    elsewhere. A word agrees when its prefixes and suffixes, normalized,
    equal the outer affixes of any of its reference readings. The stem
    is not compared, since the reference's core may still carry
    derivation. The infix is not compared either, because the reference
    does not mark it.

    Args:
        segment:   Callable from a word to its Level 4 Segmentation, such
                   as AmharicTokenizer.segment.
        reference: As returned by load_reference().
        counts:    Corpus frequency table.
        prefixes:  Inflectional prefixes, in normalized spelling.
        suffixes:  Inflectional suffixes, in normalized spelling.

    Returns:
        An Agreement, weighted by corpus frequency.
    """
    compared = agreeing = 0
    disagreements: Counter = Counter()
    for word, analyses in reference.items():
        count = counts.get(word, 0)
        if not count:
            continue
        ours = segment(word)
        if not ours.known:
            continue
        compared += count
        affixes = (
            tuple(normalize(p) for p in ours.prefixes),
            tuple(normalize(s) for s in ours.suffixes),
        )
        if any(
            (pre, suf) == affixes
            for pre, _, suf in (outer_affixes(a, prefixes, suffixes) for a in analyses)
        ):
            agreeing += count
        else:
            disagreements[word] = count
    return Agreement(compared, agreeing, disagreements)
