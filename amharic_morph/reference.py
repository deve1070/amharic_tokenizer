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

from amharic_morph.levels import (
    bare_consonant,
    fidel_order,
    is_glottal,
    is_labiovelar,
    consonant_skeleton,
)
from amharic_morph.morphology import (
    DEFAULT_PREFIXES,
    NEGATIVE_PREFIX,
    NEGATIVE_SUFFIX,
    SUFFIX_TABLE,
)
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

# Inflectional suffixes as the reference writes them after Layer 0: the
# Level 4 suffix table's morphemes, normalized.
#
# Left out are the suffixes whose letters the reference also uses for
# something Level 4 keeps inside the stem, where peeling them off would
# charge Level 4 for a split it does not make by design:
# - ኤ, which marks derivation on nouns (ጥያቅ_ኤ);
# - bare ኣ. On a noun the reference uses it for derivation (ከተም_ኣ), and
#   on a verb for the 3ms ending of an ኣ-final verb (ገብ_አ, ተ_ነስ_አ),
#   which is part of Level 4's stem ገባ;
# - ኦ and ኣት, which on a noun are derivation (ጉዝ_ኦ, አመት_አት) and on a
#   verb the gerund and the object (ሆን_ኦ). They are peeled when Level
#   4's stem is a verb. See VERB_INFLECTIONAL_SUFFIXES.
_DERIVATION_LOOKALIKES = {"ኤ", "ኣ"}
_VERB_LOOKALIKES = {"ኦ", "ኣት"}
INFLECTIONAL_SUFFIXES = tuple(
    dict.fromkeys(
        normalize(s.morpheme)
        for s in SUFFIX_TABLE
        if s.morpheme not in _DERIVATION_LOOKALIKES | _VERB_LOOKALIKES
    )
)
VERB_INFLECTIONAL_SUFFIXES = INFLECTIONAL_SUFFIXES + tuple(
    normalize(m) for m in sorted(_VERB_LOOKALIKES)
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
    long as they are in the given inventories. The negative አል counts
    as a prefix only when ም follows it in the same reading. At least one morpheme
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
        # አል is the negative only when ም follows it, as Level 4 requires.
        # In አል_ኧ (አለ) it is the verb stem.
        if m[start] == NEGATIVE_PREFIX and NEGATIVE_SUFFIX not in m[start + 1:]:
            break
        start += 1
    while end - 1 > start and m[end - 1] in suffixes:
        end -= 1
    return m[:start], m[start:end], m[end:]


def affixes_agree(
    segmentation,
    analysis: ReferenceAnalysis,
    prefixes=DEFAULT_PREFIXES,
    suffixes=INFLECTIONAL_SUFFIXES,
) -> bool:
    """
    True if a Level 4 analysis has the reading's inflectional affixes.

    Both sides are normalized, so Level 4's label ኣቸው matches the
    reference's አቸው. With the default inventory, a verb stem also peels
    the gerunds and the object (VERB_INFLECTIONAL_SUFFIXES). The stems are not compared, because the
    reference's core may still carry derivation.

    Args:
        segmentation: A Level 4 Segmentation.
        analysis:     One reference reading of the same word.
        prefixes:     Inflectional prefixes, in normalized spelling.
        suffixes:     Inflectional suffixes, in normalized spelling.

    Examples:
        >>> from amharic_morph.morphology import Segmentation
        >>> ours = Segmentation("ቤታቸው", "ቤት", suffixes=("ኣቸው",))
        >>> affixes_agree(ours, parse_annotation("ቤት_አቸው")[0])
        True
    """
    if segmentation.category == "verb" and suffixes is INFLECTIONAL_SUFFIXES:
        suffixes = VERB_INFLECTIONAL_SUFFIXES
    pre, _, suf = outer_affixes(analysis, prefixes, suffixes)
    return pre == tuple(normalize(p) for p in segmentation.prefixes) and suf == tuple(
        normalize(s) for s in segmentation.suffixes
    )


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
        if any(affixes_agree(ours, a, prefixes, suffixes) for a in analyses):
            agreeing += count
        else:
            disagreements[word] = count
    return Agreement(compared, agreeing, disagreements)


# ── Root proposals ────────────────────────────────────────────────────────────
#
# Level 1 reads a reviewed root from each lexicon entry (see
# docs/dev_plans/consonant_skeleton_redesign.md). The proposal for an entry is a
# vote: every reference word that Level 4 maps onto the entry casts its
# annotated root, weighted by its corpus frequency. Pooling the forms
# outvotes a typo on any one of them.

# Below this share of the root-carrying votes, the winner is flagged.
SPLIT_SHARE = 0.8

CHECK_SPLIT = "split vote"
CHECK_DIFFERS = "differs from skeleton"
CHECK_NO_ROOT = "no IRIT root"
CHECK_HIDDEN_W = "hidden ው?"
CHECK_HIDDEN_Y = "hidden ይ?"

# Vowel orders that a hidden weak radical leaves behind: ው surfaces as
# the ኡ or ኦ of its neighbour (ሞት, ም-ው-ት), ይ as its ኢ or ኤ.
_HIDDEN_RADICALS = ((CHECK_HIDDEN_W, "ው", (2, 7)), (CHECK_HIDDEN_Y, "ይ", (3, 5)))

# Review order: rows needing a decision first.
_CHECK_PRIORITY = (CHECK_SPLIT, CHECK_DIFFERS, CHECK_NO_ROOT)


def root_votes(segment, reference, counts) -> dict[str, Counter]:
    """
    Pool the reference roots of every word onto its Level 4 stem.

    Only a word whose Level 4 analysis the reference supports votes:
    its affixes must agree with the primary reading (affixes_agree), and
    it must have no infix, which the reference does not mark. Otherwise
    a Level 4 error sends the vote to the wrong entry: Level 4 reads
    የለምና as የ + ለ + ምን + ኣ, and its root ል-ም would land on ምን.

    A word votes with the root of its primary reading, or votes None when
    that reading has no single well-formed root. Alternatives do not
    vote: ላይ[ለ_አይ] reads the same string as ለ + አየ "to see", a
    different word, and its root is not ላይ's. Radicals are reduced
    to their bare consonant, so the reference's labiovelar ቍ is read as
    ቅ, per the Level 1 ruling on labiovelars.

    Args:
        segment:   Callable from a word to its Level 4 Segmentation.
        reference: A root-based reference, as returned by load_reference().
        counts:    Corpus frequency table. A word absent from it votes
                   with weight 1, since the reference was annotated on
                   other text.

    Returns:
        Dict mapping each stem to a Counter of radical tuples, None among
        them for forms that give no root.
    """
    votes: dict[str, Counter] = {}
    for word, analyses in reference.items():
        analysis = segment(word)
        if not analysis.known or analysis.infix or not affixes_agree(analysis, analyses[0]):
            continue
        root = analyses[0].root
        if root is not None and is_well_formed_root(root):
            root = tuple(bare_consonant(r) for r in root)
        else:
            root = None
        votes.setdefault(analysis.stem, Counter())[root] += counts.get(word, 0) or 1
    return votes


@dataclass(frozen=True)
class RootProposal:
    """
    The proposed root of one entry.

    Attributes:
        root:   Proposed radicals.
        share:  The winner's share of the root-carrying votes, or None
                when the proposal is the skeleton.
        checks: Reasons the reviewer should look closely, from the
                CHECK_* constants.
    """

    root: tuple[str, ...]
    share: float | None
    checks: tuple[str, ...]


def propose_root(word: str, votes: Counter) -> RootProposal:
    """
    Propose a root for a lexicon entry from its votes.

    The heaviest root wins. Forms the reference left unanalysed (None)
    do not count against it: the reference has no root on half the
    entries, and silence says nothing about which root is right. With no
    root votes at all, the proposal is the consonant skeleton.

    Args:
        word:  The entry.
        votes: Its Counter from root_votes(), possibly empty.

    Returns:
        A RootProposal.

    Examples:
        >>> propose_root("መሆን", Counter({("ህ", "ን"): 3})).checks
        ('differs from skeleton', 'hidden ው?')
    """
    skeleton = tuple(consonant_skeleton(word) or ())
    rooted = Counter({r: w for r, w in votes.items() if r is not None})
    checks = []
    if rooted:
        root, weight = max(rooted.items(), key=lambda item: (item[1], item[0]))
        share = round(weight / sum(rooted.values()), 4)
        if share < SPLIT_SHARE:
            checks.append(CHECK_SPLIT)
        if root != skeleton:
            checks.append(CHECK_DIFFERS)
    else:
        root, share = skeleton, None
        checks.append(CHECK_NO_ROOT)

    orders = {fidel_order(c) for c in word if not is_glottal(c) and not is_labiovelar(c)}
    for check, radical, vowel_orders in _HIDDEN_RADICALS:
        if orders & set(vowel_orders) and radical not in root:
            checks.append(check)
    return RootProposal(root, share, tuple(checks))


def _vote_summary(votes: Counter) -> str:
    rooted = Counter({r: w for r, w in votes.items() if r is not None})
    if not rooted:
        return "no root given"
    total = sum(rooted.values())
    parts = [
        f"{RADICAL_SEPARATOR.join(r)} {w / total:.0%}"
        for r, w in sorted(rooted.items(), key=lambda item: (-item[1], item[0]))[:4]
        if round(w / total, 2)
    ]
    unanalysed = votes.get(None, 0) / sum(votes.values())
    if round(unanalysed, 2):
        parts.append(f"no root on {unanalysed:.0%}")
    return "; ".join(parts)


def root_review_rows(words, votes: dict[str, Counter]) -> list[dict]:
    """
    Rows for the root review file, decisions first.

    Rows are ordered by their first check in _CHECK_PRIORITY (split
    vote, then a vote that differs from the skeleton, then no reference
    root, then plain agreement) and, within a group, by vote weight.

    Args:
        words: The lexicon entries' words.
        votes: As returned by root_votes().

    Returns:
        Dicts keyed by lexicon_builder.ROOT_REVIEW_COLUMNS, reviewed
        and notes left empty.
    """
    keyed = []
    for word in words:
        entry_votes = votes.get(word, Counter())
        proposal = propose_root(word, entry_votes)
        group = next(
            (i for i, c in enumerate(_CHECK_PRIORITY) if c in proposal.checks),
            len(_CHECK_PRIORITY),
        )
        row = {
            "word": word,
            "proposed": RADICAL_SEPARATOR.join(proposal.root),
            "skeleton": RADICAL_SEPARATOR.join(consonant_skeleton(word) or ()),
            "irit": _vote_summary(entry_votes),
            "check": "; ".join(proposal.checks),
            "reviewed": "",
            "notes": "",
        }
        keyed.append(((group, -sum(entry_votes.values()), word), row))
    return [row for _, row in sorted(keyed, key=lambda item: item[0])]
