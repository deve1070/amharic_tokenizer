"""
Affix inventories, fusion arithmetic, and Level 4 morpheme segmentation.

Level 4 is the level everything else is computed from. Levels 3, 2 and 1
in levels.py describe a single morpheme; this module is what finds the
morphemes in a surface word, so the computational order is 4 -> 3 -> 2
-> 1.

── Why segmentation is not a substring search ───────────────────────────

Spec section 2.3.2 says to search the lexicon for the independent
morpheme within the surface word. Taken literally that fails on the
spec's own example, because ቤት does not occur inside ቤቶች. The plural
morpheme ኦች does not concatenate: its ኦ fuses into the stem's final
consonant, so ት (order 6) surfaces as ቶ (order 7) and the letters of
ቤት are no longer contiguous.

Segmentation therefore works constructively. It removes affixes to build
candidate stems, inverting fusion with fidel arithmetic, and asks the
lexicon whether each candidate is a real independent morpheme. The
lexicon is the only authority; nothing here guesses.

── Inverting fusion is one-to-many ──────────────────────────────────────

A fused carrier does not record what the stem's own final vowel was.
ሀገሮች restores to ሀገር, with the carrier going back to order 6, but
ከተሞች restores to ከተማ, an ኣ-final stem whose carrier goes back to order
4. Reducing to order 6 in every case would silently produce ከተም.

So the inverter offers both readings and the lexicon chooses. This is
the general shape of the module: generate every analysis the morphology
allows, then let the lexicon and the ranking rules pick one.

── Choosing between analyses ────────────────────────────────────────────

Ranked in this order:

1. Fewest morphemes. An analysis that posits less structure wins, which
   is what makes the lexicon authoritative: ስራ is a lexicon entry, so
   it stays whole rather than becoming ስር + ኣ.
2. Affixes attested for that entry. allows_prefix and allows_suffix
   record what the corpus actually saw on that word, so an analysis
   using only attested affixes beats one that does not. Attestation is
   incomplete, so this ranks rather than rejects.
3. Longest stem, then the morpheme sequence itself for determinism.

── The infix is a pattern, not a morpheme ───────────────────────────────

Spec section 2.3.1 gives one infix example, ቀጭን -> ቀጫጭን, and reading ጫ
off it as the morpheme does not survive contact with another word: ነጭ
gives ነጫጭ but ትልቅ gives ትላልቅ and ረጅም gives ረጃጅም. The inserted fidel
is a copy of the consonant that follows it, carrying the vowel ኣ, so it
is always that consonant's order-4 form. detach_infix() tests for that
rather than consulting a list.

── Unknown words ────────────────────────────────────────────────────────

A word whose stem is not in the lexicon comes back whole, with known
false. It is not segmented on a guess. A wrong boundary injects a
morpheme that does not exist into everything downstream, while an
unsegmented word still lets the BPE layer learn subword structure from
it. This is the conservative answer to open question 6 and is a policy
choice, not a limitation of the search.
"""

from dataclasses import dataclass, field

from amharic_tokenizer.levels import fidel_order, is_ethiopic, to_order6

# Per CLAUDE.md, minus ም and ና. Both inventories are open questions, so
# they are parameters rather than constants baked into the algorithm.
#
# ም and ና are listed as prefixes in the open questions but do not behave
# as prefixes in the corpus. Being single characters they strip the front
# off ordinary words: ምግብ (food) -> ግብ, ምድር (earth) -> ድር, ናት -> ት, ምን
# -> ን. In Amharic both are conjunctions or suffixes (ና "and", ም "also"),
# which is consistent with the corpus evidence.
DEFAULT_PREFIXES = ("ስለ", "ወደ", "እንደ", "እስከ", "የ", "በ", "ከ", "ለ")

# Suffixes that concatenate without changing the stem.
DEFAULT_PLAIN_SUFFIXES = ("ን", "ም", "ና", "ች", "ው", "ዋ", "ዎ", "ዎች", "ኦች")

# The order an infix always surfaces at. See detach_infix().
_INFIX_ORDER = 4

# Vowel orders that a suffix can fuse onto a stem's final consonant.
# Order 7 (ኦ) marks the plural before ች; order 2 (ኡ) and order 4 (ኣ)
# mark the definite forms.
_FUSED_PLURAL_ORDER = 7
_FUSED_DEFINITE_ORDERS = (2, 4)

# Morpheme names for the fused definite, by the order they surface at.
_DEFINITE_MORPHEMES = {2: "ኡ", 4: "ኣ"}

# Stem-final orders a fused plural can be restored to. Order 6 is the
# bare consonant (ሀገሮች -> ሀገር); order 4 is an ኣ-final stem whose vowel
# the plural overwrites (ከተሞች -> ከተማ).
_PLURAL_CARRIER_ORDERS = (6, 4)

_ORDER_OFFSET = {order: order - 1 for order in range(1, 9)}


def _at_order(char: str, order: int) -> str:
    """The given order of a character's own fidel family."""
    if not is_ethiopic(char):
        return char
    base = ord(char) - ord(char) % 8
    return chr(base + _ORDER_OFFSET[order])


def strip_prefixes(word: str, prefixes=DEFAULT_PREFIXES) -> set[str]:
    """
    Candidate stems obtained by removing one prefix.

    Prefixes concatenate without altering the stem, so this is plain
    string removal.

    Args:
        word:     Surface form.
        prefixes: Prefix inventory.

    Returns:
        Set of candidate stems. Empty if no prefix applies.

    Examples:
        >>> sorted(strip_prefixes("የቤት"))
        ['ቤት']
    """
    stems = set()
    for prefix in prefixes:
        if word.startswith(prefix) and len(word) > len(prefix):
            stems.add(word[len(prefix):])
    return stems


def detach_prefix(word: str, prefixes=DEFAULT_PREFIXES) -> list[tuple[str, str]]:
    """
    Prefixes removable from a surface form, with the remainder.

    Args:
        word:     Surface form.
        prefixes: Prefix inventory.

    Returns:
        List of (remainder, prefix morpheme).

    Examples:
        >>> detach_prefix("የቤት")
        [('ቤት', 'የ')]
    """
    return [
        (word[len(prefix):], prefix)
        for prefix in prefixes
        if word.startswith(prefix) and len(word) > len(prefix)
    ]


def detach_suffix(
    word: str, suffixes=DEFAULT_PLAIN_SUFFIXES
) -> list[tuple[str, str]]:
    """
    Suffixes removable from a surface form, with the remainder.

    Covers all three suffix shapes: plain concatenation, vowel fusion
    followed by a consonant (the ኦች plural), and vowel fusion alone
    (the ኡ and ኣ definites). Fused suffixes are reported by their
    morpheme name, not by the characters that appear on the surface,
    because those characters are shared with the stem.

    A fused plural yields more than one remainder, since the carrier
    does not record the stem's own final vowel. Both readings are
    returned and the lexicon decides between them.

    Args:
        word:     Surface form.
        suffixes: Plain suffix inventory.

    Returns:
        List of (remainder, suffix morpheme), without duplicates.

    Examples:
        >>> detach_suffix("ቤቶች")
        [('ቤት', 'ኦች'), ('ቤታ', 'ኦች')]
        >>> detach_suffix("ቤቱ")
        [('ቤት', 'ኡ')]
        >>> detach_suffix("ቤትን")
        [('ቤት', 'ን')]
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(stem: str, morpheme: str) -> None:
        if stem and (stem, morpheme) not in seen:
            seen.add((stem, morpheme))
            found.append((stem, morpheme))

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix):
            add(word[: -len(suffix)], suffix)

    # Fusion + consonant: ቤቶች -> ቤት. The ኦ rides on the final consonant,
    # which sits at order 7, and ች follows it.
    if len(word) >= 3 and word.endswith("ች"):
        carrier = word[-2]
        if fidel_order(carrier) == _FUSED_PLURAL_ORDER:
            for order in _PLURAL_CARRIER_ORDERS:
                add(word[:-2] + _at_order(carrier, order), "ኦች")

    # Fusion alone: ቤቱ -> ቤት, ቤታ -> ቤት.
    if len(word) >= 2:
        last = word[-1]
        order = fidel_order(last)
        if order in _FUSED_DEFINITE_ORDERS:
            add(word[:-1] + to_order6(last), _DEFINITE_MORPHEMES[order])

    return found


def strip_suffixes(word: str, suffixes=DEFAULT_PLAIN_SUFFIXES) -> set[str]:
    """
    Candidate stems obtained by removing one suffix.

    Args:
        word:     Surface form.
        suffixes: Plain suffix inventory.

    Returns:
        Set of candidate stems. Empty if no suffix applies.

    Examples:
        >>> "ቤት" in strip_suffixes("ቤቶች")
        True
        >>> "ቤት" in strip_suffixes("ቤቱ")
        True
    """
    return {stem for stem, _ in detach_suffix(word, suffixes)}


def detach_infix(word: str) -> list[tuple[str, str]]:
    """
    Infixes removable from a surface form, with the remainder.

    There is no infix inventory. Spec section 2.3.1 gives one example,
    ቀጭን -> ቀጫጭን, and reading ጫ off it as a morpheme is a mistake: the
    inserted fidel is different in every other word.

        ነጭ    -> ነጫጭ     ጫ
        ትልቅ   -> ትላልቅ    ላ
        ረጅም   -> ረጃጅም    ጃ
        ጥቁር   -> ጥቃቁር    ቃ
        ቀይ    -> ቀያይ     ያ
        ወፍራም  -> ወፍራራም   ራ

    What is constant is the pattern. The infix is a COPY of the
    consonant that follows it, carrying the vowel ኣ — so it is always
    the order-4 form of that consonant's own family. ጭ takes ጫ, ል takes
    ላ, ቁ takes ቃ. Detection is that test, which is why nothing needs to
    be listed.

    This parallels the fused suffixes, where ኦች rides its ኦ on the
    stem's final consonant. Here the invariant is the vowel ኣ and the
    consonant is copied from the stem. The difference is that the copy
    is additional material — the stem keeps its own ጭ — so the inserted
    fidel is reported as it appears rather than reduced to ኣ.

    Args:
        word: Surface form.

    Returns:
        List of (remainder, infix morpheme).

    Examples:
        >>> detach_infix("ቀጫጭን")
        [('ቀጭን', 'ጫ')]
        >>> detach_infix("ነጫጭ")
        [('ነጭ', 'ጫ')]
        >>> detach_infix("ትላልቅ")
        [('ትልቅ', 'ላ')]
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index in range(1, len(word) - 1):
        infix = word[index]
        if fidel_order(infix) != _INFIX_ORDER:
            continue
        if to_order6(infix) != to_order6(word[index + 1]):
            continue
        stem = word[:index] + word[index + 1:]
        if stem and (stem, infix) not in seen:
            seen.add((stem, infix))
            found.append((stem, infix))
    return found


@dataclass(frozen=True)
class Segmentation:
    """
    One Level 4 analysis of a surface word.

    Exactly one independent morpheme, per the core constraint of the
    language, plus whatever dependent morphemes attach to it.

    Attributes:
        word:     The surface form analysed.
        stem:     The independent morpheme (ዋና ቃል).
        prefixes: Prefix morphemes, in surface order.
        suffixes: Suffix morphemes, in surface order.
        infix:    Infix morpheme, or None.
        known:    Whether stem was found in the lexicon. False means the
                  word was returned whole, not that it was guessed at.
    """

    word: str
    stem: str
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    infix: str | None = None
    known: bool = True

    @property
    def morphemes(self) -> list[str]:
        """
        All morphemes in order, the independent one among them.

        The infix follows the stem it sits inside, since it has no
        position of its own in a flat sequence.

        Examples:
            >>> Segmentation("ቤቶች", "ቤት", suffixes=("ኦች",)).morphemes
            ['ቤት', 'ኦች']
        """
        parts = list(self.prefixes)
        parts.append(self.stem)
        if self.infix:
            parts.append(self.infix)
        parts.extend(self.suffixes)
        return parts

    @property
    def morpheme_count(self) -> int:
        """How many morphemes the analysis posits."""
        return len(self.morphemes)


@dataclass(frozen=True)
class _Node:
    """A partly stripped word during the search."""

    text: str
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    infix: str | None = None
    depth_prefix: int = 0
    depth_suffix: int = 0


def _analyses(
    word: str,
    lexicon: dict,
    max_prefixes: int,
    max_suffixes: int,
    prefixes,
    suffixes,
) -> list[Segmentation]:
    """Every analysis whose stem is a lexicon entry."""
    found: list[Segmentation] = []
    seen: set[tuple] = set()
    queue = [_Node(word)]

    while queue:
        node = queue.pop()
        key = (node.text, node.prefixes, node.suffixes, node.infix)
        if key in seen:
            continue
        seen.add(key)

        if node.text in lexicon:
            found.append(
                Segmentation(
                    word=word,
                    stem=node.text,
                    prefixes=node.prefixes,
                    suffixes=node.suffixes,
                    infix=node.infix,
                )
            )

        if node.depth_prefix < max_prefixes:
            for rest, prefix in detach_prefix(node.text, prefixes):
                queue.append(
                    _Node(
                        rest,
                        node.prefixes + (prefix,),
                        node.suffixes,
                        node.infix,
                        node.depth_prefix + 1,
                        node.depth_suffix,
                    )
                )

        if node.depth_suffix < max_suffixes:
            for rest, suffix in detach_suffix(node.text, suffixes):
                queue.append(
                    _Node(
                        rest,
                        node.prefixes,
                        (suffix,) + node.suffixes,
                        node.infix,
                        node.depth_prefix,
                        node.depth_suffix + 1,
                    )
                )

        # One infix at most: it is an internal modification of a single
        # stem, not something that stacks.
        if node.infix is None:
            for rest, infix in detach_infix(node.text):
                queue.append(
                    _Node(
                        rest,
                        node.prefixes,
                        node.suffixes,
                        infix,
                        node.depth_prefix,
                        node.depth_suffix,
                    )
                )

    return found


def _attested(analysis: Segmentation, lexicon: dict) -> bool:
    """True if every affix is recorded for that entry in the lexicon."""
    entry = lexicon.get(analysis.stem, {})
    allowed_prefix = entry.get("allows_prefix") or ()
    allowed_suffix = entry.get("allows_suffix") or ()
    if analysis.infix and entry.get("allows_infix") is False:
        return False
    return all(p in allowed_prefix for p in analysis.prefixes) and all(
        s in allowed_suffix for s in analysis.suffixes
    )


def rank_analyses(
    analyses: list[Segmentation], lexicon: dict
) -> list[Segmentation]:
    """
    Order competing analyses best first.

    Fewest morphemes, then whether every affix is attested for that
    entry, then longest stem, then the morpheme sequence itself so the
    order is deterministic. See the module docstring for why.

    Args:
        analyses: Analyses of one word.
        lexicon:  Word to lexicon entry.

    Returns:
        New list, best first.
    """
    return sorted(
        analyses,
        key=lambda a: (
            a.morpheme_count,
            not _attested(a, lexicon),
            -len(a.stem),
            a.morphemes,
        ),
    )


def segment_word(
    word: str,
    lexicon: dict,
    max_prefixes: int = 2,
    max_suffixes: int = 3,
    prefixes=DEFAULT_PREFIXES,
    suffixes=DEFAULT_PLAIN_SUFFIXES,
) -> Segmentation:
    """
    Level 4: split a surface word into its morphemes.

    Args:
        word:         A single surface word, already normalized by the
                      pipeline.
        lexicon:      Word to lexicon entry, from
                      lexicon_builder.lexicon_index(). Validated
                      entries only, or segmentation anchors on
                      unconfirmed corpus statistics.
        max_prefixes: Most prefixes to strip from one word.
        max_suffixes: Most suffixes to strip from one word.
        prefixes:     Prefix inventory.
        suffixes:     Plain suffix inventory.

    Returns:
        The best Segmentation. When no analysis has a stem in the
        lexicon, the whole word is returned as the stem with known
        false.

    Examples:
        >>> lex = {"ቤት": {"allows_suffix": ["ኦች"]}}
        >>> segment_word("ቤቶች", lex).morphemes
        ['ቤት', 'ኦች']
        >>> segment_word("ቤት", lex).morphemes
        ['ቤት']
        >>> segment_word("ኮምፒውተር", lex).known
        False
    """
    if not word:
        return Segmentation(word=word, stem=word, known=False)

    found = _analyses(
        word, lexicon, max_prefixes, max_suffixes, prefixes, suffixes
    )
    if not found:
        return Segmentation(word=word, stem=word, known=False)
    return rank_analyses(found, lexicon)[0]


def segment_all(
    word: str,
    lexicon: dict,
    max_prefixes: int = 2,
    max_suffixes: int = 3,
    prefixes=DEFAULT_PREFIXES,
    suffixes=DEFAULT_PLAIN_SUFFIXES,
) -> list[Segmentation]:
    """
    Every Level 4 analysis of a word, best first.

    Useful for inspecting ambiguity that segment_word() resolves
    silently.

    Args:
        word:         A single surface word.
        lexicon:      Word to lexicon entry.
        max_prefixes: Most prefixes to strip from one word.
        max_suffixes: Most suffixes to strip from one word.
        prefixes:     Prefix inventory.
        suffixes:     Plain suffix inventory.

    Returns:
        Ranked analyses. Empty when the word has no lexicon anchor.
    """
    found = _analyses(
        word, lexicon, max_prefixes, max_suffixes, prefixes, suffixes
    )
    return rank_analyses(found, lexicon)
