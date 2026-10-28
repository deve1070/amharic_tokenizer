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

── One table of suffix shapes ───────────────────────────────────────────

Every suffix is a row of SUFFIX_TABLE describing one written shape:
plain (ስራ + ችን -> ስራችን), fused into the stem's last letter (ቤት + ኣችን
-> ቤታችን, the carrier ት rewritten to ታ), or vowel-deleting (ሄደ + ኩ ->
ሄድኩ, the carrier ደ rewritten to ድ). Fused and vowel-deleting rows are
the same mechanism: a carrier at a given order, letters after it, and
the orders the carrier may be restored to. One table is also what the
inverse (joining morphemes back into a word) will read, so the two
directions cannot drift apart the way three copies of the inversion
once did.

Each row also has a position. Suffixes follow the stem in a fixed
order, at most one per position (see ADJECTIVE ... AND), which is what
rules out ስራ + ዎ + ችን ("your our") for ስራዎችን, ስራ + ዎች + ን.

Each row also says what kind of stem it attaches to. A verb row needs a
verb, a noun row a stem whose category is set and is not verb, and an
any row attaches to everything. An entry whose category is still
unset takes only the any rows. That gate is what stops the order-6 to
order-1 restoration of the verb rows from inventing verb analyses for
nouns: ቤትን inverts to ቤተ + ን (1pl) as well as ቤት + ን, and the first
is refused unless ቤተ is a verb.

The negative perfective is the one piece that is not a suffix. አል is a
verb-only prefix, accepted only when ም is among the suffixes:
አልሆነም is አል + ሆነ + ም.

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

── One suffix, two functions ────────────────────────────────────────────

ኡ attaches the same way to a noun and to a verb, and Level 4 finds the
same boundary in both, but it is not the same morpheme:

    ቤት  + ኡ  -> ቤቱ    the definite article, "the house"
    ሆነ  + ኡ  -> ሆኑ    third person plural agreement, "they became"

The fidel arithmetic cannot tell them apart, because nothing in the
surface form differs. What decides is the part of speech of the stem,
which is why suffix_function() reads the entry's category and returns
None when that field is still unfilled. An unlabelled morpheme is
honest; a guessed one would put a definite article on a verb.

── Unknown words ────────────────────────────────────────────────────────

A word whose stem is not in the lexicon comes back whole, with known
false. It is not segmented on a guess. A wrong boundary injects a
morpheme that does not exist into everything downstream, while an
unsegmented word still lets the BPE layer learn subword structure from
it. This is the conservative answer to open question 6 and is a policy
choice, not a limitation of the search.
"""

from dataclasses import dataclass, field

from amharic_morph.levels import fidel_order, is_ethiopic, to_order, to_order6

# Per CLAUDE.md, minus ም and ና. Both inventories are open questions, so
# they are parameters rather than constants baked into the algorithm.
#
# ም and ና are listed as prefixes in the open questions but do not behave
# as prefixes in the corpus. Being single characters they strip the front
# off ordinary words: ምግብ (food) -> ግብ, ምድር (earth) -> ድር, ናት -> ት, ምን
# -> ን. In Amharic both are conjunctions or suffixes (ና "and", ም "also"),
# which is consistent with the corpus evidence.
#
# በየ is the distributive "each, every" (በየአመቱ "every year"), one
# morpheme. Read as በ + የ the word would mean "in the year's". The IRIT
# annotation writes it as one prefix too.
#
# አል is the negative of the perfective verb (አልሆነም). It is gated: see
# NEGATIVE_PREFIX.
DEFAULT_PREFIXES = ("ስለ", "ወደ", "እንደ", "እስከ", "በየ", "አል", "የ", "በ", "ከ", "ለ")

# The negative perfective prefix and the suffix it pairs with. Without
# the pairing, አል would strip the front off every word that happens to
# start with those letters.
NEGATIVE_PREFIX = "አል"
NEGATIVE_SUFFIX = "ም"

# The order an infix always surfaces at. See detach_infix().
_INFIX_ORDER = 4

# Stem-final orders a fused suffix can be restored to. A fused vowel
# OVERWRITES whatever vowel the stem's final fidel carried, so the
# carrier records the suffix and says nothing about the stem.
#
# Order 6 is the bare consonant, which is where most nouns end:
# ሀገሮች -> ሀገር, ቤቱ -> ቤት. Order 4 is an ኣ-final stem, ከተሞች -> ከተማ and
# ወጡ -> ወጣ. Order 1 is where a verb's third person masculine
# perfective ends, ሆኑ -> ሆነ and ሞቱ -> ሞተ.
_CARRIER_ORDERS = (6, 4, 1)

# Orders of a letter that ends in a vowel: everything but the bare
# consonant.
_VOWEL_ORDERS = (1, 2, 3, 4, 5, 7, 8)

# ── Derivation rules ─────────────────────────────────────────────────────
#
# Which rule detached a suffix is tracked because policy differs per
# rule. frequency.derived_stems() trusts the plural unconditionally and
# puts a frequency guard on the rest.

RULE_PREFIX = "prefix"
RULE_SUFFIX = "suffix"
RULE_PLURAL_FUSION = "plural_fusion"
RULE_DEFINITE_FUSION = "definite_fusion"

# Order-4 (ኣ) fusion is the ambiguous one: many native words simply end
# in an order-4 character without being an inflected form of anything.
# ስራ (work) is not the definite of ስር (root); አርባ (forty) is not the
# definite of አርብ (Friday). Order-7 + ች (the plural) has no such
# ambiguity — no Amharic word ends that way by coincidence.
RULE_AMBIGUOUS_FUSION = "ambiguous_fusion"

# Every other fused row. derived_stems() puts its frequency guard on it,
# because possessives run backwards over personal names (ጌታቸው, ተስፋዬ).
RULE_FUSION = "fusion"

# Stem kinds a suffix row may attach to.
ANY = "any"
NOUN = "noun"
VERB = "verb"


# Suffix positions. Suffixes follow the stem in a fixed order, at most
# one per position. The IRIT annotation attests the same order: ኡ + ን,
# ኦች + ኡ + ን and ኦች + ኣቸው + ን are its commonest sequences.
#
#   noun   stem + ADJECTIVE + PLURAL + POSSESSIVE + CASE + ALSO + AND
#          ውጤት + ኣማ + ዎች;  ቤት + ኦች + ኡ + ን + ም
#   verb   stem + SUBJECT + OBJECT + ALSO + AND
#          ሰጠ + ኡ + ኝ "they gave me";  አል + ሄደ + ኩ + ም
#
# POSSESSIVE holds the definite too: a noun takes one or the other.
ADJECTIVE = 0
PLURAL = SUBJECT = 1
POSSESSIVE = OBJECT = 2
CASE = 3
ALSO = 4
AND = 5


@dataclass(frozen=True)
class SuffixShape:
    """
    One written shape of one suffix.

    Attributes:
        morpheme:       The morpheme's name. A fused suffix is named by its
                        vowel-initial form (ኦች, ኣችን), a plain one by its
                        written shape (ዎች, ችን). Always Amharic, so Levels
                        3 and 2 can decompose it.
        kind:           ANY, NOUN or VERB: the stems it attaches to.
        tail:           The letters written after the carrier, or the
                        whole suffix for a plain row.
        slot:           Its position after a non-verb stem.
        verb_slot:      Its position after a verb, when that differs.
                        ኡ is the definite on a noun, after the plural,
                        and the subject on a verb, before the object.
        carrier_order:  The order a fused row leaves on the stem's last
                        letter. None for a plain row.
        restore_orders: The orders the carrier may be returned to.
        after_orders:   For a plain row, the orders the letter before it
                        may have, or None for any.
        rule:           RULE_* reported to the corpus analysis.
    """

    morpheme: str
    kind: str
    tail: str
    slot: int
    verb_slot: int | None = None
    carrier_order: int | None = None
    restore_orders: tuple[int, ...] = ()
    after_orders: tuple[int, ...] | None = None
    rule: str = RULE_SUFFIX

    def position(self, category: str | None) -> int:
        """The row's position after a stem of this category."""
        if category == VERB and self.verb_slot is not None:
            return self.verb_slot
        return self.slot


def _plain(morpheme, kind, slot, verb_slot=None, after_orders=None):
    return SuffixShape(morpheme, kind, morpheme, slot, verb_slot, after_orders=after_orders)


def _fused(morpheme, kind, slot, carrier_order, tail, restore_orders, verb_slot=None,
           rule=RULE_FUSION):
    return SuffixShape(
        morpheme, kind, tail, slot, verb_slot, carrier_order, restore_orders, rule=rule
    )


# The suffix inventory. Rows are tried in order; the lexicon and the
# ranking choose among the analyses they produce.
SUFFIX_TABLE = (
    # ን is the case ending on a noun and the 1pl object on a verb (ሰጠን).
    _plain("ን", ANY, CASE, OBJECT),
    _plain("ም", ANY, ALSO),
    _plain("ና", ANY, AND),
    # The plural after a vowel on a noun, 3fs on a verb (ሄደች).
    _plain("ች", ANY, PLURAL, SUBJECT),
    _plain("ው", ANY, POSSESSIVE, OBJECT),
    _plain("ዋ", ANY, POSSESSIVE, OBJECT),
    _plain("ዎ", ANY, POSSESSIVE, OBJECT),
    _plain("ዎች", ANY, PLURAL),
    _plain("ኦች", ANY, PLURAL),
    # Possessives after a vowel-final stem: ስራዬ, ስራችን, ስራቸው.
    _plain("ዬ", NOUN, POSSESSIVE, after_orders=_VOWEL_ORDERS),
    _plain("ችን", NOUN, POSSESSIVE, after_orders=_VOWEL_ORDERS),
    _plain("ችሁ", ANY, POSSESSIVE, OBJECT, _VOWEL_ORDERS),
    _plain("ቸው", ANY, POSSESSIVE, OBJECT, _VOWEL_ORDERS),
    # The same after ኢ or ኤ, with a ያ glide: ፓርቲያችን, ጥያቄያችን.
    _plain("ያችን", NOUN, POSSESSIVE, after_orders=(3, 5)),
    # Your (m, f), on any stem: ቤትህ, ቤትሽ; on a verb the object.
    _plain("ህ", ANY, POSSESSIVE, OBJECT),
    _plain("ሽ", ANY, POSSESSIVE, OBJECT),
    # The adjective ending after an ኣ-final noun: ጤናማ.
    _plain("ማ", NOUN, ADJECTIVE, after_orders=(4,)),
    # I, after an ኣ-final verb (ገባሁ); me, after any verb (ሰጠኝ).
    _plain("ሁ", VERB, SUBJECT, after_orders=(4,)),
    _plain("ኝ", VERB, OBJECT, after_orders=_VOWEL_ORDERS),
    # The plural: ቤት + ኦች -> ቤቶች, the ኦ riding on the stem's last letter.
    _fused("ኦች", ANY, PLURAL, 7, "ች", _CARRIER_ORDERS, rule=RULE_PLURAL_FUSION),
    # The definite on a noun, 3pl agreement on a verb: ቤቱ, ሆኑ.
    _fused("ኡ", ANY, POSSESSIVE, 2, "", _CARRIER_ORDERS, SUBJECT, RULE_DEFINITE_FUSION),
    # Possessives after a consonant-final noun, ruled 2026-10-22: ኣ turns
    # the order-6 letter into its order-4 form.
    _fused("ኤ", NOUN, POSSESSIVE, 5, "", (6,)),
    _fused("ዋ", NOUN, POSSESSIVE, 8, "", (6,)),
    _fused("ኣችን", NOUN, POSSESSIVE, 4, "ችን", (6,)),
    _fused("ኣችሁ", ANY, POSSESSIVE, 4, "ችሁ", (6, 1), OBJECT),
    _fused("ኣቸው", ANY, POSSESSIVE, 4, "ቸው", (6, 1), OBJECT),
    # The adjective ending: ውጤት + ኣማ -> ውጤታማ.
    _fused("ኣማ", NOUN, ADJECTIVE, 4, "ማ", (6,)),
    # Her, as an object: ሰጠ + ኣት -> ሰጣት.
    _fused("ኣት", VERB, OBJECT, 4, "ት", (1,)),
    # Perfective subjects delete the stem's last vowel: ሄደ + ኩ -> ሄድኩ.
    *(_fused(m, VERB, SUBJECT, 6, m, (1,)) for m in ("ኩ", "ክ", "ሽ", "ን")),
    # Gerunds. Bare ኣ is the 3fs (ሆና), and on a noun ኣ is never a suffix
    # of its own, so it restores only to a verb's order 1. ስራ is not the
    # "definite" of ስር.
    _fused("ኣ", VERB, SUBJECT, 4, "", (1,), rule=RULE_AMBIGUOUS_FUSION),
    _fused("ኦ", VERB, SUBJECT, 7, "", (1,)),
    # After an ኣ-final verb, and after some order-1 verbs, the gerund is a
    # separate letter: ገባ -> ገብቶ, ሰጠ -> ሰጥቶ, ቀረ -> ቀርቶ.
    _fused("ቶ", VERB, SUBJECT, 6, "ቶ", (4, 1)),
    _fused("ታ", VERB, SUBJECT, 6, "ታ", (4, 1)),
)


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


def detach_suffix_shapes(
    word: str, suffixes=SUFFIX_TABLE
) -> list[tuple[str, SuffixShape]]:
    """
    Every suffix row that inverts on a surface form, with the remainder.

    The single implementation of suffix inversion. detach_suffix() and
    detach_suffix_with_rule() are views of it; segmentation reads the
    rows themselves, because the row's kind gates the analysis once the
    stem's category is known.

    A fused row yields one remainder per restore order, since the
    carrier does not record the stem's own final vowel. The lexicon
    decides between them.

    Args:
        word:     Surface form.
        suffixes: Suffix table.

    Returns:
        List of (remainder, row), without duplicates, in table order.

    Examples:
        >>> ("ቤት", "ኣችን") in [(s, r.morpheme) for s, r in detach_suffix_shapes("ቤታችን")]
        True
    """
    found: list[tuple[str, SuffixShape]] = []
    seen: set[tuple[str, SuffixShape]] = set()

    def add(stem: str, shape: SuffixShape) -> None:
        # Restoring a carrier to the order it already sits at is a no-op
        # that would report the whole word as its own stem.
        if stem and stem != word and (stem, shape) not in seen:
            seen.add((stem, shape))
            found.append((stem, shape))

    for shape in suffixes:
        if not word.endswith(shape.tail):
            continue
        rest = word[: len(word) - len(shape.tail)]
        if shape.carrier_order is None:
            if not rest:
                continue
            if shape.after_orders is None or fidel_order(rest[-1]) in shape.after_orders:
                add(rest, shape)
            continue
        # A fused row needs the carrier and at least one letter before it.
        if len(rest) < 2 or not is_ethiopic(rest[-1]):
            continue
        carrier = rest[-1]
        if fidel_order(carrier) != shape.carrier_order:
            continue
        for order in shape.restore_orders:
            try:
                add(rest[:-1] + to_order(carrier, order), shape)
            except ValueError:
                # A labiovelar has no letter at some orders.
                continue
    return found


def detach_suffix(word: str, suffixes=SUFFIX_TABLE) -> list[tuple[str, str]]:
    """
    Suffixes removable from a surface form, with the remainder.

    Fused suffixes are reported by their morpheme name, not by the
    characters that appear on the surface, because those characters
    are shared with the stem.

    Args:
        word:     Surface form.
        suffixes: Suffix table.

    Returns:
        List of (remainder, suffix morpheme), without duplicates.

    Examples:
        >>> detach_suffix("ቤቶች")
        [('ቤቶ', 'ች'), ('ቤት', 'ኦች'), ('ቤታ', 'ኦች'), ('ቤተ', 'ኦች')]
        >>> detach_suffix("ቤቱ")
        [('ቤት', 'ኡ'), ('ቤታ', 'ኡ'), ('ቤተ', 'ኡ')]
    """
    return list(dict.fromkeys((stem, s.morpheme) for stem, s in detach_suffix_shapes(word, suffixes)))


def detach_suffix_with_rule(word: str, suffixes=SUFFIX_TABLE) -> list[tuple[str, str, str]]:
    """
    detach_suffix() with the rule that produced each remainder.

    Corpus analysis needs the rule in order to weigh a derivation
    (frequency.derived_stems). It does not gate by category: the
    candidates it works on have none.

    Args:
        word:     Surface form.
        suffixes: Suffix table.

    Returns:
        List of (remainder, suffix morpheme, RULE_*), without duplicates.

    Examples:
        >>> ("ቤት", "ኦች", "plural_fusion") in detach_suffix_with_rule("ቤቶች")
        True
        >>> detach_suffix_with_rule("ቤትን")[0]
        ('ቤት', 'ን', 'suffix')
    """
    return list(
        dict.fromkeys(
            (stem, s.morpheme, s.rule) for stem, s in detach_suffix_shapes(word, suffixes)
        )
    )


def strip_suffixes(word: str, suffixes=SUFFIX_TABLE) -> set[str]:
    """
    Candidate stems obtained by removing one suffix.

    Args:
        word:     Surface form.
        suffixes: Suffix table.

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


# Morpheme kinds, by position relative to the independent morpheme.
PREFIX = "prefix"
INDEPENDENT = "independent"
INFIX = "infix"
SUFFIX = "suffix"

# Categories whose members take the definite article. Anything that is
# not a verb does; the split is verb against the rest.
_VERB = "verb"

# Suffixes that mean one thing on a verb and another on everything else.
# Keyed suffix -> (function on a non-verb, function on a verb).
_SUFFIX_FUNCTIONS = {
    "ኡ": ("definite", "subject_3pl"),
    "ው": ("definite", "subject_3ms"),
}


def suffix_function(suffix: str, category: str | None) -> str | None:
    """
    What a suffix does, given the part of speech of its stem.

    Only the suffixes with two readings are listed. Everything else has
    one function and needs no disambiguation, so it returns None rather
    than a gloss invented here.

    Args:
        suffix:   A suffix morpheme.
        category: The stem entry's category, or None if unfilled.

    Returns:
        The function name, or None when the suffix is unambiguous or the
        stem's category is unknown.

    Examples:
        >>> suffix_function("ኡ", "noun")
        'definite'
        >>> suffix_function("ኡ", "verb")
        'subject_3pl'
        >>> suffix_function("ኡ", None) is None
        True
        >>> suffix_function("ኦች", "noun") is None
        True
    """
    readings = _SUFFIX_FUNCTIONS.get(suffix)
    if readings is None or category is None:
        return None
    return readings[1] if category == _VERB else readings[0]


@dataclass(frozen=True)
class Morpheme:
    """
    One morpheme with its position and, where it matters, its function.

    text is always a real Amharic string, so Levels 3, 2 and 1 and the
    BPE layer can consume it directly. The grammatical function is kept
    beside it rather than folded into the string, because ኡ-def would
    not be decomposable as Amharic.

    Attributes:
        text:     The morpheme.
        kind:     PREFIX, INDEPENDENT, INFIX or SUFFIX.
        function: What it does, when the morpheme has more than one
                  reading and the stem's category resolves it.
    """

    text: str
    kind: str
    function: str | None = None


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
        category: The stem entry's category, which is what resolves a
                  suffix with two readings. None when unfilled.
    """

    word: str
    stem: str
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    infix: str | None = None
    known: bool = True
    category: str | None = None

    @property
    def parts(self) -> list[Morpheme]:
        """
        All morphemes with their position and function, in order.

        Examples:
            >>> seg = Segmentation("ቤቱ", "ቤት", suffixes=("ኡ",), category="noun")
            >>> [(p.text, p.function) for p in seg.parts]
            [('ቤት', None), ('ኡ', 'definite')]
        """
        parts = [Morpheme(p, PREFIX) for p in self.prefixes]
        parts.append(Morpheme(self.stem, INDEPENDENT))
        if self.infix:
            parts.append(Morpheme(self.infix, INFIX))
        parts.extend(
            Morpheme(s, SUFFIX, suffix_function(s, self.category))
            for s in self.suffixes
        )
        return parts

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
        return [part.text for part in self.parts]

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
    # The table row of each suffix, parallel to suffixes.
    shapes: tuple[SuffixShape, ...] = ()


def _kind_allows(kind: str, category: str | None) -> bool:
    """Whether a row of this kind attaches to a stem of this category."""
    if kind == ANY:
        return True
    if category is None:
        return False
    return (category == VERB) == (kind == VERB)


def _compatible(node: _Node, category: str | None) -> bool:
    """
    Whether every affix of a finished analysis fits its stem: each row's
    kind admits the category, and the suffixes are in strictly
    increasing position.
    """
    if not all(_kind_allows(s.kind, category) for s in node.shapes):
        return False
    positions = [s.position(category) for s in node.shapes]
    if any(a >= b for a, b in zip(positions, positions[1:])):
        return False
    if NEGATIVE_PREFIX in node.prefixes:
        return category == VERB and NEGATIVE_SUFFIX in node.suffixes
    return True


def _analyses(
    word: str,
    lexicon: dict,
    max_prefixes: int,
    max_suffixes: int,
    prefixes,
    suffixes,
) -> list[Segmentation]:
    """Every analysis whose stem is a lexicon entry and whose affixes fit it."""
    found: dict[Segmentation, None] = {}
    seen: set[tuple] = set()
    queue = [_Node(word)]

    while queue:
        node = queue.pop()
        key = (node.text, node.prefixes, node.suffixes, node.infix, node.shapes)
        if key in seen:
            continue
        seen.add(key)

        if node.text in lexicon:
            category = lexicon[node.text].get("category")
            if _compatible(node, category):
                found.setdefault(
                    Segmentation(
                        word=word,
                        stem=node.text,
                        prefixes=node.prefixes,
                        suffixes=node.suffixes,
                        infix=node.infix,
                        category=category,
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
                        node.shapes,
                    )
                )

        if node.depth_suffix < max_suffixes:
            for rest, shape in detach_suffix_shapes(node.text, suffixes):
                queue.append(
                    _Node(
                        rest,
                        node.prefixes,
                        (shape.morpheme,) + node.suffixes,
                        node.infix,
                        node.depth_prefix,
                        node.depth_suffix + 1,
                        (shape,) + node.shapes,
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
                        node.shapes,
                    )
                )

    return list(found)


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
    suffixes=SUFFIX_TABLE,
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
        suffixes:     Suffix table.

    Returns:
        The best Segmentation. When no analysis has a stem in the
        lexicon, the whole word is returned as the stem with known
        false.

    Examples:
        >>> lex = {"ቤት": {"category": "noun", "allows_suffix": ["ኦች"]}}
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
    suffixes=SUFFIX_TABLE,
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
        suffixes:     Suffix table.

    Returns:
        Ranked analyses. Empty when the word has no lexicon anchor.
    """
    found = _analyses(
        word, lexicon, max_prefixes, max_suffixes, prefixes, suffixes
    )
    return rank_analyses(found, lexicon)
