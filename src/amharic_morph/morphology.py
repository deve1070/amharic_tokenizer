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
which is why suffix_readings() reads the entry's category and returns
None when that field is still unfilled. An unlabelled morpheme is
honest; a guessed one would put a definite article on a verb. A suffix
that stays ambiguous on one kind of stem carries every reading (ው on a
verb: object_3ms, subject_3pl, definite), and context decides.

── Unknown words ────────────────────────────────────────────────────────

A word whose stem is not in the lexicon comes back whole, with known
false. It is not segmented on a guess. A wrong boundary injects a
morpheme that does not exist into everything downstream, while an
unsegmented word still lets the BPE layer learn subword structure from
it. This is the conservative answer to open question 6 and is a policy
choice, not a limitation of the search.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from amharic_morph.levels import bare_consonant, fidel_order, is_ethiopic, is_labiovelar, to_order, to_order6

# The negative perfective prefix and the suffix it pairs with. Without
# the pairing, አል would strip the front off every word that happens to
# start with those letters. The imperfective negative አይ pairs the same
# way: አይሆንም.
NEGATIVE_PREFIX = "አል"
NEGATIVE_SUFFIX = "ም"
# The negative of the existential አለ is a verb of its own, and ም is its
# negative too: የለ + ው + ም -> የለውም "he does not have".
NEGATIVE_EXISTENTIAL = "የለ"

# The shortest noun a preposition fuses onto, the two-letter words the እ
# fusion accepts anyway, and the categories it never fuses onto. See
# _compatible().
_MIN_FUSED_NOUN = 3
_SHORT_FUSABLE = frozenset({"እኔ", "እኛ", "እሱ", "እሷ"})
_UNFUSABLE_CATEGORIES = ("preposition", "conjunction")
_MAX_PREPOSITIONS = 2

# The only category that takes the infix. See _compatible().
_INFIX_CATEGORY = "adjective"

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
#   verb   stem + SUBJECT + OBJECT + AUX + AUX_PERSON + ALSO + AND
#          ሰጠ + ኡ + ኝ "they gave me";  አል + ሄደ + ኩ + ም;
#          እ + ሆነ + ኣል + ሁ "I will become"
#
# POSSESSIVE holds the definite too: a noun takes one or the other.
ADJECTIVE = 0
PLURAL = SUBJECT = 1
POSSESSIVE = OBJECT = 2
CASE = AUX = 3
AUX_PERSON = 4
ALSO = 5
AND = 6
CONSTRUCT = 7

# Verb stems by aspect. A verb entry is its perfective; the other stems
# are reviewed fields on the entry (lexicon_builder.STEM_KINDS), each
# with a palatal variant for the forms that palatalize its last
# consonant (ሄጄ, ትሄጂ).
PERFECTIVE = "perfective"
IMPERFECTIVE = ("imperfective", "imperfective_palatal")
JUSSIVE = ("jussive", "jussive_palatal")
GERUND = ("gerund", "gerund_palatal")
STEM_KINDS = (*IMPERFECTIVE[:1], *JUSSIVE[:1], *GERUND[:1],
              IMPERFECTIVE[1], JUSSIVE[1], GERUND[1])


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
        aspects:        For a verb, the stem aspects it attaches to, or
                        None for any. ኩ needs the perfective (ሄድኩ), ኦ the
                        gerund stem (ፈልጎ on ፈልግ), ኣል the imperfective.
        bound:          True if another suffix must follow it: the ኣል
                        written ...ለ in እሆናለሁ is never word-final.
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
    aspects: tuple[str, ...] | None = None
    bound: bool = False

    def position(self, category: str | None) -> int:
        """The row's position after a stem of this category."""
        if category == VERB and self.verb_slot is not None:
            return self.verb_slot
        return self.slot


def _plain(morpheme, kind, slot, verb_slot=None, after_orders=None, aspects=None, bound=False):
    return SuffixShape(morpheme, kind, morpheme, slot, verb_slot, after_orders=after_orders,
                       aspects=aspects, bound=bound)


def _fused(morpheme, kind, slot, carrier_order, tail, restore_orders, verb_slot=None,
           rule=RULE_FUSION, aspects=None, bound=False):
    return SuffixShape(
        morpheme, kind, tail, slot, verb_slot, carrier_order, restore_orders, rule=rule,
        aspects=aspects, bound=bound,
    )


_PERFECTIVE = (PERFECTIVE,)
_IMPERFECTIVE_OR_JUSSIVE = IMPERFECTIVE + JUSSIVE
# The stems the auxiliary ኣል attaches to: the imperfective (ይሆናል) and the
# gerund, for the present perfect (ሆኗል).
_AUXILIARY_STEMS = IMPERFECTIVE + GERUND
AUXILIARY = "ኣል"

# ኣ on a verb stem, fused onto its last letter. On the gerund stem it is
# the 3fs (ሆን + ኣ -> ሆና). On the imperfective and jussive stem it is the
# weak last radical of a verb whose perfective ends in ኣ (ገባ, ገ-ብ-ʔ), as
# IRIT: የ + ሚ + ገብ + ኣ -> የሚገባ, ይ + ግብ + ኣ -> ይግባ. Ruled 2026-10-29.
# One row serves both, so the search detaches the ኣ once. On a noun ኣ is
# never a suffix of its own: ስራ is not the "definite" of ስር.
STEM_VOWEL = "ኣ"
# The accusative on a relative verb, used as a noun: ያለው + ን -> ያለውን,
# IRIT የ_አል_ው_ን. It takes the case position, after the definite ው, which
# on a verb is the object position. See _compatible().
_RELATIVE_CASE_ROW = SuffixShape("ን", VERB, "ን", CASE)

# The Ge'ez construct state: the noun's last consonant takes ኧ (-ä), linking
# it to the next noun as "X of Y": ቤት + ኧ -> ቤተ (ቤተ ክርስቲያን). Ruled
# 2026-10-30. A perfective verb has the same shape (ሰበረ), so it attaches
# only to an entry marked takes_construct, and alone. See _analyses().
CONSTRUCT_MARK = "ኧ"
_CONSTRUCT_ROW = SuffixShape(CONSTRUCT_MARK, NOUN, "", CONSTRUCT, carrier_order=1, restore_orders=(6,))

# Order 8 is the "wa" form of a family (ሏ, ቷ). A labiovelar's order-4 letter
# (ቋ, ጓ, ኳ) is the same sound for the velars.
_LABIALIZED_ORDER = 8
_LABIOVELAR_WA = 4

_STEM_VOWEL_ROW = SuffixShape(
    STEM_VOWEL, VERB, "", SUBJECT, carrier_order=4, restore_orders=(6,),
    rule=RULE_AMBIGUOUS_FUSION, aspects=GERUND + (IMPERFECTIVE[0], JUSSIVE[0]),
)


# The suffix inventory. Rows are tried in order; the lexicon and the
# ranking choose among the analyses they produce.
SUFFIX_TABLE = (
    # ን is the case ending on a noun and the 1pl object on a verb (ሰጠን).
    _plain("ን", ANY, CASE, OBJECT),
    _RELATIVE_CASE_ROW,
    _plain("ም", ANY, ALSO),
    _plain("ና", ANY, AND),
    # The 3fs subject (ሄደች). On a non-verb it produced only wrong
    # analyses (ማርች -> ማር + ች), so it is verb-only.
    _plain("ች", VERB, PLURAL, SUBJECT, aspects=_PERFECTIVE),
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
    _plain("ሁ", VERB, SUBJECT, after_orders=(4,), aspects=_PERFECTIVE),
    _plain("ኝ", VERB, OBJECT, after_orders=_VOWEL_ORDERS, aspects=_PERFECTIVE),
    # The plural ኣት, as IRIT: ሳምንት + ኣት -> ሳምንታት, ቀን + ኣት -> ቀናት; after
    # a vowel with a ያ glide: ጊዜያት. On a verb ኣት is the 3fs object.
    _fused("ኣት", NOUN, PLURAL, 4, "ት", (6,)),
    _plain("ያት", NOUN, PLURAL, after_orders=(3, 5)),
    # "On, against" a person, the በ-series: አለ + በት -> አለበት "he must".
    *(_plain(m, VERB, OBJECT) for m in ("በት", "ባት", "ባቸው", "ብኝ", "ብህ", "ብሽ", "ብን", "ባችሁ")),
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
    _CONSTRUCT_ROW,
    # The adjective ending: ውጤት + ኣማ -> ውጤታማ.
    _fused("ኣማ", NOUN, ADJECTIVE, 4, "ማ", (6,)),
    # Her, as an object: ሰጠ + ኣት -> ሰጣት.
    _fused("ኣት", VERB, OBJECT, 4, "ት", (1,), aspects=_PERFECTIVE),
    # Perfective subjects delete the stem's last vowel: ሄደ + ኩ -> ሄድኩ.
    *(_fused(m, VERB, SUBJECT, 6, m, (1,), aspects=_PERFECTIVE) for m in ("ኩ", "ክ", "ሽ", "ን")),
    # Gerunds, on the reviewed gerund stem: ሆን + ኦ -> ሆኖ, ፈልግ + ኦ -> ፈልጎ,
    # ሆን + ኣ -> ሆና (3fs), ሰጥ + ቶ -> ሰጥቶ.
    _STEM_VOWEL_ROW,
    _fused("ኦ", VERB, SUBJECT, 7, "", (6,), aspects=GERUND),
    _plain("ቶ", VERB, SUBJECT, after_orders=(6,), aspects=GERUND),
    _plain("ታ", VERB, SUBJECT, after_orders=(6,), aspects=GERUND),
    # The 1s on a ቶ-gerund, the ት palatalized by the ኤ: ገብ + ቼ -> ገብቼ.
    _plain("ቼ", VERB, SUBJECT, after_orders=(6,), aspects=GERUND[:1]),
    # The ቶ-gerund's ት before another suffix, as IRIT ሰጥ_ት_አል, ሰጥ_ተ_ው:
    # ሰጥ + ት + ኣል -> ሰጥቷል, ሰጥ + ት + ው -> ሰጥተው. Never word-final.
    _plain("ት", VERB, SUBJECT, after_orders=(6,), aspects=GERUND[:1], bound=True),
    # The 1pl and 2nd-person gerunds after a linking ኧ: ሆን + ን -> ሆነን,
    # ሆን + ህ -> ሆነህ, ሆን + ሽ -> ሆነሽ; the 2pl ሆን + ኣችሁ -> ሆናችሁ. With ኣል
    # they make the present perfect: ሆነናል, ሆነሃል, ሆነሻል.
    *(_fused(m, VERB, OBJECT, 1, m, (6,), aspects=GERUND[:1]) for m in ("ን", "ህ", "ሽ")),
    _fused("ኣችሁ", VERB, OBJECT, 4, "ችሁ", (6,), aspects=GERUND[:1]),
    # The 1s gerund on the palatal stem: ሄጅ + ኤ -> ሄጄ.
    _fused("ኤ", VERB, SUBJECT, 5, "", (6,), aspects=(GERUND[1],)),
    # The 2fs on the palatal imperfective or jussive stem: ት + ሄጅ + ኢ.
    _fused("ኢ", VERB, SUBJECT, 3, "", (6,), aspects=(IMPERFECTIVE[1], JUSSIVE[1])),
    # The relative or object ው after a linking ኧ: የሚኖረው, IRIT
    # የ_ሚ_ኖር_ኧ_ው. The ኧ rides on the stem's last letter.
    _fused("ው", VERB, OBJECT, 1, "ው", (6,), aspects=_IMPERFECTIVE_OR_JUSSIVE + GERUND),
    # The auxiliary, fused onto the stem. On the imperfective stem it makes
    # the main-clause imperfective: ይ + ሆን + ኣል -> ይሆናል. On the gerund stem
    # it makes the present perfect, as IRIT: ሆን + ኣል + ች -> ሆናለች,
    # ሆን + ው + ኣል -> ሆነዋል. Before a person ending it is written ...ለ.
    _fused("ኣል", VERB, AUX, 4, "ል", (6,), aspects=_AUXILIARY_STEMS),
    _fused("ኣል", VERB, AUX, 4, "ለ", (6,), aspects=_AUXILIARY_STEMS, bound=True),
    # The 3ms present perfect: the gerund's ኦ and the auxiliary's ኣ fuse
    # into the order-8 letter. ሆን + ኣል -> ሆኗል, IRIT ሆን_አል.
    _fused("ኣል", VERB, AUX, 8, "ል", (6,), aspects=GERUND[:1]),
    # Person endings after the auxiliary, as IRIT: ኣል + ሁ, ኣል + ኡ.
    *(_plain(m, VERB, AUX_PERSON, after_orders=(1,), aspects=_AUXILIARY_STEMS)
      for m in ("ሁ", "ህ", "ሽ", "ች", "ን")),
    _fused("ኡ", VERB, AUX_PERSON, 2, "", (6,), aspects=_AUXILIARY_STEMS),
    _fused("ኣችሁ", VERB, AUX_PERSON, 4, "ችሁ", (6,), aspects=_AUXILIARY_STEMS),
)


# The rows the corpus analysis reads: those that attach to a lexicon entry
# itself. Rows that need a reviewed verb stem (the gerunds, the
# imperfective) say nothing about a corpus candidate, which has no stems,
# and the gerund ኣ restoring to order 6 would read ስራ as ስር + ኣ.
CORPUS_SUFFIXES = tuple(
    s for s in SUFFIX_TABLE if s.aspects in (None, _PERFECTIVE) and s is not _CONSTRUCT_ROW
)


# ── Prefixes ─────────────────────────────────────────────────────────────
#
# Prefixes take a fixed order too, as IRIT segments them:
#
#   PREPOSITION + SUBORDINATOR + NEGATION + PERSON + stem
#   የ + ማ + ይ + ሆነ (የማይሆን);  እንደ + ሚ + ሆነ;  ስ + ት + ሄደ
#
# ሚ ሲ ሊ ቢ እንዲ ሳይ and አይ already carry the 3ms person (ስ + ይ -> ሲ);
# ስ ል ብ እንድ and ማ need a person prefix after them.
PREPOSITION = 0
SUBORDINATOR = 1
NEGATION = 2
PERSON = 3


@dataclass(frozen=True)
class PrefixShape:
    """
    One prefix.

    Attributes:
        morpheme:     The prefix.
        kind:         ANY or VERB.
        slot:         PREPOSITION, SUBORDINATOR, NEGATION or PERSON.
        aspects:      For a verb, the stem aspects it attaches to, or None.
        person:       True if it marks the subject's person (ይ, or ሲ,
                      which contains it).
        needs_person: True if a person prefix must follow (ስ in ስትሄድ).
        pairs_with:   A suffix the word must also carry (ም for አል, አይ).
        readings:     What it means, or None where nothing is recorded.
        first_only:   True if it is written only as the word's first
                      prefix. After a subordinator the 1s has no prefix
                      and the 1pl is ን: ስንል, never ስእል or ስእንል.
        not_first:    True if another prefix must come before it: the
                      negative relative is የማይል, እንደማይል, never ማይል.
    """

    morpheme: str
    kind: str
    slot: int
    aspects: tuple[str, ...] | None = None
    person: bool = False
    needs_person: bool = False
    pairs_with: str | None = None
    readings: tuple[str, ...] | None = None
    first_only: bool = False
    not_first: bool = False


def _sub(morpheme, reading, person, not_first=False):
    return PrefixShape(morpheme, VERB, SUBORDINATOR, IMPERFECTIVE, person=person,
                       needs_person=not person, readings=(reading,), not_first=not_first)


def _person(morpheme, *readings, aspects=_IMPERFECTIVE_OR_JUSSIVE, first_only=False):
    return PrefixShape(morpheme, VERB, PERSON, aspects, person=True, readings=readings,
                       first_only=first_only)


# Prepositions attach to nouns and to verbs, a verb's perfective (የሆነ)
# or its imperfective after a subordinator (የሚሆን).
_PREPOSITIONS = ("ስለ", "ወደ", "እንደ", "እስከ", "በየ", "የ", "በ", "ከ", "ለ")

PREFIX_TABLE = (
    *(PrefixShape(p, ANY, PREPOSITION, (PERFECTIVE, *IMPERFECTIVE)) for p in _PREPOSITIONS),
    _sub("ሚ", "relative", True),
    _sub("ሲ", "when", True),
    _sub("ሊ", "purpose", True),
    _sub("ቢ", "if", True),
    _sub("እንዲ", "so_that", True),
    _sub("ሳይ", "without", True),
    _sub("ማ", "relative_negative", False, not_first=True),
    # The relative for subjects other than the 3rd person, as IRIT:
    # የ_ም_ፈልግ_ኧ_ው "what I want". "I" has no prefix of its own, so ም
    # counts as the person; ት or ን may follow (የምትሆኑ, የምንሄድ).
    _sub("ም", "relative", True, not_first=True),
    _sub("ስ", "when", False),
    _sub("ል", "purpose", False),
    _sub("ብ", "if", False),
    _sub("እንድ", "so_that", False),
    PrefixShape(NEGATIVE_PREFIX, VERB, NEGATION, (PERFECTIVE,), pairs_with=NEGATIVE_SUFFIX,
                readings=("negative",)),
    PrefixShape("አይ", VERB, NEGATION, IMPERFECTIVE, person=True, pairs_with=NEGATIVE_SUFFIX,
                readings=("negative", "subject_3ms")),
    _person("ይ", "subject_3ms"),
    _person("ት", "subject_2ms", "subject_3fs"),
    _person("እ", "subject_1s", first_only=True),
    _person("እን", "subject_1pl", first_only=True),
    _person("ን", "subject_1pl"),
    _person("ል", "subject_1s", aspects=JUSSIVE),
)

# What the corpus analysis peels. It has no categories, so it takes only
# the prefixes that can precede any word, plus the negative it always
# peeled.
CORPUS_PREFIXES = (*_PREPOSITIONS, NEGATIVE_PREFIX)


def strip_prefixes(word: str, prefixes=CORPUS_PREFIXES) -> set[str]:
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


def detach_prefix(word: str, prefixes=CORPUS_PREFIXES) -> list[tuple[str, str]]:
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


# A stem's initial glottal አ fuses with a prefix whose last letter is a
# bare consonant or at order 1: that letter takes the vowel ኣ (order 4).
# ይ + አውቅ -> ያውቅ, አል + አወቀ -> አላወቀ, የ + አለ -> ያለ. Ruled 2026-10-30,
# as IRIT writes them (ይ_አውቅ_አል, የ_አል_ኧ_ው).
_FUSING_ORDERS = (1, 6)
_FUSED_ORDER = 4
_GLOTTAL_A = "አ"
# A preposition also drops a stem's initial እ, keeping its own vowel:
# በ + እዚህ -> በዚህ, የ + እኔ -> የኔ. As IRIT (በ_እዚህ). Ruled 2026-10-30.
_GLOTTAL_E = "እ"
_PREPOSITION_ORDER = 1


# Cached: it depends on the prefix alone, and the search asks for every
# prefix of the table at every node.
@lru_cache(maxsize=None)
def _fused_form(prefix: str) -> str | None:
    """How a prefix is written before a stem's initial አ, or None."""
    last = prefix[-1]
    if fidel_order(last) not in _FUSING_ORDERS:
        return None
    return prefix[:-1] + to_order(last, _FUSED_ORDER)


_SPELLINGS: dict[int, tuple] = {}


def _prefix_spellings(prefixes) -> tuple[str, ...]:
    """Every way a prefix of this table can be written, for str.startswith."""
    cached = _SPELLINGS.get(id(prefixes))
    if cached is None or cached[0] is not prefixes:
        spellings = {p.morpheme for p in prefixes}
        spellings.update(f for f in (_fused_form(p.morpheme) for p in prefixes) if f)
        cached = (prefixes, tuple(sorted(spellings)))
        _SPELLINGS[id(prefixes)] = cached
    return cached[1]


def detach_prefix_shapes(word: str, prefixes=PREFIX_TABLE) -> list[tuple[str, PrefixShape, str]]:
    """
    Prefix rows removable from a surface form, with the remainder.

    A prefix is removed as written, or fused: in its fused form, which gives
    the stem back its initial አ (see _fused_form), or as a preposition that
    dropped the stem's initial እ (በዚህ = በ + እዚህ).

    Returns:
        (remainder, row, fusion) for every match. fusion is "" when the
        prefix was written as is, else the letter it absorbed: "አ" or "እ".

    Examples:
        >>> [(r, p.morpheme) for r, p, f in detach_prefix_shapes("የሚሆን") if not f]
        [('ሚሆን', 'የ')]
        >>> ("አውቃል", "ይ", "አ") in [(r, p.morpheme, f) for r, p, f in detach_prefix_shapes("ያውቃል")]
        True
        >>> ("እዚህ", "በ", "እ") in [(r, p.morpheme, f) for r, p, f in detach_prefix_shapes("በዚህ")]
        True
    """
    found = []
    for spelling, p, fusion in _prefix_index(prefixes).get(word[:1], ()):
        if word.startswith(spelling) and len(word) > len(spelling):
            found.append((fusion + word[len(spelling):], p, fusion))
    return found


_PREFIX_INDEXES: dict[int, tuple] = {}


def _prefix_index(prefixes) -> dict[str, list]:
    """
    Every way each prefix of a table is written, by first letter.

    Each item is (spelling, row, fusion), in table order: the prefix as
    written; for a preposition, the same spelling having dropped a stem's
    እ; and its fused spelling before a stem's አ. Testing only the items
    for a word's first letter keeps the search fast.
    """
    cached = _PREFIX_INDEXES.get(id(prefixes))
    if cached is None or cached[0] is not prefixes:
        index: dict[str, list] = {}
        for p in prefixes:
            index.setdefault(p.morpheme[0], []).append((p.morpheme, p, ""))
            if p.slot == PREPOSITION and fidel_order(p.morpheme[-1]) == _PREPOSITION_ORDER:
                index[p.morpheme[0]].append((p.morpheme, p, _GLOTTAL_E))
            fused = _fused_form(p.morpheme)
            if fused:
                index.setdefault(fused[0], []).append((fused, p, _GLOTTAL_A))
        cached = (prefixes, index)
        _PREFIX_INDEXES[id(prefixes)] = cached
    return cached[1]


# Per-table index of the rows by the letter they end on. Keyed by the
# table's id; the table itself is kept in the value so the id stays valid.
_SUFFIX_INDEXES: dict[int, tuple] = {}


def _rows_ending(word: str, suffixes) -> list:
    """
    The rows of a table that can match the end of a word, in table order.

    A row with a tail can match only if the word ends with the tail's last
    letter. A purely fused row (ኡ, ኦ) can match only if the word's last
    letter is at the row's carrier order. Testing just these rows is what
    keeps the search fast as the table grows.
    """
    cached = _SUFFIX_INDEXES.get(id(suffixes))
    if cached is None or cached[0] is not suffixes:
        by_last: dict[str, list] = {}
        by_order: dict[int, list] = {}
        for position, shape in enumerate(suffixes):
            if shape.tail:
                by_last.setdefault(shape.tail[-1], []).append((position, shape))
            else:
                by_order.setdefault(shape.carrier_order, []).append((position, shape))
        cached = (suffixes, by_last, by_order)
        _SUFFIX_INDEXES[id(suffixes)] = cached
    _, by_last, by_order = cached
    last = word[-1]
    order = fidel_order(last)
    rows = by_last.get(last, []) + by_order.get(order, [])
    if order == _LABIOVELAR_WA and is_labiovelar(last):
        # A velar's "wa" (ኳ) carries the order-8 rows too; see detach_suffix_shapes.
        rows += by_order.get(_LABIALIZED_ORDER, [])
    return [shape for _, shape in sorted(rows, key=lambda item: item[0])]


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

    if not word:
        return found
    for shape in _rows_ending(word, suffixes):
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
        if (
            shape.carrier_order == _LABIALIZED_ORDER
            and is_labiovelar(carrier)
            and fidel_order(carrier) == _LABIOVELAR_WA
        ):
            # A velar writes its "wa" with the labiovelar (ጓ), not the
            # order-8 letter (ጏ): ፈልግ + ኦ + ኣል -> ፈልጓል. Restore from the
            # plain consonant, since a labiovelar stays in its own block.
            carrier = bare_consonant(carrier)
        elif fidel_order(carrier) != shape.carrier_order:
            continue
        for order in shape.restore_orders:
            try:
                add(rest[:-1] + to_order(carrier, order), shape)
            except ValueError:
                # A labiovelar has no letter at some orders.
                continue
    return found


def detach_suffix(word: str, suffixes=CORPUS_SUFFIXES) -> list[tuple[str, str]]:
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


def detach_suffix_with_rule(word: str, suffixes=CORPUS_SUFFIXES) -> list[tuple[str, str, str]]:
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


def strip_suffixes(word: str, suffixes=CORPUS_SUFFIXES) -> set[str]:
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

# What each suffix means, as (readings after a non-verb, readings after a
# verb). Reviewed by the native speaker 2026-10-29. Every suffix has an
# entry, unambiguous ones included, so "no meaning recorded" never looks
# like "meaning unknown". An empty tuple means the suffix does not attach
# to that kind of stem.
_READINGS = {
    "ኡ": (("definite", "possessive_3ms"), ("subject_3pl",)),
    "ው": (("definite", "possessive_3ms"), ("object_3ms", "subject_3pl", "definite")),
    "ኣ": ((), ("gerund_3fs",)),
    "ኦ": ((), ("gerund_3ms",)),
    "ቶ": ((), ("gerund_3ms",)),
    "ታ": ((), ("gerund_3fs",)),
    "ቼ": ((), ("gerund_1s",)),
    "ኣማ": (("adjectivizer",), ()),
    "ኧ": (("construct",), ()),
    "ማ": (("adjectivizer", "topic"), ()),
    "ዋ": (("definite_f", "possessive_3fs"), ()),
    "ኤ": (("possessive_1s",), ("gerund_1s",)),
    "ኢ": ((), ("subject_2fs",)),
    "ኣል": ((), ("auxiliary",)),
    "ት": ((), ("gerund",)),
    "ዬ": (("possessive_1s",), ()),
    "ህ": (("possessive_2ms",), ("subject_2ms", "object_2ms")),
    "ሽ": (("possessive_2fs",), ("subject_2fs", "object_2fs")),
    "ዎ": (("possessive_2polite",), ("object_2polite",)),
    "ኣችን": (("possessive_1pl",), ()),
    "ችን": (("possessive_1pl",), ()),
    "ያችን": (("possessive_1pl",), ()),
    "ኣችሁ": (("possessive_2pl",), ("subject_2pl", "object_2pl")),
    "ችሁ": (("possessive_2pl",), ("subject_2pl", "object_2pl")),
    "ኣቸው": (("possessive_3pl",), ("object_3pl",)),
    "ቸው": (("possessive_3pl",), ("object_3pl",)),
    "ኩ": ((), ("subject_1s",)),
    "ሁ": ((), ("subject_1s",)),
    "ክ": ((), ("subject_2ms",)),
    "ኝ": ((), ("object_1s",)),
    "ኣት": (("plural",), ("object_3fs",)),
    "ያት": (("plural",), ()),
    "በት": ((), ("applicative_3ms",)),
    "ባት": ((), ("applicative_3fs",)),
    "ባቸው": ((), ("applicative_3pl",)),
    "ብኝ": ((), ("applicative_1s",)),
    "ብህ": ((), ("applicative_2ms",)),
    "ብሽ": ((), ("applicative_2fs",)),
    "ብን": ((), ("applicative_1pl",)),
    "ባችሁ": ((), ("applicative_2pl",)),
    "ን": (("accusative",), ("subject_1pl", "object_1pl")),
    "ች": ((), ("subject_3fs",)),
    "ኦች": (("plural",), ()),
    "ዎች": (("plural",), ()),
    "ም": (("also",), ("also",)),
    "ና": (("and",), ("and",)),
}

# The negative prefix's reading, and the one ም takes beside it.
_NEGATIVE = ("negative",)
# ኣ on the imperfective or jussive stem; on the gerund stem it is the 3fs.
_STEM_VOWEL_READING = ("stem_vowel",)
# ኣል on the gerund stem; on the imperfective stem it is the auxiliary.
_PRESENT_PERFECT_READING = ("present_perfect",)


def suffix_readings(segmentation: "Segmentation", suffix: str) -> tuple[str, ...] | None:
    """
    What a suffix means in a given analysis.

    The stem's category picks the column: ኡ is the definite or "his" on a
    noun and the 3pl subject on a verb. ም needs the whole analysis, since
    it is the negative only on a verb that also carries the prefix አል.

    Args:
        segmentation: The analysis the suffix belongs to.
        suffix:       One of its suffixes.

    Returns:
        The readings, or None when the stem's category is unknown.

    Examples:
        >>> suffix_readings(Segmentation("ቤቱ", "ቤት", suffixes=("ኡ",), category="noun"), "ኡ")
        ('definite', 'possessive_3ms')
        >>> suffix_readings(Segmentation("ሆኑ", "ሆነ", suffixes=("ኡ",), category="verb"), "ኡ")
        ('subject_3pl',)
        >>> suffix_readings(Segmentation("ቤቱ", "ቤት", suffixes=("ኡ",)), "ኡ") is None
        True
    """
    category = segmentation.category
    if category is None:
        return None
    is_verb = category == VERB
    if is_verb and suffix == NEGATIVE_SUFFIX and (
        NEGATIVE_PREFIX in segmentation.prefixes or segmentation.stem == NEGATIVE_EXISTENTIAL
    ):
        return _NEGATIVE
    if is_verb and suffix == STEM_VOWEL and segmentation.aspect in _IMPERFECTIVE_OR_JUSSIVE:
        return _STEM_VOWEL_READING
    if is_verb and suffix == AUXILIARY and segmentation.aspect in GERUND:
        return _PRESENT_PERFECT_READING
    if is_verb and _RELATIVE_CASE_ROW in segmentation.suffix_shapes and suffix == _RELATIVE_CASE_ROW.morpheme:
        return _READINGS[suffix][0]
    return _READINGS[suffix][1 if is_verb else 0]


def _prefix_readings(prefix: str, segmentation: "Segmentation") -> tuple[str, ...] | None:
    """The readings of a prefix row that fits this analysis's aspect."""
    for row in PREFIX_TABLE:
        if row.morpheme != prefix:
            continue
        if row.kind == VERB and segmentation.category != VERB:
            continue
        if row.aspects is None or segmentation.aspect in row.aspects:
            return row.readings
    return None


@dataclass(frozen=True)
class Morpheme:
    """
    One morpheme with its position and its readings.

    text is always a real Amharic string, so Levels 3, 2 and 1 and the
    BPE layer can consume it directly. The readings are kept beside it
    rather than folded into the string, because ኡ-def would not be
    decomposable as Amharic.

    Attributes:
        text:     The morpheme.
        kind:     PREFIX, INDEPENDENT, INFIX or SUFFIX.
        readings: What it can mean here, from suffix_readings() for a
                  suffix and ("negative",) for the prefix አል. None for
                  the stem, the infix, the other prefixes, and any
                  suffix whose stem has no category yet.
    """

    text: str
    kind: str
    readings: tuple[str, ...] | None = None


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
        category: The stem entry's category, which picks each suffix's
                  readings. None when unfilled.
        aspect:   For a verb, which of its stems the word is built on:
                  PERFECTIVE, or one of IMPERFECTIVE, JUSSIVE, GERUND.
                  The stem is still reported in its lexicon form
                  (ይሆናል = ይ + ሆነ + ኣል, aspect imperfective). None for
                  a non-verb.
        suffix_shapes: The table row of each suffix, parallel to
                  suffixes. A fused row and a plain one cut the written
                  word in different places, so the BPE level needs it.
                  Not part of equality: two analyses with the same
                  morphemes are the same analysis.
        prefix_fused: The letter each prefix absorbed, or "", parallel
                  to prefixes: "አ" in ያለው = የ + አለ + ው, "እ" in በዚህ =
                  በ + እዚህ. Not part of equality, for the same reason.
    """

    word: str
    stem: str
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    infix: str | None = None
    known: bool = True
    category: str | None = None
    aspect: str | None = None
    suffix_shapes: tuple[SuffixShape, ...] = field(default=(), compare=False, repr=False)
    prefix_fused: tuple[str, ...] = field(default=(), compare=False, repr=False)

    @property
    def parts(self) -> list[Morpheme]:
        """
        All morphemes with their position and readings, in order.

        Examples:
            >>> seg = Segmentation("ቤቱ", "ቤት", suffixes=("ኡ",), category="noun")
            >>> [(p.text, p.readings) for p in seg.parts]
            [('ቤት', None), ('ኡ', ('definite', 'possessive_3ms'))]
        """
        parts = [Morpheme(p, PREFIX, _prefix_readings(p, self)) for p in self.prefixes]
        parts.append(Morpheme(self.stem, INDEPENDENT))
        if self.infix:
            parts.append(Morpheme(self.infix, INFIX))
        parts.extend(
            Morpheme(s, SUFFIX, suffix_readings(self, s)) for s in self.suffixes
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
    # The table row of each prefix, parallel to prefixes.
    prefix_shapes: tuple[PrefixShape, ...] = ()
    # The letter each prefix absorbed from what follows, or "": "አ" in
    # ያለ = የ + አለ, "እ" in በዚህ = በ + እዚህ.
    prefix_fused: tuple[str, ...] = ()


def _kind_allows(kind: str, category: str | None) -> bool:
    """Whether a row of this kind attaches to a stem of this category."""
    if kind == ANY:
        return True
    if category is None:
        return False
    return (category == VERB) == (kind == VERB)


def _compatible(node: _Node, stem: str, category: str | None, aspect: str | None) -> bool:
    """
    Whether every affix of a finished analysis fits its stem.

    - Each row's kind admits the category, and on a verb each row's
      aspects admit the stem's aspect.
    - Suffixes are in strictly increasing position. Prefixes are too,
      except that prepositions may stack.
    - A bound suffix is followed by another, a prefix that needs a
      person is followed by one, and a paired prefix (አል, አይ) has its ም.
      እ and እን come first; ማ never does.
    - Per aspect: an imperfective needs a person, and after a
      preposition a subordinator as well (የሚሆን, not የይሆናል). A jussive
      needs a person prefix and takes no other. A gerund takes no prefix
      and its first suffix is a gerund ending (ሆኖ, not ሆን). In its present perfect, a
      person ending follows the auxiliary only as ች or ሁ.
    - Only an adjective takes the infix.
    - The stem vowel ኣ attaches only to a verb whose perfective ends in
      ኣ, and that verb's imperfective or jussive stem never ends the
      word: ይገባ, ይገባል, ይገቡ, never ይገብ.
    """
    shapes, prefixes = node.shapes, node.prefix_shapes
    is_verb = category == VERB
    if not all(_kind_allows(s.kind, category) for s in shapes):
        return False
    if not all(_kind_allows(p.kind, category) for p in prefixes):
        return False
    if is_verb and any(r.aspects is not None and aspect not in r.aspects for r in (*shapes, *prefixes)):
        return False

    positions = [s.position(category) for s in shapes]
    if any(a >= b for a, b in zip(positions, positions[1:])):
        return False
    slots = [p.slot for p in prefixes]
    if any(a > b or (a == b and a != PREPOSITION) for a, b in zip(slots, slots[1:])):
        return False
    if any(p.first_only for p in prefixes[1:]):
        return False
    if prefixes and prefixes[0].not_first:
        return False
    if shapes and shapes[-1].bound:
        return False
    # The accusative on a verb follows its definite ው, on a relative verb.
    for i, shape in enumerate(shapes):
        if shape == _RELATIVE_CASE_ROW and (
            i == 0
            or shapes[i - 1].morpheme != "ው"
            or not any(p.slot in (PREPOSITION, SUBORDINATOR) for p in prefixes)
        ):
            return False
    for i, p in enumerate(prefixes):
        if p.needs_person and not any(q.slot == PERSON for q in prefixes[i + 1:]):
            return False
        if p.pairs_with and p.pairs_with not in node.suffixes:
            return False

    # The infix is the plural-adjective copy (ትላልቅ). Elsewhere the same
    # letters are something else: ይመስላል is ይ + መሰለ + ኣል, መንግስታት is the
    # plural ኣት, አባባል "saying" is a word of its own.
    if node.infix and category != _INFIX_CATEGORY:
        return False

    # A preposition fuses with an initial አ only on a noun of three letters
    # or more: on a two-letter stem the match is an accident (ካሳ is not
    # ከ + አሳ, ካፌ is not ከ + አፍ + ኤ), while ያገር, ባንድ, እስካሁን are real.
    # Verbs are exempt (ያየ). The እ fusion has the same limit, except for
    # the personal pronouns (የኔ, ለኛ), and never fuses onto a preposition or
    # a conjunction (ለንደን is not ለ + እንደ + ን). No word stacks more than two
    # prepositions (የለበሱ is not የ + ለ + በ + እሱ).
    if not is_verb and _GLOTTAL_A in node.prefix_fused and len(stem) < _MIN_FUSED_NOUN:
        return False
    if _GLOTTAL_E in node.prefix_fused and (
        category in _UNFUSABLE_CATEGORIES
        or (len(stem) < _MIN_FUSED_NOUN and stem not in _SHORT_FUSABLE)
    ):
        return False
    if sum(1 for p in prefixes if p.slot == PREPOSITION) > _MAX_PREPOSITIONS:
        return False

    if not is_verb:
        return True
    if aspect in _IMPERFECTIVE_OR_JUSSIVE:
        weak_final = fidel_order(stem[-1]) == 4
        if not shapes and weak_final:
            return False
        if _STEM_VOWEL_ROW in shapes and not weak_final:
            return False
    if aspect in IMPERFECTIVE:
        if not any(p.person for p in prefixes):
            return False
        # The auxiliary ends a main clause: ይሆናል, never ሲሆናል.
        if any(p.slot == SUBORDINATOR for p in prefixes) and any(
            s.morpheme == AUXILIARY for s in shapes
        ):
            return False
        if any(p.slot == PREPOSITION for p in prefixes) and not any(p.slot == SUBORDINATOR for p in prefixes):
            return False
    elif aspect in JUSSIVE:
        # A person prefix is required. A jussive stem alone would be the
        # imperative, but it collides with ordinary words (ድረስ "until",
        # ስር "under"), so it is not guessed.
        if not prefixes or any(p.slot != PERSON for p in prefixes):
            return False
    elif aspect in GERUND:
        # The first suffix is a gerund ending (ኦ, ኣ, ቶ, ው, ን, ...); an object
        # or the auxiliary never sits directly on the gerund stem.
        if prefixes or not shapes or shapes[0].aspects is None:
            return False
        if not _is_present_perfect_shape(shapes, category, aspect):
            return False
    return True


# The present perfect's persons are ሆኛለሁ ሆነሃል ሆነሻል ሆኗል ሆናለች ሆነናል
# ሆናችኋል ሆነዋል. After the auxiliary only these endings are written, each
# on its own stem: ች (3fs) on the gerund stem, ሁ (1s) on the palatal one.
_PERFECT_ENDINGS = {("ች", GERUND[0]), ("ሁ", GERUND[1])}


def _is_present_perfect_shape(shapes, category, aspect) -> bool:
    """
    On a gerund stem, whether the auxiliary and its endings are well placed.

    A person ending follows the auxiliary, and only ች or ሁ does. The
    auxiliary directly on the stem (the order-4 rows) needs that ending
    after it: ሆናለች, never ሆናል. Elsewhere the person comes before the
    auxiliary (ሆነ + ው + ኣል), or is fused into it (ሆኗል).
    """
    for i, shape in enumerate(shapes):
        if shape.position(category) == AUX_PERSON:
            if i == 0 or shapes[i - 1].morpheme != AUXILIARY:
                return False
            if (shape.morpheme, aspect) not in _PERFECT_ENDINGS:
                return False
    first = shapes[0]
    if first.morpheme == AUXILIARY and first.carrier_order == 4 and not first.bound:
        return False
    return True


def verb_stems(lexicon: dict) -> dict[str, tuple[tuple[str, str], ...]]:
    """
    Every reviewed verb stem, mapped to its (verb, aspect) pairs.

    A word anchors on these as well as on the entries themselves, so
    ይሆናል finds ሆነ through its imperfective stem ሆን. One stem can serve
    several aspects (ሆን is ሆነ's imperfective and its gerund).

    Args:
        lexicon: Word to lexicon entry.

    Returns:
        Stem to a tuple of (verb, aspect).

    Examples:
        >>> verb_stems({"ሆነ": {"category": "verb", "stems": {"jussive": "ሁን"}}})
        {'ሁን': (('ሆነ', 'jussive'),)}
    """
    index: dict[str, list[tuple[str, str]]] = {}
    for word, entry in lexicon.items():
        if entry.get("category") != VERB:
            continue
        for aspect, stem in (entry.get("stems") or {}).items():
            index.setdefault(stem, []).append((word, aspect))
    return {stem: tuple(pairs) for stem, pairs in index.items()}


def _family(char: str) -> int:
    return ord(char) - ord(char) % 8


def anchor_starts(lexicon: dict, stems: dict) -> frozenset:
    """
    How every possible anchor begins: (first letter, family of the second).

    A prefix is stripped from the front, and nothing later changes the
    front of the text: fusion rewrites only the last letter, within its
    family, and the infix sits after the first letter. So a text left by
    a prefix can reach an anchor only if it begins as one does, or begins
    with another prefix, and the search drops the ones that cannot. A one-letter anchor is recorded
    with None for the second.

    Args:
        lexicon: Word to lexicon entry.
        stems:   verb_stems(lexicon).

    Returns:
        The set of (letter, family or None) pairs.
    """
    return frozenset(
        (a[0], _family(a[1]) if len(a) > 1 else None) for a in (*lexicon, *stems) if a
    )


def _can_reach_an_anchor(text: str, starts: frozenset) -> bool:
    if (text[0], None) in starts:
        return True
    if len(text) > 1 and (text[0], _family(text[1])) in starts:
        return True
    # An infix right after the first letter hides the anchor's second.
    return len(text) > 2 and (text[0], _family(text[2])) in starts


def _analyses(
    word: str,
    lexicon: dict,
    stems: dict,
    starts: frozenset,
    max_prefixes: int,
    max_suffixes: int,
    prefixes,
    suffixes,
) -> list[Segmentation]:
    """Every analysis whose stem is a lexicon entry or a verb stem and whose affixes fit it."""
    found: dict[Segmentation, None] = {}
    seen: set[tuple] = set()
    queue = [_Node(word)]

    while queue:
        node = queue.pop()
        key = (node.text, node.prefix_shapes, node.infix, node.shapes)
        if key in seen:
            continue
        seen.add(key)

        anchors = []
        if node.text in lexicon:
            category = lexicon[node.text].get("category")
            anchors.append((node.text, category, PERFECTIVE if category == VERB else None))
        for verb, aspect in stems.get(node.text, ()):
            anchors.append((verb, VERB, aspect))
        for stem, category, aspect in anchors:
            if _CONSTRUCT_ROW in node.shapes and not (
                node.shapes == (_CONSTRUCT_ROW,) and lexicon.get(stem, {}).get("takes_construct")
            ):
                continue
            if _compatible(node, stem, category, aspect):
                found.setdefault(
                    Segmentation(
                        word=word,
                        stem=stem,
                        prefixes=node.prefixes,
                        suffixes=node.suffixes,
                        infix=node.infix,
                        category=category,
                        aspect=aspect,
                        suffix_shapes=node.shapes,
                        prefix_fused=node.prefix_fused,
                    )
                )

        # The infix comes off last, from the bare stem: its copy and the
        # consonant it copies are both stem letters. Detached earlier, it
        # could take a prefix's or a suffix's letter (ከታታላቅ, በጎቻችንን).
        if node.infix is not None:
            continue

        if node.depth_prefix < max_prefixes:
            for rest, prefix, fused in detach_prefix_shapes(node.text, prefixes):
                # Keep a text that can still reach an anchor, or that
                # begins with a further prefix, as written (የ + ሚሆን) or
                # fused (የ + ሚ + ያውቅ). A dropped እ is a stem's, never a
                # further prefix's, so that text must reach an anchor.
                if not _can_reach_an_anchor(rest, starts) and (
                    fused == _GLOTTAL_E or not rest.startswith(_prefix_spellings(prefixes))
                ):
                    continue
                queue.append(
                    _Node(
                        rest,
                        node.prefixes + (prefix.morpheme,),
                        node.suffixes,
                        node.infix,
                        node.depth_prefix + 1,
                        node.depth_suffix,
                        node.shapes,
                        node.prefix_shapes + (prefix,),
                        node.prefix_fused + (fused,),
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
                        node.prefix_shapes,
                        node.prefix_fused,
                    )
                )

        # One infix at most: it is an internal modification of a single
        # stem, not something that stacks.
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
                    node.prefix_shapes,
                    node.prefix_fused,
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

    Fewest morphemes, then fewest fused prefixes, then whether every
    affix is attested for that entry, then an analysis anchored on an
    entry itself before one anchored on a derived verb stem, then
    longest stem, then the morpheme sequence itself so the order is
    deterministic. See the module docstring for why.

    The second criterion prefers the word as written: የድል is የ + ድል "of
    victory", not የ + እድል "of chance", though the fused stem is longer.

    The third criterion keeps ባለው as ባለ + ው: through the stem ባል, ተባለ
    + ው has as many morphemes and a longer stem as reported, though the
    stem actually matched is shorter.

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
            sum(1 for f in a.prefix_fused if f),
            not _attested(a, lexicon),
            a.aspect not in (None, PERFECTIVE),
            -len(a.stem),
            a.morphemes,
        ),
    )


def segment_word(
    word: str,
    lexicon: dict,
    max_prefixes: int = 3,
    max_suffixes: int = 4,
    prefixes=PREFIX_TABLE,
    suffixes=SUFFIX_TABLE,
    stems: dict | None = None,
    starts: frozenset | None = None,
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
        prefixes:     Prefix table.
        suffixes:     Suffix table.
        stems:        verb_stems(lexicon), precomputed by a caller that
                      segments many words; built here when omitted.
        starts:       anchor_starts(lexicon, stems), likewise.

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

    if stems is None:
        stems = verb_stems(lexicon)
    if starts is None:
        starts = anchor_starts(lexicon, stems)
    found = _analyses(word, lexicon, stems, starts, max_prefixes, max_suffixes, prefixes, suffixes)
    if not found:
        return Segmentation(word=word, stem=word, known=False)
    return rank_analyses(found, lexicon)[0]


def segment_all(
    word: str,
    lexicon: dict,
    max_prefixes: int = 3,
    max_suffixes: int = 4,
    prefixes=PREFIX_TABLE,
    suffixes=SUFFIX_TABLE,
    stems: dict | None = None,
    starts: frozenset | None = None,
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
        prefixes:     Prefix table.
        suffixes:     Suffix table.
        stems:        verb_stems(lexicon), or None to build it here.
        starts:       anchor_starts(lexicon, stems), or None likewise.

    Returns:
        Ranked analyses. Empty when the word has no lexicon anchor.
    """
    if stems is None:
        stems = verb_stems(lexicon)
    if starts is None:
        starts = anchor_starts(lexicon, stems)
    found = _analyses(word, lexicon, stems, starts, max_prefixes, max_suffixes, prefixes, suffixes)
    return rank_analyses(found, lexicon)
