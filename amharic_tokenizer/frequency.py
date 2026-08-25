"""
Frequency analysis and lexicon candidate generation.

Runs on the output of the corpus pipeline (corpus.txt) and produces the
ranked candidate list that the independent word lexicon is built from.

── The bootstrapping signal ─────────────────────────────────────────────

The core constraint of the language does the work here: every Amharic
word is exactly one independent morpheme (ዋና ቃል) plus zero or more
dependent morphemes (ቅጥያ), and dependent morphemes never appear alone in
a corpus. Independent ones can.

So a surface form is evidence about its own stem. Strip one affix from
ቤቶች and you get ቤት; ቤት is attested standalone 119,497 times, so the
decomposition is supported by the corpus rather than assumed. A stem
whose remainder never appears alone is not a stem.

── Why stripping is not string removal ──────────────────────────────────

The plural suffix ኦች does not concatenate. ቤት + ኦች is written ቤቶች: the
ኦ vowel fuses into the final consonant, turning ት (order 6) into ቶ
(order 7). Removing the literal characters "ኦች" from ቤቶች yields nothing.

Inverting the fusion needs fidel arithmetic, which the Ethiopic block
makes easy: families are 8-aligned, so order = codepoint % 8 and the
bare consonant (order 6) is always family_base + 5. ቶ -> ት, ቱ -> ት.

Three suffix shapes are handled:
  - plain concatenation      ቤትን  -> ቤት    (drop ን)
  - fusion + consonant       ቤቶች  -> ቤት    (ቶ -> ት, drop ች)
  - fusion alone             ቤቱ   -> ቤት    (ቱ -> ት)

── Ranking ──────────────────────────────────────────────────────────────

Candidates are ranked by productivity — how many distinct affixed forms
the corpus attests for a stem — rather than by raw frequency alone. A
genuine independent word appears with many affixes (ቤት, ቤቶች, ቤቱ, ቤትን,
የቤት, በቤት). A frequent string that happens to be a substring of other
words does not.

Everything here produces CANDIDATES. Per spec section 3.2 they require
native speaker validation before entering the lexicon.
"""

import json
import re
from collections import Counter

from amharic_tokenizer.levels import (
    fidel_order,
    level1_root,
    to_order6,
)
from amharic_tokenizer.morphology import (
    DEFAULT_PLAIN_SUFFIXES,
    DEFAULT_PREFIXES,
    strip_prefixes,
    strip_suffixes,
)

_ETHIOPIC_WORD = re.compile(r"^[ሀ-፿]+$")
_PUNCTUATION_CHARS = set("።፣፤፥፦፡?!#‹›")

# Vowel orders that a suffix can fuse onto a stem's final consonant.
# Order 7 (ኦ) marks the plural before ች; order 2 (ኡ) and order 4 (ኣ)
# mark the definite forms.
_FUSED_PLURAL_ORDER = 7
_FUSED_DEFINITE_ORDERS = (2, 4)


def is_ethiopic_word(token: str) -> bool:
    """
    True if the token is a pure Ethiopic word rather than punctuation,
    a numeral, or a tagged Latin/URL span.

    Examples:
        >>> is_ethiopic_word("ቤት")
        True
        >>> is_ethiopic_word("።")
        False
        >>> is_ethiopic_word("2010")
        False
    """
    if not token or token in _PUNCTUATION_CHARS:
        return False
    return bool(_ETHIOPIC_WORD.match(token))


def build_frequency_table(corpus_path: str, encoding: str = "utf-8") -> Counter:
    """
    Count Ethiopic word types in a corpus file, streaming.

    Args:
        corpus_path: Path to corpus.txt, one sentence per line.
        encoding:    File encoding.

    Returns:
        Counter mapping word type to occurrence count.

    Raises:
        FileNotFoundError: If corpus_path does not exist.
    """
    counts: Counter = Counter()
    with open(corpus_path, "r", encoding=encoding) as f:
        for line in f:
            counts.update(t for t in line.split() if is_ethiopic_word(t))
    return counts


def coverage_curve(
    counts: Counter, targets: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)
) -> dict:
    """
    How many word types are needed to cover a given share of all tokens.

    This is the empirical answer to open question 5 — the lexicon size
    required for 90/95/99% corpus coverage — measured over surface forms.
    A morpheme-aware lexicon needs fewer entries than this, because many
    surface forms share one stem.

    Args:
        counts:  Frequency table.
        targets: Coverage fractions to report.

    Returns:
        Dict mapping each target to the number of types required.
    """
    total = sum(counts.values())
    if total == 0:
        return {f"{t:.0%}": 0 for t in targets}

    result = {}
    remaining = list(targets)
    cumulative = 0
    for rank, (_, count) in enumerate(counts.most_common(), start=1):
        cumulative += count
        while remaining and cumulative / total >= remaining[0]:
            result[f"{remaining[0]:.0%}"] = rank
            remaining.pop(0)
        if not remaining:
            break
    for target in remaining:
        result[f"{target:.0%}"] = len(counts)
    return result


# Derivation rules, tracked so that policy can differ per rule.
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


def candidate_stems_with_rules(
    word: str, prefixes=DEFAULT_PREFIXES, suffixes=DEFAULT_PLAIN_SUFFIXES
) -> dict[str, str]:
    """
    Stems reachable from a surface form, tagged with the rule that
    produced each.

    The rule matters because the ambiguous order-4 fusion needs a
    frequency check that the unambiguous rules do not.

    Args:
        word:     Surface form.
        prefixes: Prefix inventory.
        suffixes: Plain suffix inventory.

    Returns:
        Dict mapping candidate stem to the RULE_* that derived it.
    """
    rules: dict[str, str] = {}

    for stem in strip_prefixes(word, prefixes):
        rules[stem] = RULE_PREFIX

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix):
            rules.setdefault(word[: -len(suffix)], RULE_SUFFIX)

    if len(word) >= 3 and word.endswith("ች"):
        carrier = word[-2]
        if fidel_order(carrier) == _FUSED_PLURAL_ORDER:
            rules.setdefault(
                word[:-2] + to_order6(carrier), RULE_PLURAL_FUSION
            )

    if len(word) >= 2:
        last = word[-1]
        order = fidel_order(last)
        if order in _FUSED_DEFINITE_ORDERS:
            rule = (
                RULE_AMBIGUOUS_FUSION if order == 4 else RULE_DEFINITE_FUSION
            )
            rules.setdefault(word[:-1] + to_order6(last), rule)

    rules.pop(word, None)
    return {stem: rule for stem, rule in rules.items() if stem}


def candidate_stems(
    word: str, prefixes=DEFAULT_PREFIXES, suffixes=DEFAULT_PLAIN_SUFFIXES
) -> set[str]:
    """
    All stems reachable from a surface form by removing one affix.

    One affix at a time, not recursively — each stripped form is itself
    a corpus type whose own decomposition is handled when it is reached.

    Args:
        word:     Surface form.
        prefixes: Prefix inventory.
        suffixes: Plain suffix inventory.

    Returns:
        Set of candidate stems, excluding the word itself.
    """
    return set(candidate_stems_with_rules(word, prefixes, suffixes))


# The one derivation whose morphology is unambiguous: nothing but a
# plural produces order-7 + ች, so a stem reached this way is an
# inflection whatever its frequency. Order-2 definite fusion does not
# qualify — ልዩ, ሙሉ, ጥሩ and ክፉ all end in order 2 without being the
# definite of anything.
_UNAMBIGUOUS_DERIVATIONS = (RULE_PLURAL_FUSION,)


def derived_stems(
    attested: dict[str, dict[str, str]],
    counts: Counter,
) -> set[str]:
    """
    Stems that are themselves an inflected form of another stem.

    A word can pass every standalone test and still be an inflection:
    ቤቱ is written without a space, is frequent, and takes further
    affixes of its own. What gives it away is that it appears in ቤት's
    form set. Keeping both would let Level 4 match ቤቱ as a whole stem
    and lose the definite marker.

    How much weight that membership carries depends on the rule that
    produced it, which is why the rule is tracked alongside each form.

    Plural fusion is unambiguous. No Amharic word ends in order-7 + ች
    by coincidence, so ልጆች is the plural of ልጅ and nothing else, and
    the derivation stands on its own.

    Every other rule also runs backwards over ordinary words: ሰላም
    decomposes to ሰላ + ም, ወንድም to ወንድ + ም, and ልዩ to ልይ + ኡ, none of
    which is how the word was formed. Frequency separates most of
    these. A real base outranks its own inflection — ስብሰባ stands alone
    12,787 times against ስብሰባው's 1,567 — while a spurious claimer is
    rarer than the word it claims, ልይ managing 72 against ልዩ's 26,881.
    So those derivations count only when the base is the more frequent
    of the pair.

    Order-4 fusion never counts. It is tagged ambiguous precisely
    because many native words simply end in an order-4 character, and
    discarding ቀና as the definite of ቀን would lose a real word.

    Args:
        attested: Stem to a dict of surface form to the RULE_* that
                  derived the stem from it.
        counts:   Frequency table, read for standalone counts.

    Returns:
        Set of stems to discard.
    """
    derived: set[str] = set()
    for base, forms in attested.items():
        for form, rule in forms.items():
            if form == base or form not in attested:
                continue
            if rule in _UNAMBIGUOUS_DERIVATIONS:
                derived.add(form)
            elif rule != RULE_AMBIGUOUS_FUSION and counts[form] <= counts[base]:
                derived.add(form)
    return derived


def generate_candidates(
    counts: Counter,
    min_stem_frequency: int = 10,
    min_surface_frequency: int = 5,
    min_distinct_forms: int = 2,
    min_standalone_ratio: float = 0.05,
    max_ambiguous_inflection_ratio: float = 0.5,
    min_stem_length: int = 2,
    prefixes=DEFAULT_PREFIXES,
    suffixes=DEFAULT_PLAIN_SUFFIXES,
) -> list[dict]:
    """
    Rank independent word candidates by morphological productivity.

    A candidate stem must itself be attested standalone — the constraint
    that dependent morphemes never appear alone is what makes this a
    test rather than an assumption. Candidates are then ranked by how
    many distinct affixed forms the corpus attests for them.

    Two filters keep bound forms and coincidences out:

    min_standalone_ratio rejects stems that technically appear alone but
    almost never do. ነ scores 0.0005 (362 standalone against 724,358 in
    ነው/ነን/ነች) — it is a bound form, not a word. ቤት scores 0.67. The
    handful of standalone hits for a bound form are noise, so mere
    presence is too weak a test; the proportion is the real signal.

    max_ambiguous_inflection_ratio guards order-4 fusion, where many
    native words simply end in an order-4 character. A genuine definite
    form is much rarer than its base, so an "inflection" at least as
    frequent as its claimed stem is really a separate word: ስራ (74,971)
    is not the definite of ስር (23,621).

    min_stem_length rejects single-fidel stems. Every one the corpus
    produces is a bound morpheme or a stray letter: የ and ለ are
    prefixes, ን and ዎ and ኛ are suffixes, and the rest are initials
    left behind by abbreviations. They pass the standalone ratio test
    because that ratio measures spacing, not wordhood — a prefix typed
    with a space after it is indistinguishable from a word standing
    alone, and የ scores 0.97 on exactly that noise.

    Finally, a stem attested as an inflection of another stem is
    dropped. ቤቱ survives every test above on its own, but it also sits
    in ቤት's form set, and admitting both breaks Level 4: longest-match
    segmentation takes ቤቱ whole and the definite marker ኡ never appears
    in the output. derived_stems() weighs that evidence by the rule
    that produced it.

    Args:
        counts:                Frequency table.
        min_stem_frequency:    Minimum standalone count for a stem.
        min_surface_frequency: Minimum count for a surface form to count
                               as evidence.
        min_distinct_forms:    Minimum distinct affixed forms required.
        min_standalone_ratio:  Minimum standalone / total frequency.
        max_ambiguous_inflection_ratio: For order-4 fusion only, the
                               largest the derived form may be relative
                               to the stem.
        min_stem_length:       Minimum fidels in a stem.
        prefixes:              Prefix inventory.
        suffixes:              Plain suffix inventory.

    Returns:
        List of dicts sorted by distinct form count then total
        frequency, each with:
            stem, standalone_frequency, distinct_forms, total_frequency,
            standalone_ratio, forms
    """
    attested: dict[str, dict[str, str]] = {}

    for word, count in counts.items():
        if count < min_surface_frequency:
            continue
        for stem, rule in candidate_stems_with_rules(
            word, prefixes, suffixes
        ).items():
            stem_count = counts.get(stem, 0)
            # The decisive test: the remainder must stand alone.
            if stem_count < min_stem_frequency:
                continue
            if (
                rule == RULE_AMBIGUOUS_FUSION
                and count > stem_count * max_ambiguous_inflection_ratio
            ):
                continue
            attested.setdefault(stem, {})[word] = rule

    # Applied before derived_stems() so that a stem rejected here cannot
    # go on to disqualify a real word as its inflection.
    attested = {s: f for s, f in attested.items() if len(s) >= min_stem_length}
    derived = derived_stems(attested, counts)

    candidates = []
    for stem, forms in attested.items():
        if stem in derived:
            continue
        if len(forms) < min_distinct_forms:
            continue
        standalone = counts[stem]
        total = standalone + sum(counts[f] for f in forms)
        ratio = standalone / total if total else 0.0
        if ratio < min_standalone_ratio:
            continue
        candidates.append(
            {
                "stem": stem,
                "standalone_frequency": standalone,
                "distinct_forms": len(forms),
                "total_frequency": total,
                "standalone_ratio": round(ratio, 4),
                "forms": sorted(forms, key=lambda f: -counts[f]),
            }
        )

    candidates.sort(
        key=lambda c: (-c["distinct_forms"], -c["total_frequency"])
    )
    return candidates


def consonantal_root(word: str) -> list[str]:
    """
    Level 1 root consonants for an independent morpheme.

    Thin wrapper over levels.level1_root that returns [] rather than
    None, because a lexicon entry's root_consonants field is a list.

    Level 1 is blocked for arbitrary surface forms because it applies to
    the independent morpheme only, and finding that requires Level 4. A
    lexicon stem is already the independent morpheme, so its root can be
    computed directly.

    Args:
        word: An independent morpheme (a lexicon stem).

    Returns:
        List of order-6 consonants, empty if none could be extracted.

    Examples:
        >>> consonantal_root("ቤት")
        ['ብ', 'ት']
        >>> consonantal_root("አለ")
        ['እ', 'ል']
        >>> consonantal_root("ጣእም")
        ['ጥ', 'ም']
    """
    return level1_root(word) or []


def identify_affix(form: str, stem: str) -> tuple[str, str] | None:
    """
    Work out which affix turns a stem into an attested surface form.

    Args:
        form: The attested surface form.
        stem: The candidate independent morpheme.

    Returns:
        ("prefix", affix) or ("suffix", affix), or None if the form is
        not derivable from the stem by a single known affix. Fused
        suffixes are reported by their morpheme name (ኦች, ኡ, ኣ) rather
        than by the characters that appear on the surface.

    Examples:
        >>> identify_affix("የቤት", "ቤት")
        ('prefix', 'የ')
        >>> identify_affix("ቤቶች", "ቤት")
        ('suffix', 'ኦች')
        >>> identify_affix("ቤቱ", "ቤት")
        ('suffix', 'ኡ')
    """
    for prefix in DEFAULT_PREFIXES:
        if form.startswith(prefix) and form[len(prefix):] == stem:
            return ("prefix", prefix)

    for suffix in DEFAULT_PLAIN_SUFFIXES:
        if form.endswith(suffix) and form[: -len(suffix)] == stem:
            return ("suffix", suffix)

    # Fused plural: the ኦ rides on the final consonant, ች follows.
    if len(form) >= 3 and form.endswith("ች"):
        carrier = form[-2]
        if (
            fidel_order(carrier) == _FUSED_PLURAL_ORDER
            and form[:-2] + to_order6(carrier) == stem
        ):
            return ("suffix", "ኦች")

    # Fused definite: the vowel rides on the final consonant alone.
    if len(form) >= 2:
        last = form[-1]
        order = fidel_order(last)
        if order in _FUSED_DEFINITE_ORDERS and form[:-1] + to_order6(last) == stem:
            return ("suffix", "ኡ" if order == 2 else "ኣ")

    return None


def to_lexicon_entry(candidate: dict, max_forms: int = 20) -> dict:
    """
    Convert a candidate into a spec section 3.3 lexicon entry.

    Fields that the corpus can establish are filled in; fields that
    require a native speaker are left null so that validation is a
    single pass rather than two.

    Derived here:
        allows_prefix, allows_suffix  from the attested forms — these
                                      are observations, not guesses
        root_consonants               Level 1 over the stem
        source                        corpus_bootstrap

    Left for the validator:
        category, gender, allows_infix, validated, notes

    Args:
        candidate: One entry from generate_candidates().
        max_forms: Cap on attested forms carried as evidence.

    Returns:
        A lexicon entry dict.
    """
    stem = candidate["stem"]
    prefixes, suffixes = set(), set()
    for form in candidate["forms"]:
        affix = identify_affix(form, stem)
        if affix is None:
            continue
        (prefixes if affix[0] == "prefix" else suffixes).add(affix[1])

    return {
        "word": stem,
        "category": None,
        "gender": None,
        "allows_prefix": sorted(prefixes),
        "allows_suffix": sorted(suffixes),
        "allows_infix": None,
        "root_consonants": consonantal_root(stem),
        "source": "corpus_bootstrap",
        "validated": False,
        "notes": "",
        # Evidence for the validator, not part of the spec schema.
        "evidence": {
            "standalone_frequency": candidate["standalone_frequency"],
            "total_frequency": candidate["total_frequency"],
            "standalone_ratio": candidate["standalone_ratio"],
            "distinct_forms": candidate["distinct_forms"],
            "attested_forms": candidate["forms"][:max_forms],
        },
    }


def write_lexicon_candidates(
    candidates: list[dict],
    path: str,
    limit: int | None = None,
    max_forms: int = 20,
    encoding: str = "utf-8",
) -> int:
    """
    Write candidates as spec section 3.3 lexicon entries for validation.

    Args:
        candidates: Output of generate_candidates().
        path:       Destination file.
        limit:      Write only the top N.
        max_forms:  Cap on attested forms per entry.
        encoding:   File encoding.

    Returns:
        Number of entries written.
    """
    selected = candidates[:limit] if limit else candidates
    entries = [to_lexicon_entry(c, max_forms) for c in selected]
    with open(path, "w", encoding=encoding) as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return len(entries)


def write_candidates(
    candidates: list[dict],
    path: str,
    limit: int | None = None,
    max_forms: int = 20,
    encoding: str = "utf-8",
) -> int:
    """
    Write candidates to JSON for native speaker validation.

    Args:
        candidates: Output of generate_candidates().
        path:       Destination file.
        limit:      Write only the top N candidates.
        max_forms:  Cap the attested forms recorded per candidate.
        encoding:   File encoding.

    Returns:
        Number of candidates written.
    """
    selected = candidates[:limit] if limit else candidates
    payload = [
        {**c, "forms": c["forms"][:max_forms], "validated": False}
        for c in selected
    ]
    with open(path, "w", encoding=encoding) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(payload)
