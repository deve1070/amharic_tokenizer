"""
Lexicon construction and access.

Builds lexicon.json from corpus candidates and provides the lookup that
Level 4 will use once entries are validated.

Named lexicon_builder rather than lexicon because `lexicon/` is the data
directory; a module of the same name would shadow it on import.

── Two files, two jobs ──────────────────────────────────────────────────

    lexicon/candidates.json   generated, gitignored, carries an evidence
                              block per entry (frequencies, attested
                              surface forms). This is the working file to
                              consult while validating.

    lexicon/lexicon.json      the deliverable, tracked in git, spec
                              section 3.3 schema only. Edited by hand as
                              validation proceeds.

── Validation is incremental ────────────────────────────────────────────

Every entry starts with validated=false. The flag carries one claim:
this word is a genuine independent morpheme. Level 4 reads only
validated entries, so the lexicon can ship at full size and become
useful the moment the first entries are confirmed — there is no
threshold to reach before the segmenter can be built or tested.
Validate the top of the file first; it is ordered by morphological
productivity, so that is where the evidence is strongest.

Filling in category and gender is a second, independent pass. It is
tracked by annotation_gaps() rather than by the same flag, because a
word can be confirmed as an independent morpheme long before anyone
labels its part of speech, and Level 4 needs only the former.

── What the corpus can and cannot fill in ───────────────────────────────

allows_prefix and allows_suffix are derived from affixes actually
attested in the corpus.

category, gender and allows_infix need a native speaker. They are null
rather than guessed, so that an unfilled field is visibly unfilled.

root_consonants is a native speaker's ruling too, but it is never null
while unreviewed. It starts as the consonant skeleton, with root_source
set to skeleton, and Level 1 does not report it (entry_root). The root
review replaces it and sets root_source to native_speaker. A reviewed
root of None means the word has no root: a loanword or a function word.

── The category review ──────────────────────────────────────────────────

category is filled through a review file: a TSV, one row per entry,
carrying a drafted category, a gloss and the corpus evidence. The draft
is a proposal and nothing more. Only a row the reviewer has marked — ok,
or a corrected category — is read back, so an unreviewed draft value can
never reach the lexicon. The same file proposes new entries (status new):
words that a suffix rule would otherwise split wrongly, such as ድራማ
against the adjective suffix ኣማ.
"""

import csv
import json
from dataclasses import dataclass

from amharic_morph.levels import bare_consonant, consonant_skeleton, is_ethiopic
from amharic_morph.morphology import STEM_KINDS

# Spec section 3.3 field order.
SCHEMA_FIELDS = (
    "word",
    "category",
    "gender",
    "allows_prefix",
    "allows_suffix",
    "allows_infix",
    "root_consonants",
    "root_source",
    "stems",
    "source",
    "validated",
    "notes",
)

VALID_CATEGORIES = (
    "noun",
    "verb",
    "adjective",
    "adverb",
    "preposition",
    "conjunction",
    "other",
)

VALID_GENDERS = ("masculine", "feminine", "none", "ambiguous")

# Source recorded on an entry, or a root, a native speaker supplied.
# Spec section 3.3.
SOURCE_NATIVE_SPEAKER = "native_speaker"

# root_source of a root nobody has reviewed: the consonant skeleton,
# kept as the review's proposal and never reported as Level 1.
ROOT_SOURCE_SKELETON = "skeleton"

VALID_ROOT_SOURCES = (SOURCE_NATIVE_SPEAKER, ROOT_SOURCE_SKELETON)

# The stems a verb entry can carry besides its own form, the 3ms
# perfective. Each is a native speaker's ruling, stored only once
# reviewed. The vowels inside the stem change with the aspect, which no
# rule recovers reliably: ፈለገ has the imperfective ፈልግ (ይፈልጋል),
# ሆነ the jussive ሁን (ይሁን), ሰጠ the gerund ሰጥ (ሰጥቶ). The palatal stems
# are the ones the 1s gerund (ሄጄ), the 2fs imperfective (ትሄጂ) and the
# 2fs imperative (ሂጂ) are built on.


def to_schema_entry(entry: dict) -> dict:
    """
    Reduce a candidate entry to the spec section 3.3 schema.

    Drops the evidence block, which belongs in candidates.json rather
    than in the lexicon itself.

    Args:
        entry: A candidate entry, possibly carrying "evidence".

    Returns:
        Dict containing exactly the schema fields, in spec order.
    """
    return {field: entry.get(field) for field in SCHEMA_FIELDS}


def build_lexicon(candidate_entries: list[dict], limit: int | None = None) -> list[dict]:
    """
    Build lexicon entries from candidate entries.

    Args:
        candidate_entries: Output of frequency.to_lexicon_entry(), in
                           ranked order.
        limit:             Keep only the top N.

    Returns:
        List of schema-only entries, order preserved.
    """
    selected = candidate_entries[:limit] if limit else candidate_entries
    return [to_schema_entry(e) for e in selected]


def save_lexicon(entries: list[dict], path: str, encoding: str = "utf-8") -> int:
    """
    Write the lexicon to JSON.

    Args:
        entries:  Lexicon entries.
        path:     Destination file.
        encoding: File encoding.

    Returns:
        Number of entries written.
    """
    with open(path, "w", encoding=encoding) as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return len(entries)


def load_lexicon(path: str, encoding: str = "utf-8") -> list[dict]:
    """
    Read the lexicon from JSON.

    Args:
        path:     Lexicon file.
        encoding: File encoding.

    Returns:
        List of entries as stored.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def validated_entries(entries: list[dict]) -> list[dict]:
    """
    Only the entries a native speaker has confirmed.

    Level 4 must anchor on these alone. An unvalidated candidate is a
    corpus statistic, and segmenting against unconfirmed stems would
    propagate errors like ስራ being treated as an inflection of ስር.

    Args:
        entries: Lexicon entries.

    Returns:
        Entries with validated set true.
    """
    return [e for e in entries if e.get("validated") is True]


def lexicon_index(entries: list[dict], validated_only: bool = True) -> dict[str, dict]:
    """
    Index entries by word for Level 4 lookup.

    Args:
        entries:        Lexicon entries.
        validated_only: Index only validated entries. Default true —
                        see validated_entries() for why.

    Returns:
        Dict mapping word to its entry.
    """
    source = validated_entries(entries) if validated_only else entries
    return {e["word"]: e for e in source if e.get("word")}


def validation_status(entries: list[dict]) -> dict:
    """
    Progress report over a lexicon file.

    Args:
        entries: Lexicon entries.

    Returns:
        Dict with total, validated, remaining, the fraction done, and
        annotated — validated entries that also carry category and
        gender. See annotation_gaps() for why the two are counted apart.
    """
    total = len(entries)
    done = len(validated_entries(entries))
    return {
        "total": total,
        "validated": done,
        "remaining": total - done,
        "fraction_validated": round(done / total, 4) if total else 0.0,
        "annotated": done - len(annotation_gaps(entries)),
    }


def schema_errors(entry: dict) -> list[str]:
    """
    Check one entry against the spec section 3.3 schema.

    Reports structural problems only — whether a word is genuinely an
    independent morpheme is the native speaker's call, not something
    this can decide.

    Args:
        entry: A lexicon entry.

    Returns:
        List of human-readable problems. Empty if the entry is well
        formed.
    """
    problems = []

    for field in SCHEMA_FIELDS:
        if field not in entry:
            problems.append(f"missing field: {field}")

    if not entry.get("word"):
        problems.append("word is empty")

    category = entry.get("category")
    if category is not None and category not in VALID_CATEGORIES:
        problems.append(f"category not recognised: {category}")

    gender = entry.get("gender")
    if gender is not None and gender not in VALID_GENDERS:
        problems.append(f"gender not recognised: {gender}")

    for field in ("allows_prefix", "allows_suffix", "root_consonants"):
        value = entry.get(field)
        if value is not None and not isinstance(value, list):
            problems.append(f"{field} must be a list")

    stems = entry.get("stems")
    if not isinstance(stems, dict):
        problems.append("stems must be a mapping")
    else:
        unknown = sorted(set(stems) - set(STEM_KINDS))
        if unknown:
            problems.append(f"stems kind not recognised: {unknown}")

    root_source = entry.get("root_source")
    if root_source not in VALID_ROOT_SOURCES:
        problems.append(f"root_source not recognised: {root_source}")
    elif root_source == ROOT_SOURCE_SKELETON and entry.get("root_consonants") is None:
        problems.append("root_consonants is None but the root is unreviewed")

    return problems


def annotation_gaps(entries: list[dict]) -> dict[str, list[str]]:
    """
    Validated entries whose descriptive fields are still unfilled.

    Separate from schema_errors() because `validated` and the annotation
    fields answer different questions. `validated` says the word is a
    genuine independent morpheme, which is the claim Level 4 depends on:
    it looks up word, root_consonants, allows_prefix and allows_suffix,
    and never reads category or gender. Those two describe the word for
    downstream consumers of the lexicon, and a missing one leaves an
    entry incompletely described without making it unsafe to segment
    against.

    Args:
        entries: Lexicon entries.

    Returns:
        Dict mapping word to the names of its unfilled fields, for
        validated entries only. Empty when annotation is complete.
    """
    gaps = {}
    for index, entry in enumerate(entries):
        if entry.get("validated") is not True:
            continue
        missing = [f for f in ("category", "gender") if entry.get(f) is None]
        if missing:
            gaps[entry.get("word") or f"<entry {index}>"] = missing
    return gaps


# Fields the corpus establishes. Everything else in SCHEMA_FIELDS is
# either identity (word, source) or the native speaker's (category,
# gender, allows_infix, root_consonants, root_source, validated, notes).
DERIVED_FIELDS = ("allows_prefix", "allows_suffix")


def refresh_derived_fields(entries: list[dict], evidence: dict[str, dict]) -> list[dict]:
    """
    Rewrite the corpus-derived fields from a fresh candidate run.

    allows_prefix and allows_suffix are observations, not judgements,
    so they can be recomputed at any time without
    reopening validation. This is what makes a correction to the
    candidate generator reach a lexicon that has already been validated
    by hand: the derived fields are replaced and everything else is
    kept. The root is a ruling (see the root review), so it is kept too.

    An entry with no fresh evidence is copied unchanged rather than
    emptied. Failing the current candidate filters is not evidence that
    a validated word takes no affixes.

    Args:
        entries:  Lexicon entries.
        evidence: Dict mapping word to a fresh entry, as built by
                  frequency.to_lexicon_entry() over a new candidate run.

    Returns:
        A new list of entries, order preserved.
    """
    refreshed = []
    for entry in entries:
        fresh = evidence.get(entry.get("word"))
        if fresh is None:
            refreshed.append(dict(entry))
            continue
        refreshed.append({**entry, **{f: fresh[f] for f in DERIVED_FIELDS}})
    return refreshed


def set_categories(entries: list[dict], categories: dict[str, str]) -> list[dict]:
    """
    Fill in `category` on the named entries.

    This is the write path for the annotation pass that annotation_gaps()
    tracks. It matters most for verbs: morphology.suffix_readings reads
    category to decide whether ኡ on a stem is the definite article
    (ቤት + ኡ, "the house") or third person plural agreement (ሆነ + ኡ,
    "they became"), and returns None while the field is unfilled rather
    than guessing.

    Unknown words raise rather than being skipped, because a mistyped
    Amharic word would otherwise leave the entry unannotated with
    nothing to show for it.

    Args:
        entries:    Lexicon entries.
        categories: Dict mapping word to one of VALID_CATEGORIES.

    Returns:
        A new list of entries, order preserved, with category set on the
        named words and every other entry copied unchanged.

    Raises:
        ValueError: If a category is not recognised, or a word is not in
                    the lexicon.
    """
    unknown_category = {c for c in categories.values() if c not in VALID_CATEGORIES}
    if unknown_category:
        raise ValueError(f"category not recognised: {sorted(unknown_category)}")

    present = {e.get("word") for e in entries}
    missing = sorted(set(categories) - present)
    if missing:
        raise ValueError(f"not in the lexicon: {missing}")

    return [
        {**e, "category": categories.get(e.get("word"), e.get("category"))}
        for e in entries
    ]


def remove_entries(entries: list[dict], words) -> list[dict]:
    """
    Drop the named entries from the lexicon.

    The write path for a word that should never have been admitted. Its
    caller is frequency.derived_lexicon_entries(), which finds entries
    that are an inflection of another entry: keeping both breaks Level 4,
    because longest match takes the inflected form whole and the affix
    never appears. ዜጎች is a lexicon entry and the plural of ዜጋ, which is
    also an entry, so ኦች is never reported for that word.

    Unknown words raise rather than being ignored, on the same reasoning
    as set_categories(): a mistyped Amharic word would otherwise remove
    nothing and report success.

    Args:
        entries: Lexicon entries.
        words:   Words to remove, any iterable.

    Returns:
        New list with those entries removed. The input is not modified.

    Raises:
        ValueError: If a word is not in the lexicon.
    """
    words = set(words)
    missing = sorted(words - {e.get("word") for e in entries})
    if missing:
        raise ValueError(f"not in the lexicon: {missing}")

    return [e for e in entries if e.get("word") not in words]


def validate_lexicon(entries: list[dict]) -> dict[str, list[str]]:
    """
    Run schema_errors over a whole lexicon.

    Args:
        entries: Lexicon entries.

    Returns:
        Dict mapping word (or index, when the word is missing) to its
        list of problems. Empty if the lexicon is well formed.
    """
    report = {}
    for index, entry in enumerate(entries):
        problems = schema_errors(entry)
        if problems:
            report[entry.get("word") or f"<entry {index}>"] = problems
    return report


# ── Review files ─────────────────────────────────────────────────────────────
#
# A review file is a TSV drafted for a native speaker, one row per entry,
# with a proposed value, its evidence, and an empty reviewed column. Only
# a row the reviewer has marked is read back, so a draft value can never
# reach the lexicon by default. The category review and the root review
# share the file handling below and differ only in their columns and in
# what a mark means.

# Marks common to every review, besides a value written out in full.
REVIEW_ACCEPT = "ok"
REVIEW_REJECT = "reject"

def write_review(
    rows: list[dict], path: str, columns: tuple[str, ...], encoding: str = "utf-8"
) -> int:
    """
    Write a review file.

    Args:
        rows:     Dicts keyed by column name. Missing columns are written
                  empty and keys outside columns are dropped.
        path:     Destination TSV.
        columns:  Column order, e.g. CATEGORY_REVIEW_COLUMNS.
        encoding: File encoding.

    Returns:
        Number of rows written.
    """
    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return len(rows)


def _reviewed_rows(path: str, encoding: str):
    """
    (line, row) for every row whose reviewed column is filled in.

    Values are stripped, and the reviewed mark is lowercased so that ok,
    OK and Ok read the same. line is the file line, header included,
    for error messages.
    """
    with open(path, "r", encoding=encoding, newline="") as f:
        for line, row in enumerate(csv.DictReader(f, delimiter="\t"), start=2):
            row = {k: (v or "").strip() for k, v in row.items()}
            row["reviewed"] = row["reviewed"].lower()
            if row["reviewed"]:
                yield line, row


# ── Category review ──────────────────────────────────────────────────────────

CATEGORY_REVIEW_COLUMNS = (
    "word",
    "status",
    "proposed",
    "gloss",
    "final_order",
    "verb_evidence",
    "reviewed",
    "notes",
)

# status: an entry already in the lexicon, or a word proposed for it.
REVIEW_ENTRY = "entry"
REVIEW_NEW = "new"


@dataclass(frozen=True)
class ReviewDecision:
    """
    One reviewed row of the category review file.

    Attributes:
        word:     The word.
        category: The category the reviewer settled on.
        notes:    The reviewer's notes, possibly empty.
        new:      True if the word is to be added to the lexicon.
    """

    word: str
    category: str
    notes: str
    new: bool


def read_category_review(path: str, encoding: str = "utf-8") -> list[ReviewDecision]:
    """
    The reviewer's decisions from a category review file.

    A row with an empty reviewed column is skipped, because its category
    is still only a draft. ok accepts the proposed category; a category
    name replaces it; reject declines a proposed new entry. Marks are
    read case-insensitively.

    Args:
        path:     Review TSV with CATEGORY_REVIEW_COLUMNS.
        encoding: File encoding.

    Returns:
        Decisions in file order, rejected new rows left out.

    Raises:
        ValueError: On an unrecognised mark or status, on ok against a
                    proposal that is not a valid category, and on reject
                    against an existing entry. Removing an entry goes
                    through remove_entries(), where the reason is kept.
    """
    decisions = []
    for line, row in _reviewed_rows(path, encoding):
        word, mark, status = row["word"], row["reviewed"], row["status"]
        if status not in (REVIEW_ENTRY, REVIEW_NEW):
            raise ValueError(f"line {line}, {word}: status not recognised: {status!r}")

        if mark == REVIEW_REJECT:
            if status == REVIEW_ENTRY:
                raise ValueError(
                    f"line {line}, {word}: reject applies to new words only; "
                    "use remove_entries() to drop an existing entry"
                )
            continue

        category = row["proposed"] if mark == REVIEW_ACCEPT else mark
        if category not in VALID_CATEGORIES:
            raise ValueError(f"line {line}, {word}: category not recognised: {category!r}")
        decisions.append(ReviewDecision(word, category, row["notes"], status == REVIEW_NEW))
    return decisions


# ── Root review ──────────────────────────────────────────────────────────────
#
# root_consonants is a ruling, not an observation (see
# docs/dev_plans/level1_root_redesign.md). The proposal comes from the
# IRIT root lexicon through reference.root_review_rows(); the reviewer
# accepts it, writes the root out with hyphens (ህ-ው-ን), or writes none
# for a word that has no root.

ROOT_REVIEW_COLUMNS = (
    "word",
    "proposed",
    "skeleton",
    "irit",
    "check",
    "reviewed",
    "notes",
)

# The mark, and the proposal, for a word with no root: a loanword or a
# function word (spec section 2.2).
ROOT_NONE = "none"

ROOT_SEPARATOR = "-"


@dataclass(frozen=True)
class RootDecision:
    """
    One reviewed row of the root review file.

    Attributes:
        word:  The entry.
        root:  Its radicals, or None for a word with no root.
        notes: The reviewer's notes, possibly empty.
    """

    word: str
    root: list[str] | None
    notes: str


def parse_root(text: str) -> list[str] | None:
    """
    Read a root written with hyphens, or none.

    Each radical must be one bare consonant: the order-6 form, or the
    plain consonant of a labiovelar. ሆ carries a vowel and ቍ is a
    labiovelar, so neither is a radical.

    Args:
        text: E.g. "ህ-ው-ን", or "none".

    Returns:
        The radicals, or None for none.

    Raises:
        ValueError: If a radical is not a single bare consonant.

    Examples:
        >>> parse_root("ህ - ው - ን")
        ['ህ', 'ው', 'ን']
        >>> parse_root("none") is None
        True
    """
    if text.strip().lower() == ROOT_NONE:
        return None
    radicals = [r.strip() for r in text.split(ROOT_SEPARATOR)]
    for radical in radicals:
        if len(radical) != 1 or not is_ethiopic(radical) or bare_consonant(radical) != radical:
            raise ValueError(f"not a bare consonant: {radical!r} in {text!r}")
    return radicals


def entry_root(entry: dict) -> list[str] | None:
    """
    Level 1: the reviewed root of a lexicon entry.

    An unreviewed root is the consonant skeleton, which keeps
    derivational consonants (መሆን ም-ህ-ን for the root ህ-ን), so it is
    withheld rather than reported. That is the same rule Level 4 follows
    for an unknown word.

    Args:
        entry: A lexicon entry.

    Returns:
        The root's radicals, or None when the root is unreviewed or the
        reviewer ruled that the word has none.

    Examples:
        >>> entry_root({"root_consonants": ["ህ", "ው", "ን"], "root_source": "native_speaker"})
        ['ህ', 'ው', 'ን']
        >>> entry_root({"root_consonants": ["ም", "ህ", "ን"], "root_source": "skeleton"}) is None
        True
    """
    if entry.get("root_source") != SOURCE_NATIVE_SPEAKER:
        return None
    return entry.get("root_consonants")


def read_root_review(path: str, encoding: str = "utf-8") -> list[RootDecision]:
    """
    The reviewer's decisions from a root review file.

    A row with an empty reviewed column is skipped. ok accepts the
    proposed root, a hyphenated root replaces it, and none records that
    the word has no root.

    Args:
        path:     Review TSV with ROOT_REVIEW_COLUMNS.
        encoding: File encoding.

    Returns:
        Decisions in file order.

    Raises:
        ValueError: With the line and word, when the accepted or written
                    root is not well formed.
    """
    decisions = []
    for line, row in _reviewed_rows(path, encoding):
        word, mark = row["word"], row["reviewed"]
        text = row["proposed"] if mark == REVIEW_ACCEPT else mark
        try:
            root = parse_root(text)
        except ValueError as e:
            raise ValueError(f"line {line}, {word}: {e}") from None
        decisions.append(RootDecision(word, root, row["notes"]))
    return decisions


def add_entries(entries: list[dict], categories: dict[str, str]) -> list[dict]:
    """
    Add words to the lexicon on a native speaker's authority.

    The write path for a word missing from the corpus candidates whose
    absence makes a suffix rule split it wrongly: without ድራማ as an
    entry, ድራማ ("drama") segments as ድር + ኣማ. The entry is validated
    at once and its source recorded as native_speaker. allows_prefix
    and allows_suffix start empty, since they are corpus observations
    that refresh_derived_fields() fills in later. The reviewer vouched
    for the word, not its root, so the root is the consonant skeleton,
    marked unreviewed.

    Args:
        entries:    Lexicon entries.
        categories: Dict mapping each new word to its category.

    Returns:
        New list with the words appended in the given order. The input
        is not modified.

    Raises:
        ValueError: If a category is not recognised, or a word is
                    already in the lexicon.
    """
    unknown_category = {c for c in categories.values() if c not in VALID_CATEGORIES}
    if unknown_category:
        raise ValueError(f"category not recognised: {sorted(unknown_category)}")

    present = sorted(set(categories) & {e.get("word") for e in entries})
    if present:
        raise ValueError(f"already in the lexicon: {present}")

    added = [
        {
            "word": word,
            "category": category,
            "gender": None,
            "allows_prefix": [],
            "allows_suffix": [],
            "allows_infix": None,
            "root_consonants": consonant_skeleton(word) or [],
            "root_source": ROOT_SOURCE_SKELETON,
            "stems": {},
            "source": SOURCE_NATIVE_SPEAKER,
            "validated": True,
            "notes": "",
        }
        for word, category in categories.items()
    ]
    return [dict(e) for e in entries] + added


def _with_note(notes: str, note: str) -> str:
    """notes with note appended, unless it is empty or already there."""
    if not note or note in notes.split("; "):
        return notes
    return f"{notes}; {note}" if notes else note


def apply_category_review(
    entries: list[dict], decisions: list[ReviewDecision]
) -> list[dict]:
    """
    Write a category review into the lexicon.

    Existing entries get their category through set_categories(), and
    the reviewer's notes are appended to the entry's notes. New words
    are appended through add_entries().

    Args:
        entries:   Lexicon entries.
        decisions: Output of read_category_review().

    Returns:
        A new list of entries. The input is not modified.

    Raises:
        ValueError: As set_categories() and add_entries() do.
    """
    existing = [d for d in decisions if not d.new]
    new = [d for d in decisions if d.new]

    result = set_categories(entries, {d.word: d.category for d in existing})
    notes = {d.word: d.notes for d in existing}
    result = [
        {**e, "notes": _with_note(e.get("notes") or "", notes[e["word"]])}
        if e.get("word") in notes
        else e
        for e in result
    ]

    result = add_entries(result, {d.word: d.category for d in new})
    new_notes = {d.word: d.notes for d in new}
    return [
        {**e, "notes": new_notes[e["word"]]} if e.get("word") in new_notes else e
        for e in result
    ]


def apply_root_review(entries: list[dict], decisions: list[RootDecision]) -> list[dict]:
    """
    Write a root review into the lexicon.

    Each decided entry gets its root and root_source native_speaker,
    and the reviewer's notes are appended to its notes.

    Args:
        entries:   Lexicon entries.
        decisions: Output of read_root_review().

    Returns:
        A new list of entries. The input is not modified.

    Raises:
        ValueError: If a decided word is not in the lexicon.
    """
    decided = {d.word: d for d in decisions}
    missing = sorted(set(decided) - {e.get("word") for e in entries})
    if missing:
        raise ValueError(f"not in the lexicon: {missing}")

    result = []
    for e in entries:
        d = decided.get(e.get("word"))
        if d is None:
            result.append(dict(e))
            continue
        result.append({
            **e,
            "root_consonants": d.root,
            "root_source": SOURCE_NATIVE_SPEAKER,
            "notes": _with_note(e.get("notes") or "", d.notes),
        })
    return result


# ── Stem review ──────────────────────────────────────────────────────────────
#
# One row per verb and stem kind. The proposal comes from the IRIT vote
# (reference.propose_stems); the reviewer accepts it, writes the stem,
# or writes none for a verb that lacks that stem.

STEM_REVIEW_COLUMNS = ("word", "kind", "proposed", "evidence", "reviewed", "notes")


@dataclass(frozen=True)
class StemDecision:
    """
    One reviewed row of the stem review file.

    Attributes:
        word:  The verb entry, in its 3ms perfective form.
        kind:  One of STEM_KINDS.
        stem:  The stem, or None for a verb without it.
        notes: The reviewer's notes, possibly empty.
    """

    word: str
    kind: str
    stem: str | None
    notes: str


def read_stem_review(path: str, encoding: str = "utf-8") -> list[StemDecision]:
    """
    The reviewer's decisions from a stem review file.

    Args:
        path:     Review TSV with STEM_REVIEW_COLUMNS.
        encoding: File encoding.

    Returns:
        Decisions in file order. Unreviewed rows are skipped.

    Raises:
        ValueError: With the line and word, on an unknown kind or a
                    stem that is not written in Ethiopic letters.
    """
    decisions = []
    for line, row in _reviewed_rows(path, encoding):
        word, kind, mark = row["word"], row["kind"], row["reviewed"]
        if kind not in STEM_KINDS:
            raise ValueError(f"line {line}, {word}: stem kind not recognised: {kind!r}")
        if mark == ROOT_NONE:
            stem = None
        else:
            stem = row["proposed"] if mark == REVIEW_ACCEPT else mark
            if not stem or not all(is_ethiopic(c) for c in stem):
                raise ValueError(f"line {line}, {word}: not a stem: {stem!r}")
        decisions.append(StemDecision(word, kind, stem, row["notes"]))
    return decisions


def apply_stem_review(entries: list[dict], decisions: list[StemDecision]) -> list[dict]:
    """
    Write a stem review into the lexicon.

    Args:
        entries:   Lexicon entries.
        decisions: Output of read_stem_review().

    Returns:
        A new list of entries. The input is not modified.

    Raises:
        ValueError: If a decided word is not in the lexicon or is not a
                    verb.
    """
    by_word = {e.get("word"): e for e in entries}
    missing = sorted({d.word for d in decisions} - set(by_word))
    if missing:
        raise ValueError(f"not in the lexicon: {missing}")
    not_verbs = sorted({d.word for d in decisions if by_word[d.word].get("category") != "verb"})
    if not_verbs:
        raise ValueError(f"stems apply to verbs only: {not_verbs}")

    stems = {e.get("word"): dict(e.get("stems") or {}) for e in entries}
    for d in decisions:
        if d.stem is None:
            stems[d.word].pop(d.kind, None)
        else:
            stems[d.word][d.kind] = d.stem
    return [{**e, "stems": stems[e.get("word")]} for e in entries]
