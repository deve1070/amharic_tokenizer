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

allows_prefix, allows_suffix and root_consonants are derived: the first
two from affixes actually attested in the corpus, the third from Level 1
over the stem (valid because a lexicon entry IS an independent morpheme,
so the usual Level 4 dependency does not apply).

category, gender and allows_infix need a native speaker. They are null
rather than guessed, so that an unfilled field is visibly unfilled.
"""

import json

# Spec section 3.3 field order.
SCHEMA_FIELDS = (
    "word",
    "category",
    "gender",
    "allows_prefix",
    "allows_suffix",
    "allows_infix",
    "root_consonants",
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
# gender, allows_infix, validated, notes).
DERIVED_FIELDS = ("allows_prefix", "allows_suffix", "root_consonants")


def refresh_derived_fields(entries: list[dict], evidence: dict[str, dict]) -> list[dict]:
    """
    Rewrite the corpus-derived fields from a fresh candidate run.

    allows_prefix, allows_suffix and root_consonants are observations,
    not judgements, so they can be recomputed at any time without
    reopening validation. This is what makes a correction to the
    candidate generator reach a lexicon that has already been validated
    by hand: the derived fields are replaced and word, category, gender,
    allows_infix, validated and notes are kept.

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
    tracks. It matters most for verbs: morphology.suffix_function reads
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
