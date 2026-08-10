"""
Tests for lexicon construction and access.

The load-bearing behaviour is that Level 4 sees only validated entries.
An unvalidated candidate is a corpus statistic, and anchoring
segmentation on one propagates errors.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_tokenizer.lexicon_builder import (  # noqa: E402
    SCHEMA_FIELDS,
    build_lexicon,
    lexicon_index,
    load_lexicon,
    save_lexicon,
    schema_errors,
    to_schema_entry,
    validate_lexicon,
    validated_entries,
    validation_status,
)


def entry(word="ቤት", validated=False, category=None, **kwargs):
    base = {
        "word": word,
        "category": category,
        "gender": None,
        "allows_prefix": ["የ"],
        "allows_suffix": ["ኦች"],
        "allows_infix": None,
        "root_consonants": ["ብ", "ት"],
        "source": "corpus_bootstrap",
        "validated": validated,
        "notes": "",
    }
    base.update(kwargs)
    return base


# ── Schema reduction ──────────────────────────────────────────────────────────


class TestToSchemaEntry:
    """The evidence block belongs in candidates.json, not the lexicon"""

    def test_keeps_schema_fields(self):
        result = to_schema_entry(entry())
        assert set(result) == set(SCHEMA_FIELDS)

    def test_drops_evidence(self):
        with_evidence = entry()
        with_evidence["evidence"] = {"standalone_frequency": 100}
        assert "evidence" not in to_schema_entry(with_evidence)

    def test_missing_fields_become_null(self):
        assert to_schema_entry({"word": "ቤት"})["category"] is None

    def test_field_order_follows_spec(self):
        assert list(to_schema_entry(entry())) == list(SCHEMA_FIELDS)


class TestBuildLexicon:
    """Ranked order is preserved so the top of the file is validated first"""

    def test_builds_all(self):
        assert len(build_lexicon([entry("ቤት"), entry("ሰው")])) == 2

    def test_respects_limit(self):
        result = build_lexicon([entry("ቤት"), entry("ሰው")], limit=1)
        assert len(result) == 1
        assert result[0]["word"] == "ቤት"

    def test_preserves_order(self):
        result = build_lexicon([entry("ቤት"), entry("ሰው"), entry("ልጅ")])
        assert [e["word"] for e in result] == ["ቤት", "ሰው", "ልጅ"]

    def test_empty(self):
        assert build_lexicon([]) == []


# ── Round trip ────────────────────────────────────────────────────────────────


class TestSaveLoad:
    """JSON round trip with Amharic intact"""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "lexicon.json"
        save_lexicon([entry("ቤት")], str(path))
        assert load_lexicon(str(path))[0]["word"] == "ቤት"

    def test_returns_count(self, tmp_path):
        path = tmp_path / "lexicon.json"
        assert save_lexicon([entry("ቤት"), entry("ሰው")], str(path)) == 2

    def test_amharic_not_escaped(self, tmp_path):
        path = tmp_path / "lexicon.json"
        save_lexicon([entry("ቤት")], str(path))
        assert "ቤት" in path.read_text(encoding="utf-8")

    def test_valid_json(self, tmp_path):
        path = tmp_path / "lexicon.json"
        save_lexicon([entry("ቤት")], str(path))
        json.loads(path.read_text(encoding="utf-8"))

    def test_missing_file_raises(self, tmp_path):
        try:
            load_lexicon(str(tmp_path / "nope.json"))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


# ── Validation gating ─────────────────────────────────────────────────────────


class TestValidatedEntries:
    """Level 4 anchors on confirmed entries only"""

    def test_filters_unvalidated(self):
        entries = [entry("ቤት", validated=True), entry("ሰው", validated=False)]
        assert [e["word"] for e in validated_entries(entries)] == ["ቤት"]

    def test_none_validated(self):
        assert validated_entries([entry("ቤት")]) == []

    def test_missing_flag_treated_as_unvalidated(self):
        assert validated_entries([{"word": "ቤት"}]) == []


class TestLexiconIndex:
    """Lookup structure for Level 4"""

    def test_indexes_by_word(self):
        index = lexicon_index([entry("ቤት", validated=True)])
        assert index["ቤት"]["root_consonants"] == ["ብ", "ት"]

    def test_excludes_unvalidated_by_default(self):
        # The whole point: an unvalidated stem must not anchor Level 4.
        assert lexicon_index([entry("ቤት", validated=False)]) == {}

    def test_can_include_unvalidated_explicitly(self):
        index = lexicon_index([entry("ቤት")], validated_only=False)
        assert "ቤት" in index

    def test_empty_lexicon(self):
        assert lexicon_index([]) == {}


class TestValidationStatus:
    """Progress reporting over the file"""

    def test_counts(self):
        entries = [
            entry("ቤት", validated=True),
            entry("ሰው", validated=False),
            entry("ልጅ", validated=False),
        ]
        status = validation_status(entries)
        assert status["total"] == 3
        assert status["validated"] == 1
        assert status["remaining"] == 2

    def test_fraction(self):
        entries = [entry("ቤት", validated=True), entry("ሰው", validated=False)]
        assert validation_status(entries)["fraction_validated"] == 0.5

    def test_empty(self):
        assert validation_status([])["fraction_validated"] == 0.0


# ── Schema checking ───────────────────────────────────────────────────────────


class TestSchemaErrors:
    """Structural problems only — word choice is the validator's call"""

    def test_clean_entry(self):
        assert schema_errors(entry()) == []

    def test_missing_field(self):
        broken = entry()
        del broken["gender"]
        assert any("gender" in p for p in schema_errors(broken))

    def test_empty_word(self):
        assert any("word is empty" in p for p in schema_errors(entry(word="")))

    def test_unknown_category(self):
        broken = entry(category="thing")
        assert any("category" in p for p in schema_errors(broken))

    def test_valid_category_accepted(self):
        assert schema_errors(entry(validated=True, category="noun")) == []

    def test_unknown_gender(self):
        assert any("gender" in p for p in schema_errors(entry(gender="neuter")))

    def test_list_field_must_be_a_list(self):
        broken = entry(allows_prefix="የ")
        assert any("allows_prefix" in p for p in schema_errors(broken))

    def test_validated_without_category_flagged(self):
        # Usually means the flag was flipped before the linguistic fields.
        problems = schema_errors(entry(validated=True, category=None))
        assert any("category is still null" in p for p in problems)


class TestValidateLexicon:
    """Whole-file report"""

    def test_clean_lexicon(self):
        assert validate_lexicon([entry("ቤት"), entry("ሰው")]) == {}

    def test_reports_by_word(self):
        report = validate_lexicon([entry("ቤት", validated=True, category=None)])
        assert "ቤት" in report

    def test_handles_missing_word(self):
        report = validate_lexicon([{"validated": False}])
        assert "<entry 0>" in report

    def test_empty_lexicon(self):
        assert validate_lexicon([]) == {}
