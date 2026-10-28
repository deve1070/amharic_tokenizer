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

from amharic_morph.lexicon_builder import (  # noqa: E402
    CATEGORY_REVIEW_COLUMNS,
    ROOT_REVIEW_COLUMNS,
    SCHEMA_FIELDS,
    ReviewDecision,
    RootDecision,
    add_entries,
    annotation_gaps,
    apply_category_review,
    build_lexicon,
    lexicon_index,
    load_lexicon,
    read_category_review,
    read_root_review,
    refresh_derived_fields,
    remove_entries,
    save_lexicon,
    schema_errors,
    set_categories,
    to_schema_entry,
    validate_lexicon,
    validated_entries,
    validation_status,
    write_review,
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

    def test_annotated_counts_only_fully_filled_entries(self):
        entries = [
            entry("ቤት", validated=True, category="noun", gender="none"),
            entry("ሰው", validated=True, category=None),
        ]
        status = validation_status(entries)
        assert status["validated"] == 2
        assert status["annotated"] == 1

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

    def test_validated_without_category_is_not_a_schema_error(self):
        # Confirming a word is an independent morpheme and labelling its
        # part of speech are separate passes; annotation_gaps() tracks
        # the second one.
        assert schema_errors(entry(validated=True, category=None)) == []


class TestAnnotationGaps:
    """Descriptive fields, counted apart from the validated flag"""

    def test_reports_unfilled_fields(self):
        gaps = annotation_gaps([entry("ቤት", validated=True)])
        assert gaps["ቤት"] == ["category", "gender"]

    def test_ignores_unvalidated_entries(self):
        assert annotation_gaps([entry("ቤት", validated=False)]) == {}

    def test_fully_annotated_entry_has_no_gap(self):
        complete = entry("ቤት", validated=True, category="noun", gender="none")
        assert annotation_gaps([complete]) == {}

    def test_partial_annotation(self):
        partial = entry("ቤት", validated=True, category="noun")
        assert annotation_gaps([partial]) == {"ቤት": ["gender"]}

    def test_empty(self):
        assert annotation_gaps([]) == {}


class TestValidateLexicon:
    """Whole-file report"""

    def test_clean_lexicon(self):
        assert validate_lexicon([entry("ቤት"), entry("ሰው")]) == {}

    def test_reports_by_word(self):
        report = validate_lexicon([entry("ቤት", category="thing")])
        assert "ቤት" in report

    def test_handles_missing_word(self):
        report = validate_lexicon([{"validated": False}])
        assert "<entry 0>" in report

    def test_empty_lexicon(self):
        assert validate_lexicon([]) == {}


class TestSetCategories:
    """The write path for the annotation pass"""

    def test_sets_the_named_entry(self):
        result = set_categories([entry("ሆነ", validated=True)], {"ሆነ": "verb"})
        assert result[0]["category"] == "verb"

    def test_leaves_other_entries_alone(self):
        entries = [entry("ሆነ", validated=True), entry("ቤት", validated=True)]
        result = set_categories(entries, {"ሆነ": "verb"})
        assert result[1]["category"] is None

    def test_does_not_mutate_the_input(self):
        entries = [entry("ሆነ", validated=True)]
        set_categories(entries, {"ሆነ": "verb"})
        assert entries[0]["category"] is None

    def test_preserves_order(self):
        entries = [entry("ሆነ"), entry("ቤት"), entry("ወጣ")]
        result = set_categories(entries, {"ወጣ": "verb"})
        assert [e["word"] for e in result] == ["ሆነ", "ቤት", "ወጣ"]

    def test_overwrites_an_existing_category(self):
        entries = [entry("ሆነ", validated=True, category="noun")]
        assert set_categories(entries, {"ሆነ": "verb"})[0]["category"] == "verb"

    def test_rejects_an_unknown_category(self):
        try:
            set_categories([entry("ሆነ")], {"ሆነ": "action"})
        except ValueError as e:
            assert "action" in str(e)
            return
        raise AssertionError("expected ValueError")

    def test_rejects_a_word_not_in_the_lexicon(self):
        # A mistyped Amharic word would otherwise annotate nothing.
        try:
            set_categories([entry("ሆነ")], {"ሆኑ": "verb"})
        except ValueError as e:
            assert "ሆኑ" in str(e)
            return
        raise AssertionError("expected ValueError")

    def test_closes_the_annotation_gap_it_fills(self):
        entries = set_categories([entry("ሆነ", validated=True)], {"ሆነ": "verb"})
        assert annotation_gaps(entries) == {"ሆነ": ["gender"]}

    def test_empty(self):
        assert set_categories([], {}) == []


class TestRefreshDerivedFields:
    """
    A correction to the candidate generator has to reach a lexicon that
    is already validated by hand. The derived fields are observations,
    so they can be replaced; the validator's own fields cannot.
    """

    def fresh(self, word="ቤት", **kwargs):
        base = {
            "allows_prefix": ["የ", "በ"],
            "allows_suffix": ["ኡ", "ኦች"],
            "root_consonants": ["ብ", "ት"],
        }
        base.update(kwargs)
        return {word: base}

    def test_replaces_the_derived_fields(self):
        entries = [entry("ቤት", validated=True)]
        result = refresh_derived_fields(entries, self.fresh())
        assert result[0]["allows_suffix"] == ["ኡ", "ኦች"]
        assert result[0]["allows_prefix"] == ["የ", "በ"]

    def test_keeps_the_validated_flag(self):
        entries = [entry("ቤት", validated=True)]
        assert refresh_derived_fields(entries, self.fresh())[0]["validated"] is True

    def test_keeps_hand_set_annotation(self):
        entries = [entry("ቤት", validated=True, category="noun", gender="none")]
        result = refresh_derived_fields(entries, self.fresh())
        assert result[0]["category"] == "noun"
        assert result[0]["gender"] == "none"

    def test_keeps_notes(self):
        entries = [entry("ቤት", validated=True, notes="checked against Kane")]
        assert refresh_derived_fields(entries, self.fresh())[0]["notes"] == (
            "checked against Kane"
        )

    def test_entry_without_fresh_evidence_is_untouched(self):
        # Failing the current candidate filters is not evidence that a
        # validated word takes no affixes.
        entries = [entry("ወንድም", validated=True)]
        result = refresh_derived_fields(entries, self.fresh())
        assert result[0]["allows_suffix"] == ["ኦች"]

    def test_does_not_mutate_the_input(self):
        entries = [entry("ቤት", validated=True)]
        refresh_derived_fields(entries, self.fresh())
        assert entries[0]["allows_suffix"] == ["ኦች"]

    def test_preserves_order(self):
        entries = [entry("ሰው"), entry("ቤት"), entry("ልጅ")]
        result = refresh_derived_fields(entries, self.fresh())
        assert [e["word"] for e in result] == ["ሰው", "ቤት", "ልጅ"]

    def test_output_is_still_schema_clean(self):
        result = refresh_derived_fields([entry("ቤት", validated=True)], self.fresh())
        assert schema_errors(result[0]) == []

    def test_empty(self):
        assert refresh_derived_fields([], {}) == []


class TestRemoveEntries:
    """
    The write path for an entry that should never have been admitted:
    one that is an inflection of another entry.
    """

    def test_removes_the_named_entry(self):
        entries = [entry("ዜጋ"), entry("ዜጎች")]
        assert [e["word"] for e in remove_entries(entries, ["ዜጎች"])] == ["ዜጋ"]

    def test_removes_several(self):
        entries = [entry("ዜጋ"), entry("ዜጎች"), entry("ጓደኛ"), entry("ጓደኞች")]
        result = remove_entries(entries, ["ዜጎች", "ጓደኞች"])
        assert [e["word"] for e in result] == ["ዜጋ", "ጓደኛ"]

    def test_preserves_order(self):
        entries = [entry("ሆነ"), entry("ቤት"), entry("ወጣ")]
        result = remove_entries(entries, ["ቤት"])
        assert [e["word"] for e in result] == ["ሆነ", "ወጣ"]

    def test_does_not_mutate_the_input(self):
        entries = [entry("ዜጋ"), entry("ዜጎች")]
        remove_entries(entries, ["ዜጎች"])
        assert len(entries) == 2

    def test_keeps_the_entry_objects_intact(self):
        entries = [entry("ዜጋ", validated=True, category="noun")]
        assert remove_entries(entries, [])[0]["category"] == "noun"

    def test_rejects_a_word_not_in_the_lexicon(self):
        try:
            remove_entries([entry("ዜጋ")], ["ኮምፒውተር"])
        except ValueError as e:
            assert "ኮምፒውተር" in str(e)
            return
        raise AssertionError("expected ValueError")

    def test_reports_every_missing_word(self):
        try:
            remove_entries([entry("ዜጋ")], ["ሀሁ", "ኮምፒውተር"])
        except ValueError as e:
            assert "ሀሁ" in str(e) and "ኮምፒውተር" in str(e)
            return
        raise AssertionError("expected ValueError")

    def test_accepts_any_iterable(self):
        entries = [entry("ዜጋ"), entry("ዜጎች")]
        result = remove_entries(entries, (w for w in ["ዜጎች"]))
        assert [e["word"] for e in result] == ["ዜጋ"]

    def test_removing_nothing_is_a_copy(self):
        entries = [entry("ዜጋ")]
        assert remove_entries(entries, []) == entries

    def test_empty(self):
        assert remove_entries([], []) == []


# ── Category review ──────────────────────────────────────────────────────────
#
# The review file is a draft written for a native speaker. Only rows the
# reviewer has marked may reach the lexicon, so a drafted category can never
# be applied by default.


def review_row(word="ሆነ", status="entry", proposed="verb", reviewed="", **kwargs):
    row = {column: "" for column in CATEGORY_REVIEW_COLUMNS}
    row.update(word=word, status=status, proposed=proposed, reviewed=reviewed)
    row.update(kwargs)
    return row


def round_trip(tmp_path, rows):
    path = tmp_path / "review.tsv"
    write_review(rows, str(path), CATEGORY_REVIEW_COLUMNS)
    return read_category_review(str(path))


def raises_value_error(call, *fragments):
    try:
        call()
    except ValueError as e:
        for fragment in fragments:
            assert fragment in str(e)
        return
    raise AssertionError("expected ValueError")


class TestReadCategoryReview:
    def test_unreviewed_rows_are_not_returned(self, tmp_path):
        assert round_trip(tmp_path, [review_row()]) == []

    def test_ok_accepts_the_proposal(self, tmp_path):
        decisions = round_trip(tmp_path, [review_row(reviewed="ok")])
        assert decisions == [ReviewDecision("ሆነ", "verb", "", False)]

    def test_a_category_overrides_the_proposal(self, tmp_path):
        decisions = round_trip(tmp_path, [review_row("ስራ", reviewed="noun")])
        assert decisions[0].category == "noun"

    def test_marks_are_case_and_space_insensitive(self, tmp_path):
        decisions = round_trip(tmp_path, [review_row(reviewed=" OK ")])
        assert decisions[0].category == "verb"

    def test_carries_reviewer_notes(self, tmp_path):
        row = review_row("ዋጋ", proposed="noun", reviewed="ok", notes="also ተዋጋ")
        assert round_trip(tmp_path, [row])[0].notes == "also ተዋጋ"

    def test_new_row_is_flagged_new(self, tmp_path):
        row = review_row("ድራማ", status="new", proposed="noun", reviewed="ok")
        assert round_trip(tmp_path, [row])[0].new is True

    def test_rejected_new_row_is_not_returned(self, tmp_path):
        row = review_row("ድራማ", status="new", proposed="noun", reviewed="reject")
        assert round_trip(tmp_path, [row]) == []

    def test_rejecting_an_existing_entry_raises(self, tmp_path):
        # Removing an entry goes through remove_entries, where the reason
        # is recorded; a category review cannot do it silently.
        raises_value_error(
            lambda: round_trip(tmp_path, [review_row(reviewed="reject")]), "ሆነ"
        )

    def test_unrecognised_mark_raises_with_the_word(self, tmp_path):
        raises_value_error(
            lambda: round_trip(tmp_path, [review_row(reviewed="action")]),
            "ሆነ",
            "action",
        )

    def test_ok_on_an_invalid_proposal_raises(self, tmp_path):
        raises_value_error(
            lambda: round_trip(tmp_path, [review_row(proposed="", reviewed="ok")]),
            "ሆነ",
        )

    def test_unknown_status_raises(self, tmp_path):
        raises_value_error(
            lambda: round_trip(tmp_path, [review_row(status="maybe", reviewed="ok")]),
            "maybe",
        )

    def test_preserves_file_order(self, tmp_path):
        rows = [review_row("ሆነ", reviewed="ok"), review_row("ቤት", proposed="noun", reviewed="ok")]
        assert [d.word for d in round_trip(tmp_path, rows)] == ["ሆነ", "ቤት"]

    def test_text_with_quotes_survives(self, tmp_path):
        row = review_row(reviewed="ok", gloss='"to become"', notes="a\tb")
        decision = round_trip(tmp_path, [row])[0]
        assert decision.notes == "a\tb"



class TestWriteReview:
    """One writer for every review file; the columns are a parameter"""

    def test_writes_the_given_columns_in_order(self, tmp_path):
        path = tmp_path / "review.tsv"
        write_review([{"word": "ቤት", "b": "2"}], str(path), ("word", "b"))
        assert path.read_text(encoding="utf-8").splitlines() == ["word\tb", "ቤት\t2"]

    def test_missing_columns_are_empty_and_extras_dropped(self, tmp_path):
        path = tmp_path / "review.tsv"
        write_review([{"word": "ቤት", "extra": "x"}], str(path), ("word", "b"))
        assert path.read_text(encoding="utf-8").splitlines()[1] == "ቤት\t"

    def test_returns_row_count(self, tmp_path):
        assert write_review([{}, {}], str(tmp_path / "r.tsv"), ("word",)) == 2


# ── Root review ──────────────────────────────────────────────────────────────


def root_row(word="ሆነ", proposed="ህ-ው-ን", reviewed="", **kwargs):
    row = {column: "" for column in ROOT_REVIEW_COLUMNS}
    row.update(word=word, proposed=proposed, reviewed=reviewed)
    row.update(kwargs)
    return row


def root_round_trip(tmp_path, rows):
    path = tmp_path / "roots.tsv"
    write_review(rows, str(path), ROOT_REVIEW_COLUMNS)
    return read_root_review(str(path))


class TestReadRootReview:
    def test_unreviewed_rows_are_not_returned(self, tmp_path):
        assert root_round_trip(tmp_path, [root_row()]) == []

    def test_ok_accepts_the_proposal(self, tmp_path):
        decisions = root_round_trip(tmp_path, [root_row(reviewed="ok")])
        assert decisions == [RootDecision("ሆነ", ["ህ", "ው", "ን"], "")]

    def test_a_root_overrides_the_proposal(self, tmp_path):
        decisions = root_round_trip(tmp_path, [root_row(proposed="ህ-ን", reviewed="ህ-ው-ን")])
        assert decisions[0].root == ["ህ", "ው", "ን"]

    def test_spaces_around_radicals_are_ignored(self, tmp_path):
        decisions = root_round_trip(tmp_path, [root_row(reviewed=" ህ - ው - ን ")])
        assert decisions[0].root == ["ህ", "ው", "ን"]

    def test_none_means_no_root(self, tmp_path):
        decisions = root_round_trip(tmp_path, [root_row("ላይ", reviewed="none")])
        assert decisions[0].root is None

    def test_none_is_case_insensitive(self, tmp_path):
        assert root_round_trip(tmp_path, [root_row(reviewed=" None ")])[0].root is None

    def test_ok_on_a_none_proposal(self, tmp_path):
        decisions = root_round_trip(tmp_path, [root_row("እና", proposed="none", reviewed="ok")])
        assert decisions[0].root is None

    def test_carries_reviewer_notes(self, tmp_path):
        row = root_row(reviewed="ok", notes="as in ሆነ")
        assert root_round_trip(tmp_path, [row])[0].notes == "as in ሆነ"

    def test_a_radical_with_a_vowel_raises(self, tmp_path):
        # Radicals are bare consonants; ሆ carries the vowel ኦ.
        raises_value_error(
            lambda: root_round_trip(tmp_path, [root_row(reviewed="ሆ-ን")]), "ሆነ", "ሆ"
        )

    def test_a_labiovelar_radical_raises(self, tmp_path):
        # Q5: ቋ is ቅ at Level 1, so ቍ is never a radical.
        raises_value_error(lambda: root_round_trip(tmp_path, [root_row(reviewed="ቍ-ም")]), "ቍ")

    def test_a_radical_of_two_letters_raises(self, tmp_path):
        raises_value_error(lambda: root_round_trip(tmp_path, [root_row(reviewed="ህው-ን")]), "ህው")

    def test_an_empty_radical_raises(self, tmp_path):
        raises_value_error(lambda: root_round_trip(tmp_path, [root_row(reviewed="ህ--ን")]), "ሆነ")

    def test_a_latin_mark_raises(self, tmp_path):
        raises_value_error(lambda: root_round_trip(tmp_path, [root_row(reviewed="yes")]), "yes")

    def test_ok_on_an_empty_proposal_raises(self, tmp_path):
        raises_value_error(
            lambda: root_round_trip(tmp_path, [root_row(proposed="", reviewed="ok")]), "ሆነ"
        )

    def test_error_names_the_line(self, tmp_path):
        rows = [root_row("ቤት", proposed="ብ-ት", reviewed="ok"), root_row(reviewed="ሆ-ን")]
        raises_value_error(lambda: root_round_trip(tmp_path, rows), "line 3")

    def test_preserves_file_order(self, tmp_path):
        rows = [root_row(reviewed="ok"), root_row("ቤት", proposed="ብ-ት", reviewed="ok")]
        assert [d.word for d in root_round_trip(tmp_path, rows)] == ["ሆነ", "ቤት"]


class TestAddEntries:
    def test_adds_a_validated_native_speaker_entry(self):
        result = add_entries([entry("ቤት")], {"ድራማ": "noun"})
        added = result[-1]
        assert added["word"] == "ድራማ"
        assert added["category"] == "noun"
        assert added["validated"] is True
        assert added["source"] == "native_speaker"

    def test_computes_the_root(self):
        assert add_entries([], {"ፈጠራ": "noun"})[0]["root_consonants"] == ["ፍ", "ጥ", "ር"]

    def test_affix_fields_start_empty(self):
        added = add_entries([], {"ድራማ": "noun"})[0]
        assert added["allows_prefix"] == [] and added["allows_suffix"] == []

    def test_output_is_schema_clean(self):
        assert schema_errors(add_entries([], {"ድራማ": "noun"})[0]) == []

    def test_rejects_a_word_already_present(self):
        raises_value_error(lambda: add_entries([entry("ቤት")], {"ቤት": "noun"}), "ቤት")

    def test_rejects_an_unknown_category(self):
        raises_value_error(lambda: add_entries([], {"ድራማ": "thing"}), "thing")

    def test_does_not_mutate_the_input(self):
        entries = [entry("ቤት")]
        add_entries(entries, {"ድራማ": "noun"})
        assert len(entries) == 1


class TestApplyCategoryReview:
    def test_sets_categories(self):
        result = apply_category_review(
            [entry("ሆነ", validated=True)], [ReviewDecision("ሆነ", "verb", "", False)]
        )
        assert result[0]["category"] == "verb"

    def test_appends_notes_to_existing_ones(self):
        result = apply_category_review(
            [entry("ዋጋ", notes="frequent")],
            [ReviewDecision("ዋጋ", "noun", "also ተዋጋ", False)],
        )
        assert result[0]["notes"] == "frequent; also ተዋጋ"

    def test_does_not_repeat_a_note_already_recorded(self):
        decision = ReviewDecision("ዋጋ", "noun", "also ተዋጋ", False)
        once = apply_category_review([entry("ዋጋ")], [decision])
        assert apply_category_review(once, [decision])[0]["notes"] == "also ተዋጋ"

    def test_adds_new_entries_at_the_end(self):
        result = apply_category_review(
            [entry("ቤት")], [ReviewDecision("ድራማ", "noun", "loanword", True)]
        )
        assert [e["word"] for e in result] == ["ቤት", "ድራማ"]
        assert result[-1]["notes"] == "loanword"

    def test_unknown_existing_word_raises(self):
        raises_value_error(
            lambda: apply_category_review([entry("ቤት")], [ReviewDecision("ሆኑ", "verb", "", False)]),
            "ሆኑ",
        )

    def test_does_not_mutate_the_input(self):
        entries = [entry("ሆነ")]
        apply_category_review(entries, [ReviewDecision("ሆነ", "verb", "", False)])
        assert entries[0]["category"] is None

    def test_no_decisions_is_a_copy(self):
        entries = [entry("ሆነ")]
        assert apply_category_review(entries, []) == entries
