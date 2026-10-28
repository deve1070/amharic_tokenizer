"""
Tests for the IRIT reference lexicons.

The reference is a hand annotation, so the reader has two jobs: parse
the annotation format faithfully, and refuse rows that do not follow it
rather than guess at what they meant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_morph.morphology import Segmentation  # noqa: E402
from amharic_morph.reference import (  # noqa: E402
    ReferenceAnalysis,
    inflection_agreement,
    is_well_formed_root,
    load_reference,
    outer_affixes,
    parse_annotation,
)


def write(tmp_path, rows):
    path = tmp_path / "lexicon.txt"
    path.write_text(
        "Surface word,Annotated form\r\n" + "".join(f"{w},{a}\r\n" for w, a in rows),
        encoding="utf-8",
    )
    return path


# ── Parsing one annotation ────────────────────────────────────────────────────


class TestParseAnnotation:
    def test_morphemes_split_on_underscore(self):
        (analysis,) = parse_annotation("ልጅ_ኦች_አችን")
        assert analysis.morphemes == ("ልጅ", "ኦች", "አችን")

    def test_unsegmented_word_is_one_morpheme(self):
        (analysis,) = parse_annotation("ህዝብ")
        assert analysis.morphemes == ("ህዝብ",)
        assert analysis.roots == ()

    def test_root_radicals_split_on_hyphen(self):
        (analysis,) = parse_annotation("ተ_ድ-ር-ግ_ኧ")
        assert analysis.morphemes == ("ተ", "ድ-ር-ግ", "ኧ")
        assert analysis.roots == (("ድ", "ር", "ግ"),)

    def test_root_is_the_single_root(self):
        (analysis,) = parse_annotation("ተ_ድ-ር-ግ_ኧ")
        assert analysis.root == ("ድ", "ር", "ግ")

    def test_root_is_none_without_one(self):
        (analysis,) = parse_annotation("የ_ኦነግ")
        assert analysis.root is None

    def test_root_is_none_when_there_are_several(self):
        # A compound carries two roots; neither is the word's root.
        (analysis,) = parse_annotation("ስ-ር_ን-ግ-ር")
        assert len(analysis.roots) == 2
        assert analysis.root is None

    def test_alternatives_follow_the_primary(self):
        analyses = parse_annotation("በ_ላይ[በ_በላይ]")
        assert [a.morphemes for a in analyses] == [("በ", "ላይ"), ("በ", "በላይ")]

    def test_two_alternatives(self):
        analyses = parse_annotation("ማሩ[ማር_ኡ][ም-ዕ-ር_ኡ]")
        assert len(analyses) == 3
        # Layer 0 maps ዕ to እ.
        assert analyses[2].root == ("ም", "እ", "ር")

    def test_text_is_normalized_like_the_corpus(self):
        # Layer 0 maps ኣ to አ and ሐ to ሀ; the reference must match the
        # corpus it is compared with.
        (analysis,) = parse_annotation("ሀገር_ኣችን")
        assert analysis.morphemes == ("ሀገር", "አችን")
        (analysis,) = parse_annotation("ሐሳብ")
        assert analysis.morphemes == ("ሀሳብ",)

    def test_unbalanced_brackets_are_refused(self):
        assert parse_annotation("ለ_ሆን_ኧ[ለ_ሆነ[") == ()
        assert parse_annotation("አ_ቆጠር]") == ()

    def test_spaces_are_refused(self):
        assert parse_annotation("ሊ_ብ-ል_ኡ[ሊብ እልኡ]") == ()

    def test_trailing_punctuation_is_refused(self):
        assert parse_annotation("ለማ[ለም_ኣ]፥") == ()

    def test_empty_morpheme_is_refused(self):
        assert parse_annotation("በ__ቤት") == ()


class TestWellFormedRoot:
    def test_one_fidel_per_radical(self):
        assert is_well_formed_root(("ክ", "ት", "ም"))

    def test_a_radical_fused_with_a_word_is_not(self):
        # ከተማ is annotated ክ-ት-ምከተማ in one row.
        assert not is_well_formed_root(("ክ", "ት", "ምከተማ"))

    def test_empty_radical_is_not(self):
        # ከ_ተ_ን-ብ- ብ-ር_ኧ loses a radical to a stray space.
        assert not is_well_formed_root(("ን", "ብ", ""))

    def test_non_ethiopic_is_not(self):
        assert not is_well_formed_root(("ክ", "t"))

    def test_empty_root_is_not(self):
        assert not is_well_formed_root(())


# ── Loading a lexicon file ────────────────────────────────────────────────────


class TestLoadReference:
    def test_maps_surface_word_to_analyses(self, tmp_path):
        reference = load_reference(write(tmp_path, [("ልጆቻችን", "ልጅ_ኦች_አችን")]))
        assert reference["ልጆቻችን"] == (
            ReferenceAnalysis(("ልጅ", "ኦች", "አችን"), ()),
        )

    def test_header_is_not_an_entry(self, tmp_path):
        assert "Surface word" not in load_reference(write(tmp_path, []))

    def test_surface_word_is_normalized(self, tmp_path):
        reference = load_reference(write(tmp_path, [("ሐሳብ", "ሐሳብ")]))
        assert "ሀሳብ" in reference

    def test_duplicate_rows_merge_without_repeats(self, tmp_path):
        reference = load_reference(
            write(tmp_path, [("ማሩ", "ማሩ"), ("ማሩ", "ማር_ኡ"), ("ማሩ", "ማሩ")])
        )
        assert [a.morphemes for a in reference["ማሩ"]] == [("ማሩ",), ("ማር", "ኡ")]

    def test_spellings_that_normalize_together_merge(self, tmp_path):
        reference = load_reference(write(tmp_path, [("ሐሳብ", "ሐሳብ"), ("ሀሳብ", "ሀሳብ")]))
        assert len(reference["ሀሳብ"]) == 1

    def test_malformed_annotation_is_skipped(self, tmp_path):
        reference = load_reference(write(tmp_path, [("ለሆነ", "ለ_ሆን_ኧ[ለ_ሆነ[")]))
        assert "ለሆነ" not in reference

    def test_surface_with_punctuation_is_skipped(self, tmp_path):
        reference = load_reference(write(tmp_path, [("ለማ፥", "ለማ[ለም_ኣ]፥")]))
        assert reference == {}

    def test_surface_with_bracket_is_skipped(self, tmp_path):
        reference = load_reference(write(tmp_path, [("[በኤድስ", "[በ_ኤድስ")]))
        assert reference == {}

    def test_row_with_wrong_column_count_is_skipped(self, tmp_path):
        path = tmp_path / "lexicon.txt"
        path.write_text("Surface word,Annotated form\n\nቤት,ቤት,x\nሰው,ሰው\n", encoding="utf-8")
        assert list(load_reference(path)) == ["ሰው"]


# ── Comparing Level 4 with the reference ──────────────────────────────────────


class TestOuterAffixes:
    def test_inflection_on_both_sides_is_peeled(self):
        analysis = ReferenceAnalysis(("የ", "ልጅ", "ኦች", "አችን"), ())
        prefixes, core, suffixes = outer_affixes(analysis)
        assert prefixes == ("የ",)
        assert core == ("ልጅ",)
        assert suffixes == ("ኦች", "አችን")

    def test_derivation_stays_in_the_core(self):
        # መ_ንግስ_ት: the reference splits derivation, Level 4 does not.
        prefixes, core, suffixes = outer_affixes(ReferenceAnalysis(("መ", "ንግስ", "ት"), ()))
        assert prefixes == ()
        assert core == ("መ", "ንግስ", "ት")
        assert suffixes == ()

    def test_core_is_never_emptied(self):
        # ን alone is a word in this annotation; it cannot also be a suffix.
        prefixes, core, suffixes = outer_affixes(ReferenceAnalysis(("ን",), ()))
        assert core == ("ን",)
        assert suffixes == ()

    def test_stacked_prefixes(self):
        prefixes, core, _ = outer_affixes(ReferenceAnalysis(("በየ", "አመት", "ኡ"), ()))
        assert prefixes == ("በየ",)
        assert core == ("አመት",)

    def test_inventories_are_parameters(self):
        _, core, suffixes = outer_affixes(
            ReferenceAnalysis(("ቤት", "ኦች"), ()), prefixes=(), suffixes=()
        )
        assert core == ("ቤት", "ኦች")
        assert suffixes == ()


def segmenter(table):
    """A Level 4 stand-in: word -> Segmentation from a fixed table."""
    def segment(word):
        return table.get(word, Segmentation(word, word, known=False))
    return segment


class TestInflectionAgreement:
    def test_agreement_is_token_weighted(self):
        reference = {
            "ቤቶች": (ReferenceAnalysis(("ቤት", "ኦች"), ()),),
            "ቤቱ": (ReferenceAnalysis(("ቤት", "ኡ"), ()),),
        }
        segment = segmenter({
            "ቤቶች": Segmentation("ቤቶች", "ቤት", suffixes=("ኦች",)),
            "ቤቱ": Segmentation("ቤቱ", "ቤቱ"),
        })
        score = inflection_agreement(segment, reference, {"ቤቶች": 3, "ቤቱ": 1})
        assert score.compared_tokens == 4
        assert score.agreeing_tokens == 3
        assert score.disagreements == {"ቤቱ": 1}

    def test_our_fused_vowel_label_matches_the_normalized_reference(self):
        # Level 4 labels the possessive ኣቸው; Layer 0 writes the reference
        # as አቸው. They are the same morpheme.
        reference = {"ቤታቸው": (ReferenceAnalysis(("ቤት", "አቸው"), ()),)}
        segment = segmenter({"ቤታቸው": Segmentation("ቤታቸው", "ቤት", suffixes=("ኣቸው",))})
        score = inflection_agreement(segment, reference, {"ቤታቸው": 1})
        assert score.agreeing_tokens == 1

    def test_any_alternative_may_agree(self):
        reference = {"በላይ": (
            ReferenceAnalysis(("በ", "ላይ"), ()),
            ReferenceAnalysis(("በላይ",), ()),
        )}
        segment = segmenter({"በላይ": Segmentation("በላይ", "በላይ")})
        assert inflection_agreement(segment, reference, {"በላይ": 5}).agreeing_tokens == 5

    def test_unknown_words_are_not_compared(self):
        reference = {"ሆስፒታል": (ReferenceAnalysis(("ሆስፒታል",), ()),)}
        score = inflection_agreement(segmenter({}), reference, {"ሆስፒታል": 9})
        assert score.compared_tokens == 0

    def test_words_absent_from_the_corpus_are_not_compared(self):
        reference = {"ቤቶች": (ReferenceAnalysis(("ቤት", "ኦች"), ()),)}
        segment = segmenter({"ቤቶች": Segmentation("ቤቶች", "ቤት", suffixes=("ኦች",))})
        assert inflection_agreement(segment, reference, {}).compared_tokens == 0

    def test_rate(self):
        reference = {"ቤቶች": (ReferenceAnalysis(("ቤት", "ኦች"), ()),)}
        segment = segmenter({"ቤቶች": Segmentation("ቤቶች", "ቤት", suffixes=("ኦች",))})
        assert inflection_agreement(segment, reference, {"ቤቶች": 2}).rate == 1.0

    def test_rate_with_nothing_compared_is_none(self):
        assert inflection_agreement(segmenter({}), {}, {}).rate is None
