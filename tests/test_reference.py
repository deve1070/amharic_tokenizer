"""
Tests for the IRIT reference lexicons.

The reference is a hand annotation, so the reader has two jobs: parse
the annotation format faithfully, and refuse rows that do not follow it
rather than guess at what they meant.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_morph.morphology import Segmentation  # noqa: E402
from amharic_morph.reference import (  # noqa: E402
    ReferenceAnalysis,
    inflection_agreement,
    propose_root,
    root_review_rows,
    root_votes,
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

    def test_al_is_the_negative_prefix_only_with_m(self):
        # አለ is annotated አል_ኧ: there አል is the verb stem, not the
        # negative, which Level 4 accepts only paired with ም.
        prefixes, core, _ = outer_affixes(ReferenceAnalysis(("አል", "ኧ"), ()))
        assert prefixes == ()
        assert core == ("አል", "ኧ")

    def test_negative_al_is_peeled_when_m_follows(self):
        prefixes, core, suffixes = outer_affixes(ReferenceAnalysis(("አል", "ሆን", "ኧ", "ም"), ()))
        assert prefixes == ("አል",)
        assert suffixes == ("ም",)

    def test_derivational_at_is_not_peeled(self):
        # አመታት is አመት_አት, a Ge'ez plural, not the verb object ኣት.
        _, core, suffixes = outer_affixes(ReferenceAnalysis(("አመት", "አት"), ()))
        assert core == ("አመት", "አት")
        assert suffixes == ()

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

    def test_gerund_counts_on_a_verb(self):
        # ሆኖ is ሆን_ኦ in the reference and ሆነ + ኦ in Level 4.
        reference = {"ሆኖ": (ReferenceAnalysis(("ሆን", "ኦ"), ()),)}
        segment = segmenter({"ሆኖ": Segmentation("ሆኖ", "ሆነ", suffixes=("ኦ",), category="verb")})
        assert inflection_agreement(segment, reference, {"ሆኖ": 3}).agreeing_tokens == 3

    def test_o_on_a_noun_is_left_as_derivation(self):
        # ጉዞ is ጉዝ_ኦ in the reference: derivation, which Level 4 keeps
        # inside the entry ጉዞ.
        reference = {"ጉዞ": (ReferenceAnalysis(("ጉዝ", "ኦ"), ()),)}
        segment = segmenter({"ጉዞ": Segmentation("ጉዞ", "ጉዞ", category="noun")})
        assert inflection_agreement(segment, reference, {"ጉዞ": 3}).agreeing_tokens == 3

    def test_rate(self):
        reference = {"ቤቶች": (ReferenceAnalysis(("ቤት", "ኦች"), ()),)}
        segment = segmenter({"ቤቶች": Segmentation("ቤቶች", "ቤት", suffixes=("ኦች",))})
        assert inflection_agreement(segment, reference, {"ቤቶች": 2}).rate == 1.0

    def test_rate_with_nothing_compared_is_none(self):
        assert inflection_agreement(segmenter({}), {}, {}).rate is None


# ── Root proposals ────────────────────────────────────────────────────────────


def rooted(*radicals, morphemes=None):
    return ReferenceAnalysis(morphemes or ("-".join(radicals),), (tuple(radicals),))


class TestRootVotes:
    """Every inflected form of an entry votes for its root"""

    def test_votes_are_pooled_by_stem_and_weighted_by_corpus_count(self):
        reference = {
            "ሞቱ": (rooted("ም", "ው", "ት", morphemes=("ም-ው-ት", "ኡ")),),
            "ሞት": (rooted("ም", "ው", "ት"),),
        }
        segment = segmenter({
            "ሞቱ": Segmentation("ሞቱ", "ሞት", suffixes=("ኡ",)),
            "ሞት": Segmentation("ሞት", "ሞት"),
        })
        votes = root_votes(segment, reference, {"ሞቱ": 3, "ሞት": 5})
        assert votes["ሞት"][("ም", "ው", "ት")] == 8

    def test_a_form_absent_from_the_corpus_still_counts_once(self):
        reference = {"ሞቱ": (rooted("ም", "ው", "ት", morphemes=("ም-ው-ት", "ኡ")),)}
        segment = segmenter({"ሞቱ": Segmentation("ሞቱ", "ሞት", suffixes=("ኡ",))})
        assert root_votes(segment, reference, {})["ሞት"][("ም", "ው", "ት")] == 1

    def test_a_form_with_no_root_votes_none(self):
        reference = {"ሀገሩ": (ReferenceAnalysis(("ሀገር", "ኡ")),)}
        segment = segmenter({"ሀገሩ": Segmentation("ሀገሩ", "ሀገር", suffixes=("ኡ",))})
        assert root_votes(segment, reference, {"ሀገሩ": 2})["ሀገር"] == {None: 2}

    def test_unknown_words_do_not_vote(self):
        assert root_votes(segmenter({}), {"ሆስፒታል": (rooted("ህ", "ስ"),)}, {}) == {}

    def test_only_the_primary_reading_votes(self):
        # ላይ is annotated ላይ[ለ_አይ]. The alternative reads the string as
        # ለ + አየ "to see", a different word; its root is not ላይ's.
        reference = {"ላይ": (ReferenceAnalysis(("ላይ",)), rooted("እ", "ይ", morphemes=("ለ", "እ-ይ")))}
        segment = segmenter({"ላይ": Segmentation("ላይ", "ላይ")})
        assert root_votes(segment, reference, {"ላይ": 4})["ላይ"] == {None: 4}

    def test_a_form_whose_affixes_disagree_does_not_vote(self):
        # Level 4 reads የለምና as የ + ለ + ምን + ኣ; the reference reads it as
        # የ + ል-ም + ና, so the two disagree on what the stem is.
        reference = {"የለምና": (rooted("ል", "ም", morphemes=("የ", "ል-ም", "ና")),)}
        segment = segmenter({
            "የለምና": Segmentation("የለምና", "ምን", prefixes=("የ", "ለ"), suffixes=("ኣ",)),
        })
        assert root_votes(segment, reference, {"የለምና": 814}) == {}

    def test_a_form_with_an_infix_does_not_vote(self):
        # The reference does not mark the infix, so the vote cannot be
        # checked against it.
        reference = {"አባባሎች": (rooted("እ", "ል", morphemes=("አ", "ባ", "ብ", "እ-ል", "ኦች")),)}
        segment = segmenter({
            "አባባሎች": Segmentation("አባባሎች", "አባል", suffixes=("ኦች",), infix="ባ"),
        })
        assert root_votes(segment, reference, {}) == {}

    def test_malformed_root_does_not_vote_for_itself(self):
        reference = {"ከተማ": (rooted("ክ", "ት", "ምከተማ"),)}
        segment = segmenter({"ከተማ": Segmentation("ከተማ", "ከተማ")})
        assert root_votes(segment, reference, {})["ከተማ"] == {None: 1}

    def test_labiovelar_radical_becomes_its_plain_consonant(self):
        # Q5: ቋ is ቅ at Level 1, so IRIT's ቍ is read as ቅ.
        reference = {"አቋም": (rooted("እ", "ቍ", "ም"),)}
        segment = segmenter({"አቋም": Segmentation("አቋም", "አቋም")})
        assert ("እ", "ቅ", "ም") in root_votes(segment, reference, {})["አቋም"]


class TestProposeRoot:
    def test_unanimous_vote_is_proposed(self):
        proposal = propose_root("ሞት", Counter({("ም", "ው", "ት"): 9}))
        assert proposal.root == ("ም", "ው", "ት")
        assert proposal.share == 1.0

    def test_winner_by_weight(self):
        proposal = propose_root("ሙሉ", Counter({("ም", "ል"): 86, ("ም", "ው", "ል"): 14}))
        assert proposal.root == ("ም", "ል")
        assert proposal.share == 0.86

    def test_none_votes_do_not_dilute_the_share(self):
        # IRIT leaves many forms unanalysed; that says nothing against the
        # root the analysed forms agree on.
        proposal = propose_root("ሞት", Counter({("ም", "ው", "ት"): 4, None: 96}))
        assert proposal.share == 1.0

    def test_no_root_votes_falls_back_to_the_skeleton(self):
        proposal = propose_root("ሀገር", Counter({None: 5}))
        assert proposal.root == ("ህ", "ግ", "ር")
        assert "no IRIT root" in proposal.checks

    def test_no_votes_at_all_falls_back_to_the_skeleton(self):
        assert propose_root("ቤት", Counter()).root == ("ብ", "ት")

    def test_split_vote_is_flagged(self):
        proposal = propose_root("ሙሉ", Counter({("ም", "ል"): 60, ("ም", "ው", "ል"): 40}))
        assert "split vote" in proposal.checks

    def test_clear_majority_is_not_split(self):
        proposal = propose_root("ሙሉ", Counter({("ም", "ል"): 80, ("ም", "ው", "ል"): 20}))
        assert "split vote" not in proposal.checks

    def test_difference_from_skeleton_is_flagged(self):
        proposal = propose_root("መሆን", Counter({("ህ", "ን"): 3}))
        assert "differs from skeleton" in proposal.checks

    def test_agreement_with_skeleton_is_not_flagged(self):
        assert propose_root("ፍቅር", Counter({("ፍ", "ቅ", "ር"): 3})).checks == ()

    def test_hidden_w_is_flagged_when_the_root_lacks_it(self):
        # Q2: ሆነ is ህ-ው-ን. ሆ carries ኦ, the vowel a hidden ው leaves.
        proposal = propose_root("ሆነ", Counter({("ህ", "ን"): 3}))
        assert "hidden ው?" in proposal.checks

    def test_hidden_w_is_not_flagged_when_the_root_has_it(self):
        proposal = propose_root("ሞት", Counter({("ም", "ው", "ት"): 3}))
        assert "hidden ው?" not in proposal.checks

    def test_hidden_y_is_flagged_on_e_and_i_vowels(self):
        assert "hidden ይ?" in propose_root("ቤት", Counter()).checks

    def test_glide_of_a_labiovelar_is_not_a_hidden_radical(self):
        # Q5: the ዋ of ቋ is dropped, so ቋንቋ is not asked about ው.
        assert "hidden ው?" not in propose_root("ቋንቋ", Counter()).checks


class TestRootReviewRows:
    def rows(self):
        votes = {
            "ፍቅር": Counter({("ፍ", "ቅ", "ር"): 50}),
            "መሆን": Counter({("ህ", "ን"): 900}),
            "ሙሉ": Counter({("ም", "ል"): 60, ("ም", "ው", "ል"): 40}),
            "ሀገር": Counter({None: 70}),
        }
        return root_review_rows(["ፍቅር", "መሆን", "ሙሉ", "ሀገር", "ቤት"], votes)

    def test_one_row_per_entry(self):
        assert sorted(r["word"] for r in self.rows()) == ["ሀገር", "መሆን", "ሙሉ", "ቤት", "ፍቅር"]

    def test_rows_needing_a_decision_come_first(self):
        # split vote, then differs from skeleton, then no IRIT root, then
        # plain agreement.
        assert [r["word"] for r in self.rows()] == ["ሙሉ", "መሆን", "ሀገር", "ቤት", "ፍቅር"]

    def test_row_fields_are_written_out(self):
        row = next(r for r in self.rows() if r["word"] == "ሙሉ")
        assert row["proposed"] == "ም-ል"
        assert row["skeleton"] == "ም-ል"
        assert row["irit"] == "ም-ል 60%; ም-ው-ል 40%"
        assert "split vote" in row["check"]
        assert row["reviewed"] == ""

    def test_no_root_is_reported(self):
        row = next(r for r in self.rows() if r["word"] == "ሀገር")
        assert row["irit"] == "no root given"

    def test_unanalysed_share_is_reported_next_to_the_roots(self):
        rows = root_review_rows(["ሞት"], {"ሞት": Counter({("ም", "ው", "ት"): 30, None: 10})})
        assert rows[0]["irit"] == "ም-ው-ት 100%; no root on 25%"

    def test_shares_that_round_to_zero_are_left_out(self):
        votes = {"ሞት": Counter({("ም", "ው", "ት"): 999, ("ም", "ት"): 1, None: 1})}
        assert root_review_rows(["ሞት"], votes)[0]["irit"] == "ም-ው-ት 100%"
