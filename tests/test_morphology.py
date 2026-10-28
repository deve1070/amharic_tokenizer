"""
Tests for the affix inventories and Level 4 morpheme segmentation.

Level 4 anchors on the lexicon: a stem is a morpheme because a
validated entry says so, never because the search could reach it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_morph.morphology import (  # noqa: E402
    INDEPENDENT,
    PREFIX,
    SUFFIX,
    Segmentation,
    detach_infix,
    detach_prefix,
    detach_suffix,
    rank_analyses,
    segment_all,
    segment_word,
    strip_prefixes,
    strip_suffixes,
    suffix_function,
)


def lexicon(*words, **entries):
    """Minimal lexicon index: a word with no recorded affixes."""
    index = {w: {"word": w, "allows_prefix": [], "allows_suffix": []} for w in words}
    index.update(entries)
    return index


# ── Affix removal ─────────────────────────────────────────────────────────────


class TestStripPrefixes:
    """Prefixes concatenate, so removal is plain string slicing"""

    def test_ye_prefix(self):
        assert "ቤት" in strip_prefixes("የቤት")

    def test_be_prefix(self):
        assert "ቤት" in strip_prefixes("በቤት")

    def test_multi_character_prefix(self):
        assert "ሰዎች" in strip_prefixes("ስለሰዎች")

    def test_no_prefix(self):
        assert strip_prefixes("ቤት") == set()

    def test_prefix_alone_is_not_stripped(self):
        # Stripping የ from የ would leave nothing.
        assert strip_prefixes("የ") == set()


class TestStripSuffixesPlain:
    """Suffixes that concatenate without altering the stem"""

    def test_accusative_n(self):
        assert "ቤት" in strip_suffixes("ቤትን")

    def test_woch_plural(self):
        assert "ኢትዮጵያ" in strip_suffixes("ኢትዮጵያዎች")

    def test_no_suffix(self):
        assert strip_suffixes("ቤት") == set()


class TestStripSuffixesFused:
    """
    ኦች and ኡ fuse their vowel into the stem's final consonant.

    ቤት + ኦች is written ቤቶች, not ቤትኦች — removing the literal characters
    would find nothing, so the fusion has to be inverted arithmetically.
    """

    def test_plural_fusion(self):
        assert "ቤት" in strip_suffixes("ቤቶች")

    def test_definite_fusion(self):
        assert "ቤት" in strip_suffixes("ቤቱ")

    def test_feminine_definite_fusion(self):
        assert "ቤት" in strip_suffixes("ቤታ")

    def test_plural_fusion_other_family(self):
        # ልጅ + ኦች -> ልጆች
        assert "ልጅ" in strip_suffixes("ልጆች")

    def test_literal_removal_would_fail(self):
        # The naive approach finds nothing, which is why fusion exists.
        assert "ቤቶች".replace("ኦች", "") == "ቤቶች"

    def test_too_short_for_fusion(self):
        assert strip_suffixes("ች") == set()


class TestPrefixInventory:
    """
    ም and ና are listed as prefixes in the open questions but do not
    behave as prefixes in the corpus — being single characters, they
    strip the front off ordinary words.
    """

    def test_m_is_not_a_default_prefix(self):
        # ምግብ (food) must not decompose to ግብ.
        assert "ግብ" not in strip_prefixes("ምግብ")

    def test_na_is_not_a_default_prefix(self):
        assert "ት" not in strip_prefixes("ናት")

    def test_real_prefixes_still_work(self):
        assert "ቤት" in strip_prefixes("የቤት")
        assert "ሰዎች" in strip_prefixes("ስለሰዎች")

    def test_can_be_reenabled_explicitly(self):
        assert "ግብ" in strip_prefixes("ምግብ", prefixes=("ም",))

    def test_distributive_beye_is_one_prefix(self):
        # በየአመቱ "every year" is በየ + አመት + ኡ. Read as በ + የ it would
        # be "in the year's", a different word.
        result = segment_word("በየአመቱ", lexicon("አመት"))
        assert result.prefixes == ("በየ",)
        assert result.morphemes == ["በየ", "አመት", "ኡ"]

    def test_be_and_ye_still_attach_alone(self):
        assert segment_word("በቤት", lexicon("ቤት")).prefixes == ("በ",)
        assert segment_word("የቤት", lexicon("ቤት")).prefixes == ("የ",)

    def test_beye_does_not_cut_into_a_stem_starting_with_ye(self):
        # በየትኛው is በ + የትኛው "in which". With የትኛው an entry and ትኛው
        # not one, the lexicon keeps የ inside the stem.
        result = segment_word("በየትኛው", lexicon("የትኛው"))
        assert result.morphemes == ["በ", "የትኛው"]


# ── Affix removal with the morpheme named ────────────────────────────────────


class TestDetachPrefix:
    def test_reports_the_morpheme(self):
        assert detach_prefix("የቤት") == [("ቤት", "የ")]

    def test_multi_character_prefix(self):
        assert detach_prefix("ስለሰው") == [("ሰው", "ስለ")]

    def test_no_prefix(self):
        assert detach_prefix("ቤት") == []


class TestDetachSuffix:
    """
    Fused suffixes are reported by morpheme, not by the characters on
    the surface, because those characters belong to the stem as well.
    """

    def test_plain_suffix(self):
        assert ("ቤት", "ን") in detach_suffix("ቤትን")

    def test_fused_plural_names_the_morpheme(self):
        # The surface has no ኦ of its own; ቶ carries it.
        assert ("ቤት", "ኦች") in detach_suffix("ቤቶች")

    def test_fused_definite_names_the_morpheme(self):
        assert ("ቤት", "ኡ") in detach_suffix("ቤቱ")

    def test_feminine_definite(self):
        assert ("ቤት", "ኣ") in detach_suffix("ቤታ")

    def test_no_suffix(self):
        assert detach_suffix("ቤት") == []


class TestFusionIsOneToMany:
    """
    A fused carrier does not record the stem's own final vowel, so both
    readings are offered and the lexicon chooses.
    """

    def test_consonant_final_stem_offered(self):
        assert ("ሀገር", "ኦች") in detach_suffix("ሀገሮች")

    def test_vowel_final_stem_offered(self):
        # ከተማ + ኦች -> ከተሞች. Reducing to order 6 alone would give ከተም.
        assert ("ከተማ", "ኦች") in detach_suffix("ከተሞች")

    def test_every_stem_final_order_is_offered(self):
        stems = {s for s, m in detach_suffix("ከተሞች") if m == "ኦች"}
        assert stems == {"ከተም", "ከተማ", "ከተመ"}

    def test_verb_perfective_stem_is_reachable(self):
        # ሆነ + ኡ -> ሆኑ. The ኡ overwrites the ኧ of ነ, so the carrier
        # restores to order 1, not to the bare consonant.
        assert ("ሆነ", "ኡ") in detach_suffix("ሆኑ")

    def test_noun_stem_is_reachable_from_the_same_shape(self):
        assert ("ቤት", "ኡ") in detach_suffix("ቤቱ")

    def test_the_lexicon_picks_the_right_one(self):
        assert segment_word("ከተሞች", lexicon("ከተማ")).morphemes == ["ከተማ", "ኦች"]


class TestDetachInfix:
    """
    There is no infix inventory. The inserted fidel is a copy of the
    consonant that follows it, carrying ኣ, so it is always that
    consonant's order-4 form.
    """

    def test_spec_example(self):
        assert detach_infix("ቀጫጭን") == [("ቀጭን", "ጫ")]

    def test_two_fidel_base(self):
        assert detach_infix("ነጫጭ") == [("ነጭ", "ጫ")]

    def test_the_inserted_fidel_is_not_always_the_same(self):
        # Reading ጫ off the spec's one example would miss all of these.
        assert detach_infix("ትላልቅ") == [("ትልቅ", "ላ")]
        assert detach_infix("ረጃጅም") == [("ረጅም", "ጃ")]
        assert detach_infix("ጥቃቁር") == [("ጥቁር", "ቃ")]
        assert detach_infix("ቀያይ") == [("ቀይ", "ያ")]

    def test_copy_may_sit_past_the_first_radical(self):
        assert detach_infix("ወፍራራም") == [("ወፍራም", "ራ")]

    def test_wrong_order_is_not_an_infix(self):
        # ጭ before ጭ is the same family but not order 4.
        assert detach_infix("ቀጭጭን") == []

    def test_different_family_is_not_an_infix(self):
        assert detach_infix("ቀጫልን") == []

    def test_word_initial_position_is_not_an_infix(self):
        assert detach_infix("ጫማ") == []

    def test_absent_infix(self):
        assert detach_infix("ቤቶች") == []


# ── Level 4 ───────────────────────────────────────────────────────────────────


class TestSegmentWord:
    """The spec's worked example and the shapes around it"""

    def test_spec_example(self):
        assert segment_word("ቤቶች", lexicon("ቤት")).morphemes == ["ቤት", "ኦች"]

    def test_bare_stem_is_one_morpheme(self):
        assert segment_word("ቤት", lexicon("ቤት")).morphemes == ["ቤት"]

    def test_prefix(self):
        assert segment_word("የቤት", lexicon("ቤት")).morphemes == ["የ", "ቤት"]

    def test_prefix_and_suffix(self):
        result = segment_word("የቤቶች", lexicon("ቤት"))
        assert result.morphemes == ["የ", "ቤት", "ኦች"]

    def test_prefixes_are_in_surface_order(self):
        assert segment_word("ስለሰው", lexicon("ሰው")).prefixes == ("ስለ",)

    def test_suffixes_are_in_surface_order(self):
        # ቤቶችም is ቤት + ኦች + ም, stripped right to left but reported left
        # to right.
        result = segment_word("ቤቶችም", lexicon("ቤት"))
        assert result.suffixes == ("ኦች", "ም")
        assert result.morphemes == ["ቤት", "ኦች", "ም"]

    def test_the_stem_is_the_independent_morpheme(self):
        assert segment_word("የቤቶች", lexicon("ቤት")).stem == "ቤት"

    def test_infix(self):
        result = segment_word("ቀጫጭን", lexicon("ቀጭን"))
        assert result.morphemes == ["ቀጭን", "ጫ"]
        assert result.infix == "ጫ"

    def test_infix_on_another_stem(self):
        assert segment_word("ትላልቅ", lexicon("ትልቅ")).morphemes == ["ትልቅ", "ላ"]

    def test_infix_with_a_suffix(self):
        result = segment_word("ትላልቆች", lexicon("ትልቅ"))
        assert result.morphemes == ["ትልቅ", "ላ", "ኦች"]

    def test_empty_word(self):
        assert segment_word("", lexicon("ቤት")).known is False


class TestUnknownWords:
    """
    Open question 6: a word with no lexicon anchor comes back whole
    rather than segmented on a guess.
    """

    def test_returned_whole(self):
        result = segment_word("ኮምፒውተር", lexicon("ቤት"))
        assert result.morphemes == ["ኮምፒውተር"]

    def test_flagged_not_known(self):
        assert segment_word("ኮምፒውተር", lexicon("ቤት")).known is False

    def test_no_affixes_invented(self):
        # ኮምፒውተርን strips to ኮምፒውተር, which is not in the lexicon, so the
        # accusative is not reported either.
        result = segment_word("ኮምፒውተርን", lexicon("ቤት"))
        assert result.morphemes == ["ኮምፒውተርን"]
        assert result.known is False

    def test_a_known_word_is_flagged_known(self):
        assert segment_word("ቤቶች", lexicon("ቤት")).known is True


class TestLexiconIsTheAuthority:
    """Nothing is a morpheme because the search could reach it"""

    def test_empty_lexicon_segments_nothing(self):
        assert segment_word("ቤቶች", {}).morphemes == ["ቤቶች"]

    def test_a_word_in_the_lexicon_stays_whole(self):
        # ስራ is an entry, so it is not ስር + ኣ even though that parses.
        result = segment_word("ስራ", lexicon("ስራ", "ስር"))
        assert result.morphemes == ["ስራ"]

    def test_without_the_entry_the_parse_is_taken(self):
        assert segment_word("ስራ", lexicon("ስር")).morphemes == ["ስር", "ኣ"]


class TestRanking:
    """Fewest morphemes, then attested affixes, then longest stem"""

    def test_fewest_morphemes_wins(self):
        analyses = segment_all("ቤቶች", lexicon("ቤቶች", "ቤት"))
        assert analyses[0].morphemes == ["ቤቶች"]

    def test_attested_affixes_break_the_tie(self):
        # Both parse as one prefix plus a two-fidel stem; only ቤት
        # records የ as a prefix it takes.
        index = {
            "ቤት": {"allows_prefix": ["የ"], "allows_suffix": []},
            "ቤቶ": {"allows_prefix": [], "allows_suffix": []},
        }
        best = segment_word("የቤቶች", index)
        assert best.stem == "ቤት"

    def test_longest_stem_breaks_a_remaining_tie(self):
        index = {
            "ሀገር": {"allows_prefix": [], "allows_suffix": []},
            "ሀገሮ": {"allows_prefix": [], "allows_suffix": []},
        }
        assert segment_word("ሀገሮች", index).stem == "ሀገር"

    def test_ranking_is_deterministic(self):
        index = lexicon("ቤት", "ቤቶ", "ቤታ")
        first = [a.morphemes for a in segment_all("ቤቶች", index)]
        second = [a.morphemes for a in segment_all("ቤቶች", index)]
        assert first == second

    def test_rank_analyses_orders_a_given_list(self):
        long = Segmentation("የቤቶች", "ቤት", prefixes=("የ",), suffixes=("ኦች",))
        short = Segmentation("የቤቶች", "የቤቶች")
        assert rank_analyses([long, short], {})[0] is short


class TestSegmentAll:
    """Ambiguity that segment_word() resolves silently"""

    def test_lists_every_analysis(self):
        analyses = segment_all("ቤቶች", lexicon("ቤት", "ቤታ"))
        assert {tuple(a.morphemes) for a in analyses} == {
            ("ቤት", "ኦች"),
            ("ቤታ", "ኦች"),
            # Stacking reaches ቤታ and then strips its ኣ as well.
            ("ቤት", "ኣ", "ኦች"),
        }

    def test_the_extra_analyses_rank_below_the_short_ones(self):
        analyses = segment_all("ቤቶች", lexicon("ቤት", "ቤታ"))
        assert analyses[-1].morphemes == ["ቤት", "ኣ", "ኦች"]

    def test_empty_when_no_anchor(self):
        assert segment_all("ኮምፒውተር", lexicon("ቤት")) == []

    def test_best_first(self):
        analyses = segment_all("ቤቶች", lexicon("ቤቶች", "ቤት"))
        assert analyses[0].morpheme_count <= analyses[-1].morpheme_count


class TestStackingIsBounded:
    """Affixes stack, but not without limit"""

    def test_two_suffixes(self):
        assert segment_word("ቤቶችን", lexicon("ቤት")).suffixes == ("ኦች", "ን")

    def test_suffix_depth_is_configurable(self):
        assert segment_word("ቤቶችም", lexicon("ቤት"), max_suffixes=1).known is False

    def test_prefix_depth_is_configurable(self):
        assert segment_word("የቤት", lexicon("ቤት"), max_prefixes=0).known is False


class TestSegmentationShape:
    """The record Levels 3, 2 and 1 are computed from"""

    def test_morpheme_count(self):
        assert Segmentation("ቤቶች", "ቤት", suffixes=("ኦች",)).morpheme_count == 2

    def test_infix_follows_its_stem(self):
        seg = Segmentation("ቀጫጭን", "ቀጭን", infix="ጫ", suffixes=("ኦች",))
        assert seg.morphemes == ["ቀጭን", "ጫ", "ኦች"]

    def test_is_hashable(self):
        assert len({Segmentation("ቤት", "ቤት"), Segmentation("ቤት", "ቤት")}) == 1


# ── One suffix, two functions ────────────────────────────────────────────────


class TestSuffixFunction:
    """
    ኡ attaches the same way to a noun and a verb and Level 4 finds the
    same boundary, but it is not the same morpheme. The stem's category
    is what decides.
    """

    def test_definite_on_a_noun(self):
        assert suffix_function("ኡ", "noun") == "definite"

    def test_agreement_on_a_verb(self):
        assert suffix_function("ኡ", "verb") == "subject_3pl"

    def test_w_splits_the_same_way(self):
        assert suffix_function("ው", "noun") == "definite"
        assert suffix_function("ው", "verb") == "subject_3ms"

    def test_unfilled_category_is_not_guessed(self):
        # A guess would put a definite article on a verb.
        assert suffix_function("ኡ", None) is None

    def test_unambiguous_suffix_gets_no_gloss(self):
        assert suffix_function("ኦች", "noun") is None

    def test_adjectives_take_the_definite_reading(self):
        assert suffix_function("ኡ", "adjective") == "definite"


class TestFunctionsInASegmentation:
    def test_noun_definite(self):
        index = {"ቤት": {"category": "noun", "allows_suffix": ["ኡ"]}}
        parts = segment_word("ቤቱ", index).parts
        assert [(p.text, p.function) for p in parts] == [
            ("ቤት", None),
            ("ኡ", "definite"),
        ]

    def test_verb_plural(self):
        index = {"ሆነ": {"category": "verb", "allows_suffix": ["ኡ"]}}
        parts = segment_word("ሆኑ", index).parts
        assert [(p.text, p.function) for p in parts] == [
            ("ሆነ", None),
            ("ኡ", "subject_3pl"),
        ]

    def test_same_boundary_either_way(self):
        noun = segment_word("ቤቱ", {"ቤት": {"category": "noun"}})
        verb = segment_word("ሆኑ", {"ሆነ": {"category": "verb"}})
        assert noun.suffixes == verb.suffixes == ("ኡ",)

    def test_morphemes_stay_amharic_strings(self):
        # The function lives beside the morpheme, not inside it, so
        # Levels 3, 2 and 1 can still decompose the text.
        index = {"ቤት": {"category": "noun"}}
        assert segment_word("ቤቱ", index).morphemes == ["ቤት", "ኡ"]

    def test_kinds_are_reported(self):
        index = {"ቤት": {"category": "noun"}}
        parts = segment_word("የቤቶች", index).parts
        assert [p.kind for p in parts] == [PREFIX, INDEPENDENT, SUFFIX]

    def test_unknown_word_has_one_independent_part(self):
        parts = segment_word("ኮምፒውተር", {}).parts
        assert [(p.text, p.kind) for p in parts] == [("ኮምፒውተር", INDEPENDENT)]


# ── One implementation of suffix inversion ───────────────────────────────────

from amharic_morph.morphology import (  # noqa: E402
    RULE_AMBIGUOUS_FUSION,
    RULE_DEFINITE_FUSION,
    RULE_PLURAL_FUSION,
    RULE_SUFFIX,
    detach_suffix_with_rule,
)


class TestDetachSuffixWithRule:
    """The rule-tagged form the corpus analysis reads"""

    def test_plain_suffix(self):
        assert ("ቤት", "ን", RULE_SUFFIX) in detach_suffix_with_rule("ቤትን")

    def test_plural_fusion(self):
        assert ("ቤት", "ኦች", RULE_PLURAL_FUSION) in detach_suffix_with_rule("ቤቶች")

    def test_order2_definite_is_unambiguous(self):
        assert ("ቤት", "ኡ", RULE_DEFINITE_FUSION) in detach_suffix_with_rule("ቤቱ")

    def test_order4_definite_is_ambiguous(self):
        # ስራ is a word in its own right, not the definite of ስር.
        assert ("ስር", "ኣ", RULE_AMBIGUOUS_FUSION) in detach_suffix_with_rule("ስራ")

    def test_reaches_a_verb_stem_at_order1(self):
        # ሆኑ is ሆነ + ኡ. Restoring the carrier to order 6 alone gives ሆን,
        # which is where the corpus analysis and the segmenter used to
        # disagree.
        stems = [stem for stem, _, _ in detach_suffix_with_rule("ሆኑ")]
        assert "ሆነ" in stems and "ሆን" in stems

    def test_reaches_an_order4_stem_through_the_plural(self):
        stems = [stem for stem, _, _ in detach_suffix_with_rule("ከተሞች")]
        assert "ከተማ" in stems

    def test_agrees_with_detach_suffix(self):
        for word in ("ቤቶች", "ቤቱ", "ቤትን", "ሆኑ", "ከተሞች", "ስራ"):
            assert detach_suffix(word) == [
                (stem, morpheme) for stem, morpheme, _ in detach_suffix_with_rule(word)
            ]
