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
    suffix_readings,
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

    def test_bare_a_is_not_a_noun_suffix(self):
        # Ruled 2026-10-22: on a noun ኣ is only the first sound of ኣችን,
        # ኣቸው, ኣችሁ and ኣማ. ቤታ does not reduce to ቤት.
        assert "ቤት" not in strip_suffixes("ቤታ")

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

    def test_bare_a_is_not_in_the_corpus_view(self):
        # Bare ኣ is the 3fs gerund on a reviewed gerund stem (ሆን + ኣ ->
        # ሆና). The corpus view has no stems, so it never offers it.
        assert all(m != "ኣ" for _, m in detach_suffix("ቤታ"))
        assert ("ቤት", "ኣ") in [(s, r.morpheme) for s, r in detach_suffix_shapes("ቤታ")]

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
        assert segment_word("ቤቶች", lexicon("ቤት")).morphemes == ["ቤት", "ኦች"]

    def test_ser_plus_a_is_not_a_parse(self):
        # Bare ኣ is not a noun suffix, so without ስራ in the lexicon the
        # word is unknown rather than ስር + ኣ.
        assert segment_word("ስራ", lexicon("ስር")).known is False


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
        assert {tuple(a.morphemes) for a in analyses} == {("ቤት", "ኦች"), ("ቤታ", "ኦች")}

    def test_the_extra_analyses_rank_below_the_short_ones(self):
        analyses = segment_all("ቤቶችን", lexicon("ቤት", "ቤቶች"))
        assert analyses[0].morphemes == ["ቤቶች", "ን"]
        assert analyses[-1].morphemes == ["ቤት", "ኦች", "ን"]

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


class TestSuffixReadings:
    """
    A suffix carries a tuple of readings, chosen by the stem's category:
    ኡ is the definite or "his" on a noun and the 3pl subject on a verb.
    The table is the one the native speaker reviewed 2026-10-29.
    """

    def seg(self, stem, category, suffixes=(), prefixes=()):
        return Segmentation("w", stem, prefixes=prefixes, suffixes=suffixes, category=category)

    def test_u_on_a_noun(self):
        assert suffix_readings(self.seg("ቤት", "noun", ("ኡ",)), "ኡ") == ("definite", "possessive_3ms")

    def test_u_on_a_verb(self):
        assert suffix_readings(self.seg("ሆነ", "verb", ("ኡ",)), "ኡ") == ("subject_3pl",)

    def test_w_on_a_verb_has_three_readings(self):
        readings = suffix_readings(self.seg("ሰጠ", "verb", ("ው",)), "ው")
        assert readings == ("object_3ms", "subject_3pl", "definite")

    def test_adjectives_read_as_non_verbs(self):
        assert suffix_readings(self.seg("ትልቅ", "adjective", ("ኡ",)), "ኡ")[0] == "definite"

    def test_unfilled_category_is_not_guessed(self):
        # A guess would put a definite article on a verb.
        assert suffix_readings(self.seg("ቤት", None, ("ኡ",)), "ኡ") is None

    def test_every_suffix_has_readings(self):
        # Even unambiguous suffixes, so "no meaning recorded" never looks
        # like "meaning unknown".
        assert suffix_readings(self.seg("ቤት", "noun", ("ኦች",)), "ኦች") == ("plural",)
        assert suffix_readings(self.seg("ሄደ", "verb", ("ኩ",)), "ኩ") == ("subject_1s",)

    def test_n_on_a_noun_and_a_verb(self):
        assert suffix_readings(self.seg("ቤት", "noun", ("ን",)), "ን") == ("accusative",)
        assert suffix_readings(self.seg("ሰጠ", "verb", ("ን",)), "ን") == ("subject_1pl", "object_1pl")

    def test_m_is_negative_only_with_al(self):
        negative = self.seg("ሆነ", "verb", ("ም",), ("አል",))
        assert suffix_readings(negative, "ም") == ("negative",)
        assert suffix_readings(self.seg("ሆነ", "verb", ("ኦ", "ም")), "ም") == ("also",)
        assert suffix_readings(self.seg("ቤት", "noun", ("ም",)), "ም") == ("also",)

    def test_every_table_row_has_readings_for_its_kind(self):
        for shape in SUFFIX_TABLE:
            if shape.kind in (NOUN, ANY) and shape.kind != ANY:
                assert suffix_readings(self.seg("x", "noun", (shape.morpheme,)), shape.morpheme)
            if shape.kind == VERB:
                assert suffix_readings(self.seg("x", "verb", (shape.morpheme,)), shape.morpheme)
            if shape.kind == ANY:
                assert (suffix_readings(self.seg("x", "noun", (shape.morpheme,)), shape.morpheme)
                        or suffix_readings(self.seg("x", "verb", (shape.morpheme,)), shape.morpheme))


class TestReadingsInASegmentation:
    def test_noun_definite(self):
        parts = segment_word("ቤቱ", typed(ቤት="noun")).parts
        assert [(p.text, p.readings) for p in parts] == [
            ("ቤት", None),
            ("ኡ", ("definite", "possessive_3ms")),
        ]

    def test_verb_plural(self):
        parts = segment_word("ሆኑ", typed(ሆነ="verb")).parts
        assert [(p.text, p.readings) for p in parts] == [("ሆነ", None), ("ኡ", ("subject_3pl",))]

    def test_negative_prefix_and_suffix(self):
        parts = segment_word("አልሆነም", typed(ሆነ="verb")).parts
        assert [(p.text, p.readings) for p in parts] == [
            ("አል", ("negative",)),
            ("ሆነ", None),
            ("ም", ("negative",)),
        ]

    def test_other_prefixes_carry_no_readings(self):
        # The reviewed table covers suffixes and the negative prefix only.
        assert segment_word("የቤቶች", typed(ቤት="noun")).parts[0].readings is None

    def test_same_boundary_either_way(self):
        noun = segment_word("ቤቱ", typed(ቤት="noun"))
        verb = segment_word("ሆኑ", typed(ሆነ="verb"))
        assert noun.suffixes == verb.suffixes == ("ኡ",)

    def test_morphemes_stay_amharic_strings(self):
        # The readings live beside the morpheme, not inside it, so Levels
        # 3, 2 and 1 can still decompose the text.
        assert segment_word("ቤቱ", typed(ቤት="noun")).morphemes == ["ቤት", "ኡ"]

    def test_kinds_are_reported(self):
        parts = segment_word("የቤቶች", typed(ቤት="noun")).parts
        assert [p.kind for p in parts] == [PREFIX, INDEPENDENT, SUFFIX]

    def test_unknown_word_has_one_independent_part(self):
        parts = segment_word("ኮምፒውተር", {}).parts
        assert [(p.text, p.kind) for p in parts] == [("ኮምፒውተር", INDEPENDENT)]


class TestPlainChIsVerbOnly:
    """
    Plain ች is the 3fs subject (ሄደች). On a non-verb it produced only
    wrong analyses: አይቼ -> አይ + ች + ኤ, ማርች -> ማር + ች.
    """

    def test_on_a_verb(self):
        assert segment_word("ሄደች", typed(ሄደ="verb")).morphemes == ["ሄደ", "ች"]

    def test_refused_on_a_noun(self):
        assert segment_word("ማርች", typed(ማር="noun")).known is False


# ── One implementation of suffix inversion ───────────────────────────────────

from amharic_morph.morphology import (  # noqa: E402
    RULE_AMBIGUOUS_FUSION,
    RULE_DEFINITE_FUSION,
    RULE_FUSION,
    RULE_PLURAL_FUSION,
    RULE_SUFFIX,
    detach_suffix_with_rule,
)
from amharic_morph.morphology import SUFFIX_TABLE, detach_suffix_shapes  # noqa: E402


class TestDetachSuffixWithRule:
    """The rule-tagged form the corpus analysis reads"""

    def test_plain_suffix(self):
        assert ("ቤት", "ን", RULE_SUFFIX) in detach_suffix_with_rule("ቤትን")

    def test_plural_fusion(self):
        assert ("ቤት", "ኦች", RULE_PLURAL_FUSION) in detach_suffix_with_rule("ቤቶች")

    def test_order2_definite_is_unambiguous(self):
        assert ("ቤት", "ኡ", RULE_DEFINITE_FUSION) in detach_suffix_with_rule("ቤቱ")

    def test_bare_a_is_ambiguous(self):
        # Read from the full table, the gerund ኣ is tagged ambiguous, so it
        # never counts as a derivation: ስራ is a word in its own right.
        assert ("ስር", "ኣ", RULE_AMBIGUOUS_FUSION) in detach_suffix_with_rule("ስራ", SUFFIX_TABLE)

    def test_new_fused_rows_use_the_guarded_rule(self):
        assert ("ቤት", "ኣችን", RULE_FUSION) in detach_suffix_with_rule("ቤታችን")
        assert ("ሄደ", "ኩ", RULE_FUSION) in detach_suffix_with_rule("ሄድኩ")

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


# ── The suffix table ─────────────────────────────────────────────────────────

from amharic_morph.levels import is_ethiopic  # noqa: E402
from amharic_morph.morphology import (  # noqa: E402
    ANY,
    NOUN,
    SUFFIX_TABLE,
    VERB,
    SuffixShape,
)


class TestSuffixTable:
    """
    Every row inverts its own written shape. Detachment does not gate by
    category; segmentation does, once the stem is known.
    """

    def test_possessive_after_a_consonant(self):
        # ህዝብ + ኣችን -> ህዝባችን: ኣ turns ብ into ባ.
        assert ("ህዝብ", "ኣችን") in detach_suffix("ህዝባችን")

    def test_possessive_after_a_vowel(self):
        assert ("ስራ", "ችን") in detach_suffix("ስራችን")

    def test_possessive_glide_after_i_or_e(self):
        assert ("ፓርቲ", "ያችን") in detach_suffix("ፓርቲያችን")

    def test_plain_possessive_needs_a_vowel_before_it(self):
        assert ("ቤት", "ችን") not in detach_suffix("ቤትችን")

    def test_first_person_singular(self):
        assert ("ቤት", "ኤ") in detach_suffix("ቤቴ")
        assert ("ስራ", "ዬ") in detach_suffix("ስራዬ")

    def test_her(self):
        assert ("ቤት", "ዋ") in detach_suffix("ቤቷ")
        assert ("ከተማ", "ዋ") in detach_suffix("ከተማዋ")

    def test_their_on_a_noun_and_a_verb(self):
        assert ("ቤት", "ኣቸው") in detach_suffix("ቤታቸው")
        assert ("ሰጠ", "ኣቸው") in detach_suffix("ሰጣቸው")
        assert ("ስራ", "ቸው") in detach_suffix("ስራቸው")

    def test_your_plural(self):
        assert ("ቤት", "ኣችሁ") in detach_suffix("ቤታችሁ")

    def test_adjective_ama(self):
        assert ("ውጤት", "ኣማ") in detach_suffix("ውጤታማ")
        assert ("ጤና", "ማ") in detach_suffix("ጤናማ")

    def test_plain_ma_follows_only_an_a_final_stem(self):
        assert ("ቤት", "ማ") not in detach_suffix("ቤትማ")

    def test_perfective_subjects_delete_the_stem_vowel(self):
        assert ("ሄደ", "ኩ") in detach_suffix("ሄድኩ")
        assert ("ሄደ", "ክ") in detach_suffix("ሄድክ")
        assert ("ሄደ", "ሽ") in detach_suffix("ሄድሽ")
        assert ("ሄደ", "ን") in detach_suffix("ሄድን")

    def test_first_person_after_an_a_final_verb(self):
        assert ("ገባ", "ሁ") in detach_suffix("ገባሁ")

    def test_object_me(self):
        assert ("ሰጠ", "ኝ") in detach_suffix("ሰጠኝ")

    def test_object_her(self):
        assert ("ሰጠ", "ኣት") in detach_suffix("ሰጣት")

    def test_gerunds_leave_the_gerund_stem(self):
        assert ("ሆን", "ኦ") in detach_suffix("ሆኖ", SUFFIX_TABLE)
        assert ("ሆን", "ኣ") in detach_suffix("ሆና", SUFFIX_TABLE)
        assert ("ገብ", "ቶ") in detach_suffix("ገብቶ", SUFFIX_TABLE)
        assert ("ገብ", "ታ") in detach_suffix("ገብታ", SUFFIX_TABLE)

    def test_to_gerund_leaves_the_gerund_stem(self):
        # ሰጥቶ = ሰጥ + ቶ, the reviewed gerund stem of ሰጠ.
        assert ("ሰጥ", "ቶ") in detach_suffix("ሰጥቶ", SUFFIX_TABLE)
        assert ("ቀር", "ታ") in detach_suffix("ቀርታ", SUFFIX_TABLE)

    def test_gerund_o_is_not_the_plural(self):
        # Both leave the carrier at order 7; the plural needs ች after it.
        assert ("ቤት", "ኦ") not in detach_suffix("ቤቶች")

    def test_every_morpheme_is_written_in_ethiopic(self):
        # Levels 3 and 2 decompose morphemes, so a name must be Amharic.
        for shape in SUFFIX_TABLE:
            assert shape.morpheme and all(is_ethiopic(c) for c in shape.morpheme)

    def test_every_kind_is_known(self):
        assert {s.kind for s in SUFFIX_TABLE} <= {ANY, NOUN, VERB}

    def test_fused_rows_restore_to_an_order_the_carrier_can_leave(self):
        for shape in SUFFIX_TABLE:
            if shape.carrier_order is not None:
                assert shape.restore_orders
                assert shape.carrier_order not in shape.restore_orders

    def test_a_custom_table_replaces_the_default(self):
        only_n = (SuffixShape("ን", ANY, "ን", slot=3),)
        assert detach_suffix("ቤቶች", only_n) == []
        assert detach_suffix("ቤትን", only_n) == [("ቤት", "ን")]


def typed(**categories):
    """Lexicon index with a category per word."""
    return {w: {"word": w, "category": c, "allows_prefix": [], "allows_suffix": []}
            for w, c in categories.items()}


class TestCategoryGating:
    """
    A noun row needs a non-verb stem, a verb row a verb, and an entry
    whose category is not yet set accepts only the rows marked any.
    """

    def test_verb_suffix_on_a_verb(self):
        assert segment_word("ሄድኩ", typed(ሄደ="verb")).morphemes == ["ሄደ", "ኩ"]

    def test_verb_suffix_on_an_unset_category_is_refused(self):
        assert segment_word("ሄድኩ", typed(ሄደ=None)).known is False

    def test_unset_category_still_takes_shared_suffixes(self):
        assert segment_word("ሄደን", typed(ሄደ=None)).morphemes == ["ሄደ", "ን"]
        assert segment_word("ቤታቸው", typed(ቤት=None)).morphemes == ["ቤት", "ኣቸው"]

    def test_order1_restoration_does_not_make_a_noun_a_verb(self):
        # ቤትን could invert to ቤተ + ን (1pl); ቤተ is not a verb.
        result = segment_word("ቤትን", typed(ቤት="noun", ቤተ="noun"))
        assert result.morphemes == ["ቤት", "ን"]

    def test_possessive_on_a_noun(self):
        assert segment_word("ህዝባችን", typed(ህዝብ="noun")).morphemes == ["ህዝብ", "ኣችን"]

    def test_possessive_on_an_unset_category_is_refused(self):
        assert segment_word("ህዝባችን", typed(ህዝብ=None)).known is False

    def test_possessive_is_refused_on_a_verb(self):
        assert segment_word("ቤቴ", typed(ቤት="verb")).known is False

    def test_gerunds_need_a_reviewed_stem(self):
        # Without a stored gerund stem nothing is guessed; TestGerund
        # covers the forms with one.
        lex = typed(ሆነ="verb", ገባ="verb")
        assert segment_word("ሆኖ", lex).known is False
        assert segment_word("ገብቶ", lex).known is False


    def test_gerunds_are_refused_on_a_noun(self):
        lex = typed(ሆነ="noun")
        assert segment_word("ሆና", lex).known is False
        assert segment_word("ሆኖ", lex).known is False

    def test_bare_a_no_longer_splits_a_noun(self):
        # ነጋ used to be ነገ + ኣ.
        assert segment_word("ነጋ", typed(ነገ="noun")).known is False

    def test_ama_and_the_words_that_end_like_it(self):
        lex = typed(ውጤት="noun", ድር="noun", ድራማ="noun")
        assert segment_word("ውጤታማ", lex).morphemes == ["ውጤት", "ኣማ"]
        assert segment_word("ድራማ", lex).morphemes == ["ድራማ"]

    def test_possessive_beats_the_false_infix(self):
        # ጓደኞቻችን is ጓደኛ + ኦች + ኣችን. Without ኣችን the search took ቻ as
        # an infix: ጓደኛ + ቻ + ኦች + ን.
        result = segment_word("ጓደኞቻችን", typed(ጓደኛ="noun"))
        assert result.morphemes == ["ጓደኛ", "ኦች", "ኣችን"]


class TestNegativePerfective:
    """አል ... ም: a prefix accepted only on a verb that also carries ም"""

    def lex(self):
        return typed(ሆነ="verb", ሄደ="verb", ቤት="noun")

    def test_negative(self):
        assert segment_word("አልሆነም", self.lex()).morphemes == ["አል", "ሆነ", "ም"]

    def test_negative_with_a_subject(self):
        assert segment_word("አልሄድኩም", self.lex()).morphemes == ["አል", "ሄደ", "ኩ", "ም"]

    def test_a_conjunction_may_follow_the_negative_m(self):
        # አልሆነምና "since it did not become".
        result = segment_word("አልሆነምና", self.lex())
        assert result.morphemes == ["አል", "ሆነ", "ም", "ና"]

    def test_al_without_m_is_refused(self):
        assert segment_word("አልሆነ", self.lex()).known is False

    def test_al_on_a_noun_is_refused(self):
        assert segment_word("አልቤትም", self.lex()).known is False

    def test_al_on_an_unset_category_is_refused(self):
        assert segment_word("አልሆነም", typed(ሆነ=None)).known is False


class TestSuffixPositions:
    """
    Suffixes take a fixed order, at most one per position. On a noun:
    plural, then one possessive or definite, then ን, then ም, then ና.
    On a verb: subject or gerund, then object, then ም, then ና. The order
    is the one the IRIT annotation attests.
    """

    def test_plural_then_case_beats_two_possessives(self):
        # ስራዎችን was ስራ + ዎ + ችን, "your our".
        assert segment_word("ስራዎችን", typed(ስራ="noun")).morphemes == ["ስራ", "ዎች", "ን"]

    def test_plural_definite_case(self):
        assert segment_word("ቤቶቹን", typed(ቤት="noun")).morphemes == ["ቤት", "ኦች", "ኡ", "ን"]

    def test_definite_case_particle(self):
        assert segment_word("ቤቱንም", typed(ቤት="noun")).morphemes == ["ቤት", "ኡ", "ን", "ም"]

    def test_two_possessives_on_a_noun_are_refused(self):
        # ጥላሁን is a name, not ጥላ + ህ + ኡ + ን.
        assert segment_word("ጥላሁን", typed(ጥላ="noun")).known is False
        assert segment_word("ብሩህ", typed(ብር="noun")).known is False

    def test_plural_after_a_possessive_is_refused(self):
        assert segment_word("ቤቴዎች", typed(ቤት="noun")).known is False

    def test_adjective_ending_precedes_the_plural(self):
        assert segment_word("ውጤታማዎች", typed(ውጤት="noun")).morphemes == ["ውጤት", "ኣማ", "ዎች"]

    def test_verb_subject_then_object(self):
        # ሰጡኝ "they gave me": ኡ is the subject on a verb and precedes the
        # object, where on a noun it would follow the plural.
        assert segment_word("ሰጡኝ", typed(ሰጠ="verb")).morphemes == ["ሰጠ", "ኡ", "ኝ"]

    def test_object_before_subject_is_refused(self):
        assert segment_word("ሰጠኝኩ", typed(ሰጠ="verb")).known is False

    def test_m_then_na(self):
        assert segment_word("ቤቱምና", typed(ቤት="noun")).morphemes == ["ቤት", "ኡ", "ም", "ና"]

    def test_every_row_has_a_position(self):
        for shape in SUFFIX_TABLE:
            assert isinstance(shape.slot, int)


class TestFourSuffixes:
    """
    A noun can fill four positions after the stem: plural, possessive or
    definite, case, and a particle.
    """

    def test_four_suffixes_by_default(self):
        result = segment_word("ልጆቹንም", typed(ልጅ="noun"))
        assert result.morphemes == ["ልጅ", "ኦች", "ኡ", "ን", "ም"]

    def test_with_a_possessive(self):
        result = segment_word("ልጆቻቸውንና", typed(ልጅ="noun"))
        assert result.morphemes == ["ልጅ", "ኦች", "ኣቸው", "ን", "ና"]


# ── Verb aspects: imperfective, jussive, gerund ──────────────────────────────

from amharic_morph.morphology import (  # noqa: E402
    CORPUS_PREFIXES,
    PREFIX_TABLE,
    verb_stems,
)


def verbs():
    """A lexicon with reviewed stems, as the stem review produces."""
    def verb(word, **stems):
        return {"word": word, "category": "verb", "allows_prefix": [], "allows_suffix": [],
                "stems": stems}
    index = {
        "ሆነ": verb("ሆነ", imperfective="ሆን", jussive="ሁን", gerund="ሆን",
                   imperfective_palatal="ሆኝ", gerund_palatal="ሆኝ"),
        "ሄደ": verb("ሄደ", imperfective="ሄድ", jussive="ሂድ", gerund="ሄድ",
                   imperfective_palatal="ሄጅ", jussive_palatal="ሂጅ", gerund_palatal="ሄጅ"),
        "ፈለገ": verb("ፈለገ", imperfective="ፈልግ", jussive="ፈልግ", gerund="ፈልግ"),
        "ገባ": verb("ገባ", imperfective="ገብ", jussive="ግብ", gerund="ገብ"),
        "ሰጠ": verb("ሰጠ", gerund="ሰጥ"),
        "ኖረ": verb("ኖረ", imperfective="ኖር"),
        "መሰለ": verb("መሰለ", imperfective="መስል"),
        "ደረሰ": verb("ደረሰ", imperfective="ደርስ", jussive="ድረስ"),
        "አለ": verb("አለ", imperfective="ል", jussive="በል", gerund="ብል", gerund_palatal="ብይ"),
        "የለ": verb("የለ"),
    }
    index["ቤት"] = {"word": "ቤት", "category": "noun", "allows_prefix": [], "allows_suffix": []}
    return index


def seg(word):
    return segment_word(word, verbs())


class TestVerbStemIndex:
    def test_maps_each_stem_to_its_verb_and_aspect(self):
        index = verb_stems(verbs())
        assert ("ሆነ", "imperfective") in index["ሆን"]
        assert ("ሆነ", "gerund") in index["ሆን"]
        assert index["ሂድ"] == (("ሄደ", "jussive"),)

    def test_a_noun_has_no_stems(self):
        assert "ቤት" not in verb_stems(verbs())


class TestMainImperfective:
    """person + imperfective stem + ኣል (+ person ending), as IRIT: ይ_ሆን_አል"""

    def test_third_person(self):
        result = seg("ይሆናል")
        assert result.morphemes == ["ይ", "ሆነ", "ኣል"]
        assert result.aspect == "imperfective"

    def test_stem_is_reported_in_lexicon_form(self):
        assert seg("ይፈልጋል").morphemes == ["ይ", "ፈለገ", "ኣል"]
        assert seg("ይገባል").morphemes == ["ይ", "ገባ", "ኣል"]

    def test_plural(self):
        assert seg("ይሆናሉ").morphemes == ["ይ", "ሆነ", "ኣል", "ኡ"]

    def test_first_person(self):
        assert seg("እሆናለሁ").morphemes == ["እ", "ሆነ", "ኣል", "ሁ"]

    def test_third_feminine(self):
        assert seg("ትሆናለች").morphemes == ["ት", "ሆነ", "ኣል", "ች"]

    def test_second_plural(self):
        assert seg("ትሆናላችሁ").morphemes == ["ት", "ሆነ", "ኣል", "ኣችሁ"]

    def test_bound_auxiliary_needs_a_person_ending(self):
        # ...ናለ without an ending after it is not a word.
        assert seg("ይሆናለ").known is False


class TestBoundImperfective:
    def test_relative(self):
        result = seg("የሚሆን")
        assert result.morphemes == ["የ", "ሚ", "ሆነ"]
        assert result.aspect == "imperfective"

    def test_subordinators_carry_the_third_person(self):
        for word, sub in (("ሲሆን", "ሲ"), ("ሊሆን", "ሊ"), ("ቢሆን", "ቢ"), ("እንዲሆን", "እንዲ")):
            assert seg(word).morphemes == [sub, "ሆነ"]

    def test_subordinator_then_person(self):
        assert seg("ስትሄድ").morphemes == ["ስ", "ት", "ሄደ"]

    def test_subordinator_needing_a_person_alone_is_refused(self):
        assert seg("ስሄድ").known is False

    def test_preposition_before_the_relative(self):
        assert seg("እንደሚሆን").morphemes == ["እንደ", "ሚ", "ሆነ"]

    def test_plural_on_the_bound_stem(self):
        assert seg("የሚሆኑ").morphemes == ["የ", "ሚ", "ሆነ", "ኡ"]

    def test_relative_with_the_linking_vowel(self):
        # የሚኖረው: IRIT የ_ሚ_ኖር_ኧ_ው.
        assert seg("የሚኖረው").morphemes == ["የ", "ሚ", "ኖረ", "ው"]

    def test_imperfective_stem_alone_is_refused(self):
        assert seg("ሄድ").known is False


class TestStemVowel:
    """
    A verb whose perfective ends in ኣ (ገባ) keeps that ኣ as its own
    morpheme on the imperfective and jussive stem, as IRIT: የ_ሚ_ገብ_ኣ.
    Ruled 2026-10-29: the stem is ገብ, and ኣ fuses onto its last letter.
    """

    def test_relative(self):
        result = seg("የሚገባ")
        assert result.morphemes == ["የ", "ሚ", "ገባ", "ኣ"]
        assert result.aspect == "imperfective"

    def test_subordinator(self):
        assert seg("ሲገባ").morphemes == ["ሲ", "ገባ", "ኣ"]

    def test_jussive(self):
        result = seg("ይግባ")
        assert result.morphemes == ["ይ", "ገባ", "ኣ"]
        assert result.aspect == "jussive"

    def test_object_follows_the_stem_vowel(self):
        # IRIT ይ_ገብ_አ_ው_አል: the ው is the 3ms object, after the ኣ.
        assert seg("ይገባዋል").morphemes == ["ይ", "ገባ", "ኣ", "ው", "ኣል"]

    def test_negative(self):
        assert seg("አይገባም").morphemes == ["አይ", "ገባ", "ኣ", "ም"]

    def test_the_auxiliary_still_fuses_directly(self):
        assert seg("ይገባል").morphemes == ["ይ", "ገባ", "ኣል"]

    def test_bare_stem_is_refused(self):
        # ይነስ is "let it be less", not a form of ተነሳ.
        assert seg("ይገብ").known is False
        assert seg("ሲገብ").known is False

    def test_only_on_a_verb_whose_perfective_ends_in_a(self):
        assert seg("ይሄዳ").known is False

    def test_reading(self):
        assert suffix_readings(seg("የሚገባ"), "ኣ") == ("stem_vowel",)

    def test_gerund_a_keeps_its_reading(self):
        assert suffix_readings(seg("ሆና"), "ኣ") == ("gerund_3fs",)


class TestSay:
    """አለ "say": its written stems are ል, በል, ብል and ብይ."""

    def test_one_letter_imperfective_stem(self):
        assert seg("ይላል").morphemes == ["ይ", "አለ", "ኣል"]
        assert seg("ሲል").morphemes == ["ሲ", "አለ"]
        assert seg("የሚል").morphemes == ["የ", "ሚ", "አለ"]

    def test_gerund(self):
        assert seg("ብሎ").morphemes == ["አለ", "ኦ"]
        assert seg("ብዬ").morphemes == ["አለ", "ኤ"]

    def test_first_plural_after_a_subordinator_is_n(self):
        assert seg("ስንል").morphemes == ["ስ", "ን", "አለ"]


class TestFirstPersonPrefixIsWordInitial:
    """
    እ and እን are written only at the start of a verb. After a
    subordinator the 1s has no prefix and the 1pl is ን (ስንል). IRIT has
    no subordinator followed by እ.
    """

    def test_e_after_a_subordinator_is_refused(self):
        # ስእል is "picture", not ስ + እ + ል.
        assert seg("ስእል").known is False
        assert seg("ልእልና").known is False

    def test_e_alone_still_works(self):
        assert seg("እላለሁ").morphemes == ["እ", "አለ", "ኣል", "ሁ"]


class TestSubordinateClauseShape:
    def test_no_auxiliary_after_a_subordinator(self):
        # ቢላል is the name Bilal: ሲሆናል, ቢሆናል are not Amharic.
        assert seg("ቢላል").known is False
        assert seg("ሲሆናል").known is False

    def test_negative_relative_needs_a_prefix_before_it(self):
        # ማይል is "mile"; the negative relative is የማይል, እንደማይል.
        assert seg("ማይል").known is False
        assert seg("የማይል").morphemes == ["የ", "ማ", "ይ", "አለ"]


class TestNegativeExistential:
    """የለ "is not": its family is the ordinary perfective suffixes plus ም."""

    def test_family(self):
        assert seg("የለም").morphemes == ["የለ", "ም"]
        assert seg("የሉም").morphemes == ["የለ", "ኡ", "ም"]
        assert seg("የለውም").morphemes == ["የለ", "ው", "ም"]
        assert seg("የለኝም").morphemes == ["የለ", "ኝ", "ም"]

    def test_m_is_negative(self):
        assert suffix_readings(seg("የለም"), "ም") == ("negative",)

    def test_m_elsewhere_is_also(self):
        assert suffix_readings(seg("ሆነም"), "ም") == ("also",)


class TestNegativeImperfective:
    def test_ai_with_m(self):
        result = seg("አይሆንም")
        assert result.morphemes == ["አይ", "ሆነ", "ም"]
        assert result.aspect == "imperfective"

    def test_ai_without_m_is_refused(self):
        assert seg("አይሆን").known is False

    def test_negative_relative(self):
        assert seg("የማይሆን").morphemes == ["የ", "ማ", "ይ", "ሆነ"]

    def test_without(self):
        assert seg("ሳይሆን").morphemes == ["ሳይ", "ሆነ"]


class TestJussive:
    def test_jussive(self):
        result = seg("ይሁን")
        assert result.morphemes == ["ይ", "ሆነ"]
        assert result.aspect == "jussive"

    def test_bare_jussive_stem_is_not_taken_as_an_imperative(self):
        # A stem alone collides with ordinary words: ድረስ is "until" far
        # more often than "arrive!". The imperative is not guessed.
        assert seg("ሂድ").known is False
        assert seg("ድረስ").known is False

    def test_first_person_jussive(self):
        assert seg("ልሂድ").morphemes == ["ል", "ሄደ"]


class TestGerund:
    def test_o(self):
        result = seg("ሆኖ")
        assert result.morphemes == ["ሆነ", "ኦ"]
        assert result.aspect == "gerund"

    def test_o_on_a_three_radical_stem(self):
        assert seg("ፈልጎ").morphemes == ["ፈለገ", "ኦ"]

    def test_to(self):
        assert seg("ሰጥቶ").morphemes == ["ሰጠ", "ቶ"]
        assert seg("ገብቶ").morphemes == ["ገባ", "ቶ"]

    def test_third_feminine(self):
        assert seg("ሆና").morphemes == ["ሆነ", "ኣ"]

    def test_first_person_on_the_palatal_stem(self):
        result = seg("ሄጄ")
        assert result.morphemes == ["ሄደ", "ኤ"]
        assert result.aspect == "gerund_palatal"

    def test_gerund_needs_its_stem(self):
        # ኖረ has no reviewed gerund stem, so ኖሮ is not guessed.
        assert seg("ኖሮ").known is False

    def test_gerund_takes_no_prefix(self):
        assert seg("የሆኖ").known is False


class TestPalatal:
    def test_second_feminine(self):
        result = seg("ትሄጂ")
        assert result.morphemes == ["ት", "ሄደ", "ኢ"]
        assert result.aspect == "imperfective_palatal"

    def test_second_feminine_with_the_auxiliary(self):
        assert seg("ትሄጃለሽ").morphemes == ["ት", "ሄደ", "ኣል", "ሽ"]


class TestVerbsAndTheInfix:
    def test_no_infix_on_a_verb(self):
        # ...ላል looks like the plural-adjective copy (ትላልቅ), but on a verb
        # it is the auxiliary: ይ + መሰለ + ኣል.
        assert seg("ይመስላል").morphemes == ["ይ", "መሰለ", "ኣል"]


class TestDirectEntryRanksFirst:
    def test_entry_beats_a_derived_stem_of_equal_length(self):
        direct = Segmentation("w", "ባለ", suffixes=("ው",), category="other")
        derived = Segmentation("w", "ተባለ", suffixes=("ው",), category="verb", aspect="gerund")
        assert rank_analyses([derived, direct], {})[0] is direct


class TestPerfectiveIsUnchanged:
    def test_perfective_suffix(self):
        result = seg("ሄድኩ")
        assert result.morphemes == ["ሄደ", "ኩ"]
        assert result.aspect == "perfective"

    def test_relative_perfective(self):
        assert seg("የሆነ").morphemes == ["የ", "ሆነ"]

    def test_person_prefix_on_the_perfective_is_refused(self):
        assert seg("ይሆነ").known is False

    def test_nouns_have_no_aspect(self):
        result = seg("ቤቶች")
        assert result.morphemes == ["ቤት", "ኦች"]
        assert result.aspect is None


class TestPrefixTable:
    def test_corpus_prefixes_are_the_prepositions_and_al(self):
        # The corpus analysis has no categories, so it peels only what can
        # safely precede any word, plus the negative it always peeled.
        assert set(CORPUS_PREFIXES) == {"ስለ", "ወደ", "እንደ", "እስከ", "በየ", "የ", "በ", "ከ", "ለ", "አል"}

    def test_every_prefix_is_ethiopic(self):
        for row in PREFIX_TABLE:
            assert all(is_ethiopic(c) for c in row.morpheme)

    def test_prefix_readings(self):
        parts = seg("የሚሆን").parts
        assert [(p.text, p.readings) for p in parts][:2] == [("የ", None), ("ሚ", ("relative",))]
