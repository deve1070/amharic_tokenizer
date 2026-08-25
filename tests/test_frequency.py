"""
Tests for frequency analysis and lexicon candidate generation.

The suffix cases matter most: ኦች and ኡ fuse their vowel into the stem's
final consonant, so stripping them is fidel arithmetic rather than string
removal.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_tokenizer.frequency import (  # noqa: E402
    build_frequency_table,
    candidate_stems,
    coverage_curve,
    fidel_order,
    generate_candidates,
    is_ethiopic_word,
    strip_prefixes,
    strip_suffixes,
    to_order6,
    write_candidates,
)


# ── Fidel arithmetic ──────────────────────────────────────────────────────────


class TestFidelOrder:
    """Ethiopic families are 8-aligned, so order = codepoint % 8 + 1"""

    def test_order1(self):
        assert fidel_order("ተ") == 1

    def test_order2(self):
        assert fidel_order("ቱ") == 2

    def test_order5(self):
        assert fidel_order("ቴ") == 5

    def test_order6_is_bare_consonant(self):
        assert fidel_order("ት") == 6

    def test_order7(self):
        assert fidel_order("ቶ") == 7

    def test_labialized_is_order8(self):
        assert fidel_order("ሏ") == 8

    def test_non_ethiopic_is_zero(self):
        assert fidel_order("a") == 0
        assert fidel_order("") == 0


class TestToOrder6:
    """Inverting vowel fusion means moving a consonant back to order 6"""

    def test_order7_to_order6(self):
        assert to_order6("ቶ") == "ት"

    def test_order2_to_order6(self):
        assert to_order6("ቱ") == "ት"

    def test_order6_unchanged(self):
        assert to_order6("ት") == "ት"

    def test_various_families(self):
        assert to_order6("ዎ") == "ው"
        assert to_order6("ሮ") == "ር"
        assert to_order6("ኖ") == "ን"
        assert to_order6("ሙ") == "ም"

    def test_non_ethiopic_unchanged(self):
        assert to_order6("a") == "a"


# ── Token filtering ───────────────────────────────────────────────────────────


class TestIsEthiopicWord:
    """Only Ethiopic words feed the frequency table"""

    def test_word(self):
        assert is_ethiopic_word("ቤት") is True

    def test_punctuation_excluded(self):
        assert is_ethiopic_word("።") is False
        assert is_ethiopic_word("፣") is False

    def test_numeral_excluded(self):
        assert is_ethiopic_word("2010") is False

    def test_latin_excluded(self):
        assert is_ethiopic_word("Biden") is False

    def test_tagged_span_excluded(self):
        assert is_ethiopic_word("<LATIN>Biden</LATIN>") is False

    def test_empty_excluded(self):
        assert is_ethiopic_word("") is False


# ── Affix stripping ───────────────────────────────────────────────────────────


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


class TestCandidateStems:
    """Prefix and suffix stripping combined, one affix at a time"""

    def test_combines_both(self):
        stems = candidate_stems("የቤቶች")
        assert "ቤቶች" in stems

    def test_excludes_the_word_itself(self):
        assert "ቤት" not in candidate_stems("ቤት")

    def test_no_empty_stems(self):
        assert all(s for s in candidate_stems("የቤት"))


# ── Frequency table ───────────────────────────────────────────────────────────


class TestBuildFrequencyTable:
    """Streaming count over corpus.txt"""

    def test_counts_words(self, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("ቤት ሰው ቤት\nቤት ሰው\n", encoding="utf-8")
        counts = build_frequency_table(str(path))
        assert counts["ቤት"] == 3
        assert counts["ሰው"] == 2

    def test_excludes_punctuation(self, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("ቤት ሰው ።\n", encoding="utf-8")
        counts = build_frequency_table(str(path))
        assert "።" not in counts

    def test_excludes_tagged_spans(self, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("ቤት <LATIN>Biden</LATIN>\n", encoding="utf-8")
        counts = build_frequency_table(str(path))
        assert len(counts) == 1

    def test_empty_file(self, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("", encoding="utf-8")
        assert build_frequency_table(str(path)) == Counter()


# ── Coverage ──────────────────────────────────────────────────────────────────


class TestCoverageCurve:
    """Types needed for a share of tokens — open question 5"""

    def test_empty(self):
        assert coverage_curve(Counter())["50%"] == 0

    def test_single_type_covers_everything(self):
        assert coverage_curve(Counter({"ቤት": 10}))["99%"] == 1

    def test_uniform_distribution(self):
        # Four equally frequent types: half the tokens need two types.
        counts = Counter({"ሀ": 25, "ለ": 25, "ሐ": 25, "መ": 25})
        assert coverage_curve(counts)["50%"] == 2

    def test_skewed_distribution(self):
        counts = Counter({"ቤት": 90, "ሰው": 5, "ውሻ": 5})
        assert coverage_curve(counts)["50%"] == 1

    def test_all_targets_present(self):
        result = coverage_curve(Counter({"ቤት": 10, "ሰው": 5}))
        assert set(result) == {"50%", "75%", "90%", "95%", "99%"}


# ── Candidate generation ──────────────────────────────────────────────────────


class TestGenerateCandidates:
    """
    A stem must be attested standalone. That is the corpus test of the
    constraint that dependent morphemes never appear alone.
    """

    def make_counts(self):
        return Counter({
            "ቤት": 100,     # the stem, standalone
            "ቤቶች": 50,     # fused plural
            "ቤቱ": 40,      # fused definite
            "ቤትን": 30,     # accusative
            "የቤት": 20,     # prefixed
        })

    def test_finds_the_stem(self):
        candidates = generate_candidates(self.make_counts())
        assert candidates[0]["stem"] == "ቤት"

    def test_counts_distinct_forms(self):
        candidates = generate_candidates(self.make_counts())
        assert candidates[0]["distinct_forms"] == 4

    def test_records_standalone_frequency(self):
        candidates = generate_candidates(self.make_counts())
        assert candidates[0]["standalone_frequency"] == 100

    def test_total_frequency_sums_forms(self):
        candidates = generate_candidates(self.make_counts())
        assert candidates[0]["total_frequency"] == 100 + 50 + 40 + 30 + 20

    def test_unattested_stem_is_rejected(self):
        # ዘቢብ never appears alone, so ዘቢቦች yields no candidate.
        counts = Counter({"ዘቢቦች": 50})
        assert generate_candidates(counts) == []

    def test_stem_below_min_frequency_rejected(self):
        counts = Counter({"ቤት": 2, "ቤቶች": 50, "ቤቱ": 40})
        assert generate_candidates(counts, min_stem_frequency=10) == []

    def test_min_distinct_forms_enforced(self):
        # Two attested forms: ቤቶች and ቤቱ.
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 40})
        assert generate_candidates(counts, min_distinct_forms=2) != []
        assert generate_candidates(counts, min_distinct_forms=3) == []

    def test_single_form_rejected_by_default(self):
        counts = Counter({"ቤት": 100, "ቤቶች": 50})
        assert generate_candidates(counts) == []

    def test_rare_surface_forms_ignored(self):
        counts = Counter({"ቤት": 100, "ቤቶች": 1, "ቤቱ": 1})
        assert generate_candidates(counts, min_surface_frequency=5) == []

    def test_ranked_by_productivity(self):
        counts = Counter({
            "ቤት": 100, "ቤቶች": 50, "ቤቱ": 40, "ቤትን": 30,
            "ሰው": 500, "ሰዎች": 200,
        })
        candidates = generate_candidates(counts)
        # ቤት has more distinct forms, so it outranks the more frequent ሰው.
        assert candidates[0]["stem"] == "ቤት"

    def test_forms_sorted_by_frequency(self):
        candidates = generate_candidates(self.make_counts())
        assert candidates[0]["forms"][0] == "ቤቶች"

    def test_empty_counts(self):
        assert generate_candidates(Counter()) == []


# ── Output ────────────────────────────────────────────────────────────────────


class TestWriteCandidates:
    """Candidates go out as JSON for native speaker validation"""

    def test_writes_file(self, tmp_path):
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 40})
        candidates = generate_candidates(counts)
        path = tmp_path / "candidates.json"
        written = write_candidates(candidates, str(path))
        assert written == 1
        assert path.exists()

    def test_marks_unvalidated(self, tmp_path):
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 40})
        path = tmp_path / "candidates.json"
        write_candidates(generate_candidates(counts), str(path))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded[0]["validated"] is False

    def test_respects_limit(self, tmp_path):
        counts = Counter({
            "ቤት": 100, "ቤቶች": 50, "ቤቱ": 40,
            "ሰው": 100, "ሰዎች": 50, "ሰውን": 40,
        })
        path = tmp_path / "candidates.json"
        assert write_candidates(generate_candidates(counts), str(path), limit=1) == 1

    def test_amharic_not_escaped(self, tmp_path):
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 40})
        path = tmp_path / "candidates.json"
        write_candidates(generate_candidates(counts), str(path))
        assert "ቤት" in path.read_text(encoding="utf-8")


# ── Filtering bound forms and coincidences ───────────────────────────────────

from amharic_tokenizer.frequency import (  # noqa: E402
    RULE_AMBIGUOUS_FUSION,
    RULE_PLURAL_FUSION,
    RULE_PREFIX,
    candidate_stems_with_rules,
)


class TestDerivationRules:
    """Each stem is tagged with the rule that produced it"""

    def test_prefix_rule(self):
        assert candidate_stems_with_rules("የቤት")["ቤት"] == RULE_PREFIX

    def test_plural_fusion_rule(self):
        assert candidate_stems_with_rules("ቤቶች")["ቤት"] == RULE_PLURAL_FUSION

    def test_order4_is_marked_ambiguous(self):
        # ስር -> ስራ is the shape that produces false positives.
        assert candidate_stems_with_rules("ስራ")["ስር"] == RULE_AMBIGUOUS_FUSION


class TestStandaloneRatio:
    """
    A bound form can appear alone a few times through noise. The
    proportion, not the presence, separates it from a real word.
    """

    def test_bound_form_rejected(self):
        # ነ appears alone rarely but ነው/ነን/ነች are everywhere.
        counts = Counter({"ነ": 300, "ነው": 500_000, "ነን": 100_000, "ነች": 50_000})
        assert generate_candidates(counts) == []

    def test_real_word_kept(self):
        counts = Counter({"ቤት": 100_000, "ቤቶች": 30_000, "ቤቱ": 20_000})
        assert generate_candidates(counts)[0]["stem"] == "ቤት"

    def test_ratio_reported(self):
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 50})
        assert generate_candidates(counts)[0]["standalone_ratio"] == 0.5

    def test_threshold_is_configurable(self):
        # min_stem_length is relaxed too, so this isolates the ratio knob.
        counts = Counter({"ነ": 300, "ነው": 500_000, "ነን": 100_000})
        relaxed = generate_candidates(
            counts, min_standalone_ratio=0.0, min_stem_length=1
        )
        assert relaxed != []


class TestAmbiguousFusionGuard:
    """
    Order-4 fusion only counts when the derived form is much rarer than
    the stem, as a real definite form would be.
    """

    def test_frequent_lookalike_rejected(self):
        # ስራ (work) is not the definite of ስር (root).
        counts = Counter({"ስር": 23_621, "ስራ": 74_971, "ስሩ": 5_000, "የስር": 3_000})
        candidates = {c["stem"]: c for c in generate_candidates(counts)}
        assert "ስራ" not in candidates.get("ስር", {}).get("forms", [])

    def test_rare_definite_accepted(self):
        counts = Counter({"ቤት": 100_000, "ቤታ": 2_000, "ቤቶች": 30_000})
        candidates = {c["stem"]: c for c in generate_candidates(counts)}
        assert "ቤታ" in candidates["ቤት"]["forms"]

    def test_unambiguous_plural_not_guarded(self):
        # The plural rule is exempt: no word ends in order-7 + ች by chance.
        counts = Counter({"ልጅ": 1_000, "ልጆች": 5_000, "ልጁ": 800})
        candidates = {c["stem"]: c for c in generate_candidates(counts)}
        assert "ልጆች" in candidates["ልጅ"]["forms"]


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


# ── Lexicon entries (spec section 3.3) ───────────────────────────────────────

from amharic_tokenizer.frequency import (  # noqa: E402
    consonantal_root,
    identify_affix,
    to_lexicon_entry,
    write_lexicon_candidates,
)


class TestConsonantalRoot:
    """
    Level 1 over a lexicon stem.

    Level 1 is blocked for arbitrary surface forms because it applies to
    the independent morpheme only. A stem already is one, so the root is
    computable without Level 4.
    """

    def test_bet(self):
        assert consonantal_root("ቤት") == ["ብ", "ት"]

    def test_word_initial_glottal_kept(self):
        # Spec 2.4.2: word-initial አ is a root consonant, so it survives.
        # It is reduced to እ (order 6) like every other consonant —
        # see the note in consonantal_root about this spec deviation.
        assert consonantal_root("አለ") == ["እ", "ል"]

    def test_glottal_is_reduced_not_kept_verbatim(self):
        # The distinction that matters: the consonant is present, but at
        # order 6. Level 2 uses አ-family characters for vowels, so an
        # order-1 አ in a root would be ambiguous.
        assert "አ" not in consonantal_root("አለ")

    def test_word_internal_glottal_stripped(self):
        # Spec 2.4.2: word-internal እ is a template vowel.
        assert consonantal_root("ጣእም") == ["ጥ", "ም"]

    def test_vowels_reduced_to_order6(self):
        assert consonantal_root("ሰላም") == ["ስ", "ል", "ም"]

    def test_empty(self):
        assert consonantal_root("") == []


class TestIdentifyAffix:
    """Which affix derives a surface form from its stem"""

    def test_prefix(self):
        assert identify_affix("የቤት", "ቤት") == ("prefix", "የ")

    def test_multi_character_prefix(self):
        assert identify_affix("ስለቤት", "ቤት") == ("prefix", "ስለ")

    def test_plain_suffix(self):
        assert identify_affix("ቤትን", "ቤት") == ("suffix", "ን")

    def test_fused_plural_reported_as_morpheme(self):
        # The surface shows ቶች; the morpheme is ኦች.
        assert identify_affix("ቤቶች", "ቤት") == ("suffix", "ኦች")

    def test_fused_definite_reported_as_morpheme(self):
        assert identify_affix("ቤቱ", "ቤት") == ("suffix", "ኡ")

    def test_unrelated_form(self):
        assert identify_affix("ሰላም", "ቤት") is None


class TestToLexiconEntry:
    """Spec 3.3 schema, with corpus-derivable fields pre-filled"""

    def make(self):
        counts = Counter({
            "ቤት": 100, "ቤቶች": 50, "ቤቱ": 40, "ቤትን": 30, "የቤት": 20,
        })
        return to_lexicon_entry(generate_candidates(counts)[0])

    def test_has_all_spec_fields(self):
        entry = self.make()
        for field in (
            "word", "category", "gender", "allows_prefix", "allows_suffix",
            "allows_infix", "root_consonants", "source", "validated", "notes",
        ):
            assert field in entry

    def test_word_is_the_stem(self):
        assert self.make()["word"] == "ቤት"

    def test_prefixes_derived_from_attested_forms(self):
        assert self.make()["allows_prefix"] == ["የ"]

    def test_suffixes_derived_from_attested_forms(self):
        assert set(self.make()["allows_suffix"]) == {"ኦች", "ኡ", "ን"}

    def test_root_consonants_computed(self):
        assert self.make()["root_consonants"] == ["ብ", "ት"]

    def test_source_marked(self):
        assert self.make()["source"] == "corpus_bootstrap"

    def test_human_fields_left_null(self):
        entry = self.make()
        assert entry["category"] is None
        assert entry["gender"] is None
        assert entry["allows_infix"] is None
        assert entry["validated"] is False

    def test_evidence_attached(self):
        evidence = self.make()["evidence"]
        assert evidence["standalone_frequency"] == 100
        assert "ቤቶች" in evidence["attested_forms"]


class TestWriteLexiconCandidates:
    """Output file for the validation pass"""

    def test_writes_entries(self, tmp_path):
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 40})
        path = tmp_path / "candidates.json"
        assert write_lexicon_candidates(generate_candidates(counts), str(path)) == 1

    def test_valid_json_with_amharic(self, tmp_path):
        counts = Counter({"ቤት": 100, "ቤቶች": 50, "ቤቱ": 40})
        path = tmp_path / "candidates.json"
        write_lexicon_candidates(generate_candidates(counts), str(path))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded[0]["word"] == "ቤት"
        assert loaded[0]["root_consonants"] == ["ብ", "ት"]

    def test_respects_limit(self, tmp_path):
        counts = Counter({
            "ቤት": 100, "ቤቶች": 50, "ቤቱ": 40,
            "ሰው": 100, "ሰዎች": 50, "ሰውን": 40,
        })
        path = tmp_path / "candidates.json"
        assert write_lexicon_candidates(
            generate_candidates(counts), str(path), limit=1
        ) == 1


# ── Stems that are not independent words ─────────────────────────────────────

from amharic_tokenizer.frequency import (  # noqa: E402
    RULE_DEFINITE_FUSION,
    RULE_SUFFIX,
    derived_stems,
)


class TestMinStemLength:
    """
    Single fidels reach the candidate list through spacing noise. A
    prefix typed with a space after it is indistinguishable from a word
    standing alone, so the standalone ratio cannot reject them.
    """

    def test_single_fidel_rejected(self):
        counts = Counter({"የ": 54_653, "የው": 2_000, "የን": 1_500, "የም": 1_200})
        assert generate_candidates(counts) == []

    def test_two_fidel_word_kept(self):
        counts = Counter({"ቤት": 100_000, "ቤቶች": 30_000, "ቤቱ": 20_000})
        assert generate_candidates(counts)[0]["stem"] == "ቤት"

    def test_threshold_is_configurable(self):
        counts = Counter({"የ": 54_653, "የው": 2_000, "የን": 1_500})
        assert generate_candidates(counts, min_stem_length=1) != []


class TestDerivedStems:
    """
    A stem attested as an inflection of another stem is discarded, or
    Level 4 matches it whole and loses the affix. How much a derivation
    counts depends on the rule that produced it.
    """

    def test_unambiguous_fusion_needs_no_frequency_check(self):
        # Nothing but a plural produces order-7 + ች.
        attested = {"ልጅ": {"ልጆች": RULE_PLURAL_FUSION}, "ልጆች": {"ልጆቹ": RULE_SUFFIX}}
        counts = Counter({"ልጅ": 33_026, "ልጆች": 21_730})
        assert derived_stems(attested, counts) == {"ልጆች"}

    def test_unambiguous_fusion_applies_even_when_the_form_is_commoner(self):
        attested = {"ልጅ": {"ልጆች": RULE_PLURAL_FUSION}, "ልጆች": {"ልጆቹ": RULE_SUFFIX}}
        counts = Counter({"ልጅ": 1_000, "ልጆች": 50_000})
        assert derived_stems(attested, counts) == {"ልጆች"}

    def test_suffix_derivation_holds_when_the_base_is_commoner(self):
        attested = {"ስብሰባ": {"ስብሰባው": RULE_SUFFIX}, "ስብሰባው": {"ስብሰባውን": RULE_SUFFIX}}
        counts = Counter({"ስብሰባ": 12_787, "ስብሰባው": 1_567})
        assert derived_stems(attested, counts) == {"ስብሰባው"}

    def test_definite_fusion_ignored_when_the_claimer_is_rarer(self):
        # ልዩ ends in order 2 without being the definite of ልይ.
        attested = {"ልይ": {"ልዩ": RULE_DEFINITE_FUSION}, "ልዩ": {"ልዩው": RULE_SUFFIX}}
        counts = Counter({"ልይ": 72, "ልዩ": 26_881})
        assert derived_stems(attested, counts) == set()

    def test_suffix_derivation_ignored_when_the_claimer_is_rarer(self):
        # ሰላም is not ሰላ + ም; the affix rules just run backwards over it.
        attested = {"ሰላ": {"ሰላም": RULE_SUFFIX}, "ሰላም": {"ሰላሙ": RULE_SUFFIX}}
        counts = Counter({"ሰላ": 29, "ሰላም": 20_213})
        assert derived_stems(attested, counts) == set()

    def test_ambiguous_fusion_never_derives(self):
        # ቀና is a word, not the definite of ቀን, though it is rarer.
        attested = {"ቀን": {"ቀና": RULE_AMBIGUOUS_FUSION}, "ቀና": {"ቀናው": RULE_SUFFIX}}
        counts = Counter({"ቀን": 74_594, "ቀና": 2_910})
        assert derived_stems(attested, counts) == set()

    def test_base_survives(self):
        attested = {"ቤት": {"ቤቱ": RULE_DEFINITE_FUSION}, "ቤቱ": {"ቤቱን": RULE_SUFFIX}}
        counts = Counter({"ቤት": 118_411, "ቤቱ": 22_836})
        assert "ቤት" not in derived_stems(attested, counts)

    def test_forms_that_are_not_stems_are_ignored(self):
        # ቤቶች is a form of ቤት but never a stem in its own right.
        attested = {"ቤት": {"ቤቶች": RULE_PLURAL_FUSION}}
        assert derived_stems(attested, Counter({"ቤት": 100})) == set()

    def test_empty(self):
        assert derived_stems({}, Counter()) == set()


class TestDerivedStemsAreNotCandidates:
    """End to end through generate_candidates()"""

    def test_definite_form_excluded(self):
        counts = Counter({
            "ቤት": 100_000, "ቤቶች": 30_000, "ቤቱ": 20_000,
            "ቤቱን": 3_000, "ቤቱም": 2_500, "የቤቱ": 2_000,
        })
        stems = {c["stem"] for c in generate_candidates(counts)}
        assert "ቤት" in stems
        assert "ቤቱ" not in stems

    def test_base_keeps_the_derived_form_as_evidence(self):
        counts = Counter({
            "ቤት": 100_000, "ቤቶች": 30_000, "ቤቱ": 20_000,
            "ቤቱን": 3_000, "ቤቱም": 2_500,
        })
        entry = next(c for c in generate_candidates(counts) if c["stem"] == "ቤት")
        assert "ቤቱ" in entry["forms"]
