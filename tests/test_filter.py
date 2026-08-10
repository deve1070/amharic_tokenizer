"""
Tests for Layer 5 — Filter

Covers:
  - ethiopic_ratio calculation (plain and with <LATIN>/<URL> tags)
  - Ethiopic content threshold pass/fail
  - Minimum and maximum sentence length
  - Boilerplate detection
  - Deduplication (exact and near-duplicate)
  - Combined filter pass

Run with: python -m pytest tests/test_filter.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amharic_tokenizer.filter import (
    deduplicate,
    ethiopic_ratio,
    filter_sentences,
    is_boilerplate,
    passes_all_filters,
    passes_ethiopic_threshold,
    passes_max_length,
    passes_min_length,
)

# ── ethiopic_ratio ────────────────────────────────────────────────────────────


class TestEthiopicRatio:
    """Ethiopic content ratio calculation"""

    def test_pure_ethiopic(self):
        assert ethiopic_ratio("ሰላም ነው") == 1.0

    def test_empty_string(self):
        assert ethiopic_ratio("") == 0.0

    def test_punctuation_only(self):
        # No countable content characters -> 0.0, not divide-by-zero error
        assert ethiopic_ratio("። ! ?") == 0.0

    def test_punctuation_excluded_from_ratio(self):
        # Punctuation should not affect the ratio either way
        assert ethiopic_ratio("ሰላም።") == 1.0
        assert ethiopic_ratio("ሰላም ነው።") == 1.0

    def test_half_ethiopic_half_latin_tag(self):
        # "ሰላም" (3 ethiopic chars) + "Hello" (5 latin chars inside tag)
        result = ethiopic_ratio("ሰላም <LATIN>Hello</LATIN>")
        expected = 3 / (3 + 5)
        assert abs(result - expected) < 0.001

    def test_tags_themselves_excluded(self):
        # The <LATIN> and </LATIN> markup must not count as content
        # characters at all -- only "Hi" (2 chars) counts as non-ethiopic
        result = ethiopic_ratio("<LATIN>Hi</LATIN>")
        assert result == 0.0

    def test_url_tag_counts_as_non_ethiopic(self):
        result = ethiopic_ratio("ሰላም <URL>http://x.com</URL>")
        assert result < 1.0
        assert result > 0.0

    def test_pure_latin_tagged(self):
        assert ethiopic_ratio("<LATIN>Hello World</LATIN>") == 0.0


# ── passes_ethiopic_threshold ──────────────────────────────────────────────────


class TestPassesEthiopicThreshold:
    """Threshold pass/fail at the default 0.90 cutoff"""

    def test_pure_ethiopic_passes(self):
        assert passes_ethiopic_threshold("ሰላም ነው እንዴት ነህ") is True

    def test_mostly_latin_fails(self):
        assert passes_ethiopic_threshold("<LATIN>Hello World Today</LATIN> ነው") is False

    def test_exactly_at_threshold_passes(self):
        # 9 ethiopic chars + 1 latin char = ratio exactly 0.90, should pass (>=)
        assert passes_ethiopic_threshold("ሰላምሰላምሰላም <LATIN>X</LATIN>") is True

    def test_custom_threshold(self):
        # ratio is 0.375 -- passes at 0.30, fails at 0.90
        sentence = "ሰላም <LATIN>Hello</LATIN>"
        assert passes_ethiopic_threshold(sentence, threshold=0.30) is True
        assert passes_ethiopic_threshold(sentence, threshold=0.90) is False


# ── Minimum length ────────────────────────────────────────────────────────────


class TestMinLength:
    """Minimum 5 Ethiopic word tokens by default"""

    def test_five_tokens_passes(self):
        assert passes_min_length("አንድ ሁለት ሶስት አራት አምስት") is True

    def test_four_tokens_fails(self):
        assert passes_min_length("አንድ ሁለት ሶስት አራት") is False

    def test_latin_tokens_not_counted(self):
        # 3 ethiopic + 2 latin tokens -- only ethiopic counted, fails min 5
        sentence = "አንድ ሁለት ሶስት <LATIN>four</LATIN> <LATIN>five</LATIN>"
        assert passes_min_length(sentence) is False

    def test_punctuation_token_not_counted(self):
        sentence = "አንድ ሁለት ሶስት አራት ።"
        assert passes_min_length(sentence) is False

    def test_custom_min_tokens(self):
        assert passes_min_length("አንድ ሁለት", min_tokens=2) is True
        assert passes_min_length("አንድ ሁለት", min_tokens=3) is False


# ── Maximum length ────────────────────────────────────────────────────────────


class TestMaxLength:
    """Maximum 512 characters by default"""

    def test_short_sentence_passes(self):
        assert passes_max_length("ሰላም ነው።") is True

    def test_exactly_at_max_passes(self):
        sentence = "ሰ" * 512
        assert passes_max_length(sentence) is True

    def test_over_max_fails(self):
        sentence = "ሰ" * 513
        assert passes_max_length(sentence) is False

    def test_custom_max(self):
        assert passes_max_length("ሰላም", max_chars=2) is False
        assert passes_max_length("ሰላም", max_chars=10) is True


# ── Boilerplate ───────────────────────────────────────────────────────────────


class TestBoilerplate:
    """Known boilerplate patterns are flagged"""

    def test_copyright_notice(self):
        assert is_boilerplate("<LATIN>Copyright</LATIN> 2024 ሁሉም መብት የተጠበቀ ነው") is True

    def test_all_rights_reserved(self):
        assert is_boilerplate("<LATIN>All Rights Reserved</LATIN>") is True

    def test_click_here(self):
        assert is_boilerplate("ለበለጠ መረጃ <LATIN>click here</LATIN>") is True

    def test_subscribe_prompt(self):
        assert is_boilerplate("<LATIN>Subscribe</LATIN> ለዜናችን") is True

    def test_normal_sentence_not_flagged(self):
        assert is_boilerplate("ሰላም ነው እንዴት ነህ ብርሃኑ") is False

    def test_case_insensitive(self):
        assert is_boilerplate("<LATIN>COPYRIGHT</LATIN> 2024") is True
        assert is_boilerplate("<LATIN>copyright</LATIN> 2024") is True


# ── Deduplication ─────────────────────────────────────────────────────────────


class TestDeduplicate:
    """Exact and near-duplicate removal"""

    def test_no_duplicates(self):
        sentences = ["ሰላም ነው።", "እንዴት ነህ።"]
        assert deduplicate(sentences) == sentences

    def test_exact_duplicate_removed(self):
        result = deduplicate(["ሰላም ነው።", "ሰላም ነው።", "እንዴት ነህ።"])
        assert result == ["ሰላም ነው።", "እንዴት ነህ።"]

    def test_near_duplicate_whitespace_removed(self):
        # Differs only in spacing -- treated as duplicate
        result = deduplicate(["ሰላም ነው።", "ሰላም  ነው።"])
        assert result == ["ሰላም ነው።"]

    def test_near_duplicate_punctuation_removed(self):
        # Differs only in terminal punctuation -- treated as duplicate
        result = deduplicate(["ሰላም ነው።", "ሰላም ነው!"])
        assert result == ["ሰላም ነው።"]

    def test_first_occurrence_kept(self):
        result = deduplicate(["ሀ ለ ሐ", "ሀ ለ ሐ", "ሀ ለ ሐ"])
        assert result == ["ሀ ለ ሐ"]
        assert len(result) == 1

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_genuinely_different_sentences_kept(self):
        sentences = ["ሰላም ነው።", "ደህና ነህ።", "እመሰግናለሁ።"]
        assert deduplicate(sentences) == sentences


# ── passes_all_filters ──────────────────────────────────────────────────────────


class TestPassesAllFilters:
    """Combined single-sentence filter check"""

    def test_good_sentence_passes(self):
        sentence = "ይህ ጥሩ የአማርኛ ዓረፍተ ነገር ነው።"
        assert passes_all_filters(sentence) is True

    def test_too_short_fails(self):
        assert passes_all_filters("ሰላም ነው") is False

    def test_too_much_latin_fails(self):
        sentence = "<LATIN>This is mostly English text here today now</LATIN> ነው"
        assert passes_all_filters(sentence) is False

    def test_boilerplate_fails(self):
        sentence = "ሁሉም ነገር <LATIN>Copyright</LATIN> 2024 ሁሉም መብት የተጠበቀ ነው"
        assert passes_all_filters(sentence) is False

    def test_empty_fails(self):
        assert passes_all_filters("") is False

    def test_whitespace_only_fails(self):
        assert passes_all_filters("    ") is False

    def test_too_long_fails(self):
        sentence = "ሰላም " * 200  # well over 512 chars
        assert passes_all_filters(sentence) is False


# ── filter_sentences (full pipeline) ─────────────────────────────────────────────


class TestFilterSentences:
    """End-to-end filtering: per-sentence filters then dedup"""

    def test_mixed_quality_list(self):
        sentences = [
            "ይህ ጥሩ የአማርኛ ዓረፍተ ነገር ነው።",  # good, keep
            "ሰላም",  # too short, drop
            "ይህ ጥሩ የአማርኛ ዓረፍተ ነገር ነው።",  # exact dup of first, drop
            "<LATIN>Copyright</LATIN> reserved",  # boilerplate, drop
        ]
        result = filter_sentences(sentences)
        assert result == ["ይህ ጥሩ የአማርኛ ዓረፍተ ነገር ነው።"]

    def test_empty_list(self):
        assert filter_sentences([]) == []

    def test_all_pass(self):
        sentences = [
            "ይህ የመጀመሪያው ዓረፍተ ነገር ምሳሌ ነው።",
            "ይህኛው ሁለተኛው የተለየ ዓረፍተ ነገር ነው።",
        ]
        result = filter_sentences(sentences)
        assert len(result) == 2


# ── Script-neutral characters in the ratio ───────────────────────────────────

from amharic_tokenizer.filter import (  # noqa: E402
    DROP_TIGRINYA,
    drop_reason,
    is_likely_tigrinya,
    tigrinya_markers,
)


class TestRatioIgnoresScriptNeutralCharacters:
    """
    The threshold asks "is this Amharic or another script?", so only
    letters can answer it. Digits and punctuation are script-neutral.

    Amharic uses Arabic numerals by settled decision, so counting them
    against the ratio discarded ordinary Amharic sentences.
    """

    def test_digits_do_not_lower_the_ratio(self):
        assert ethiopic_ratio("የ 161 ነዋሪዎች") == 1.0

    def test_ascii_punctuation_does_not_lower_the_ratio(self):
        assert ethiopic_ratio("አዲስ አበባ - ቦሌ , ለቤት ኪራይ") == 1.0

    def test_parentheses_do_not_lower_the_ratio(self):
        assert ethiopic_ratio("በደል ይረሳል ወይስ ይለመዳል ( 3 )") == 1.0

    def test_real_sentence_now_passes(self):
        # Dropped before the denominator fix, despite being pure Amharic.
        sentence = "የ 161 ነዋሪዎች እዚህ ቋሚ በሆነ ሁኔታ ይኖራሉ ።"
        assert passes_ethiopic_threshold(sentence) is True

    def test_latin_letters_still_lower_the_ratio(self):
        # 3 Ethiopic letters out of 8 total
        assert ethiopic_ratio("ሰላም <LATIN>Hello</LATIN>") == 0.375

    def test_latin_heavy_sentence_still_fails(self):
        sentence = "<LATIN>Posted</LATIN> <LATIN>in</LATIN> <LATIN>Ethiopia</LATIN>"
        assert passes_ethiopic_threshold(sentence) is False

    def test_no_letters_at_all_is_zero(self):
        assert ethiopic_ratio("12 ፥ 31-32 ) ።") == 0.0

    def test_bare_punctuation_is_zero(self):
        assert ethiopic_ratio("»") == 0.0


# ── Tigrinya detection ────────────────────────────────────────────────────────


class TestTigrinyaMarkers:
    """Distinctive Tigrinya function words"""

    def test_amharic_has_no_markers(self):
        assert tigrinya_markers("ሰላም ነው እንዴት ነህ") == 0

    def test_counts_distinct_markers(self):
        assert tigrinya_markers("እቲ ቤት ትምህርቲ ናይ ግምቢ እዩ") == 3

    def test_repeated_marker_counted_once(self):
        assert tigrinya_markers("እዩ እዩ እዩ") == 1


class TestIsLikelyTigrinya:
    """
    Heuristic, deliberately conservative. Amharic text mentioning Tigray
    place names contains ቐ without being Tigrinya.
    """

    def test_plain_amharic_is_not_flagged(self):
        assert is_likely_tigrinya("ኢትዮጵያ ትልቅ ሀገር ናት እና ህዝቧ ብዙ ነው") is False

    def test_two_function_words_flags(self):
        assert is_likely_tigrinya("እቲ ቤት ትምህርቲ ናይ ግምቢ እዩ") is True

    def test_one_function_word_plus_qe_flags(self):
        assert is_likely_tigrinya("ናይ ቓልሲ ታሪኽ") is True

    def test_qe_alone_does_not_flag(self):
        # Amharic sentence naming a Tigray city.
        sentence = "የመቐለ ከተማ የንጹህ መጠጥ ውሀ ችግርን ይፈታል የተባለ ግድብ ይገነባል ።"
        assert is_likely_tigrinya(sentence) is False

    def test_single_weak_marker_alone_does_not_flag(self):
        # ድማ is weak: one occurrence is not enough on its own.
        assert is_likely_tigrinya("ድማ ሰው መጣ") is False


class TestTigrinyaDropIsOptIn:
    """Measured by default; dropping requires explicit opt-in"""

    TIGRINYA = "እቲ ቤት ትምህርቲ ናይ ግምቢ ዘብአኛ ቤት ትምህርቲ እዩ ነይሩ"

    def test_not_dropped_by_default(self):
        assert drop_reason(self.TIGRINYA) is None

    def test_dropped_when_enabled(self):
        assert drop_reason(self.TIGRINYA, drop_tigrinya=True) == DROP_TIGRINYA

    def test_filter_sentences_keeps_by_default(self):
        assert filter_sentences([self.TIGRINYA]) == [self.TIGRINYA]

    def test_filter_sentences_drops_when_enabled(self):
        assert filter_sentences([self.TIGRINYA], drop_tigrinya=True) == []


class TestTigrinyaStrongVsWeakMarkers:
    """
    Some markers have an Amharic reading and cannot decide alone.

    አብ is Tigrinya "in" but also Amharic "father". Layer 0 normalizes
    the Tigrinya spelling ኣብ into አብ, so on normalized text there is no
    orthographic difference left to use.
    """

    def test_strong_marker_alone_flags(self):
        assert is_likely_tigrinya("ቅዱስ ማቴዎስ ሀደ ካብ ጸሀፍቲ ወንጌል እዩ") is True

    def test_nay_alone_flags(self):
        assert is_likely_tigrinya("ናይ ቤት ትምህርቲ መጽሀፍ") is True

    def test_ab_alone_does_not_flag(self):
        # Amharic scripture: እግዚአብሄር አብ = God the Father.
        sentence = "ሰውን ከእግዚአብሄር አብ ጋር አስታርቆ ለሰው ሰላምን ሰጠ"
        assert is_likely_tigrinya(sentence) is False

    def test_two_weak_markers_flag(self):
        assert is_likely_tigrinya("አብ ድማ ገዛ") is True

    def test_weak_marker_plus_qe_flags(self):
        assert is_likely_tigrinya("አብ ቓልሲ ታሪኽ") is True

    def test_plain_amharic_still_clean(self):
        assert is_likely_tigrinya("ኢትዮጵያ ትልቅ ሀገር ናት እና ህዝቧ ብዙ ነው") is False

    def test_mekelle_alone_still_not_flagged(self):
        sentence = "የመቐለ ከተማ የንጹህ መጠጥ ውሀ ችግርን ይፈታል የተባለ ግድብ ይገነባል ።"
        assert is_likely_tigrinya(sentence) is False
