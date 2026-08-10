"""
Tests for Layer 7 — Quality Check.

Covers the statistics the spec requires after pipeline completion, plus
the per-filter drop accounting that makes a large run diagnosable.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_tokenizer.filter import DROP_REASONS  # noqa: E402
from amharic_tokenizer.quality_check import (  # noqa: E402
    CorpusStatistics,
    compute_statistics,
    count_ethiopic_tokens,
    count_latin_tags,
    count_url_tags,
    has_encoding_anomaly,
    latin_density,
    sentence_metadata,
)


# ── Token counting ────────────────────────────────────────────────────────────


class TestCountEthiopicTokens:
    """Matches Layer 5's definition of an Ethiopic word token"""

    def test_simple_sentence(self):
        assert count_ethiopic_tokens("ሰላም ነው") == 2

    def test_padded_full_stop_does_not_count(self):
        # Layer 3 pads ። into its own token; it is punctuation, not a word.
        assert count_ethiopic_tokens("ሰላም ነው ።") == 2

    def test_latin_tagged_token_does_not_count(self):
        assert count_ethiopic_tokens("<LATIN>Biden</LATIN> ተናገሩ") == 1

    def test_url_tagged_token_does_not_count(self):
        assert count_ethiopic_tokens("<URL>www.example.com</URL> ተናገሩ") == 1

    def test_numeral_does_not_count(self):
        assert count_ethiopic_tokens("2010 ተጀመረ") == 1

    def test_empty(self):
        assert count_ethiopic_tokens("") == 0


class TestTagCounting:
    """<LATIN> and <URL> tag counts"""

    def test_no_tags(self):
        assert count_latin_tags("ሰላም ነው") == 0
        assert count_url_tags("ሰላም ነው") == 0

    def test_single_latin_tag(self):
        assert count_latin_tags("<LATIN>Biden</LATIN> ተናገሩ") == 1

    def test_multiple_latin_tags(self):
        text = "<LATIN>A</LATIN> እና <LATIN>B</LATIN>"
        assert count_latin_tags(text) == 2

    def test_single_url_tag(self):
        assert count_url_tags("<URL>www.example.com</URL>") == 1

    def test_latin_and_url_counted_separately(self):
        text = "<LATIN>Biden</LATIN> <URL>www.a.com</URL>"
        assert count_latin_tags(text) == 1
        assert count_url_tags(text) == 1


# ── Quality flags ─────────────────────────────────────────────────────────────


class TestEncodingAnomalies:
    """Replacement and control characters indicate upstream breakage"""

    def test_clean_sentence(self):
        assert has_encoding_anomaly("ሰላም ነው") is False

    def test_replacement_character(self):
        assert has_encoding_anomaly("ሰላም � ነው") is True

    def test_control_character(self):
        assert has_encoding_anomaly("ሰላም \x00 ነው") is True

    def test_tab_is_not_an_anomaly(self):
        assert has_encoding_anomaly("ሰላም\tነው") is False

    def test_newline_is_not_an_anomaly(self):
        assert has_encoding_anomaly("ሰላም\nነው") is False


class TestLatinDensity:
    """Fraction of tokens that are Latin-tagged"""

    def test_no_latin(self):
        assert latin_density("ሰላም ነው") == 0.0

    def test_all_latin(self):
        assert latin_density("<LATIN>Hello</LATIN>") == 1.0

    def test_empty_sentence(self):
        assert latin_density("") == 0.0

    def test_partial(self):
        # 1 Latin tag out of 4 tokens
        assert latin_density("ሰላም <LATIN>Hi</LATIN> ነው ።") == 0.25


# ── Metadata rows ─────────────────────────────────────────────────────────────


class TestSentenceMetadata:
    """metadata.jsonl rows, per spec section 4.4"""

    def test_has_required_fields(self):
        meta = sentence_metadata("ሰላም ነው ።")
        assert set(meta) == {
            "source", "url", "date", "ethiopic_ratio",
            "latin_tag_count", "sentence_length",
        }

    def test_cc100_has_no_url_or_date(self):
        # CC-100 carries no per-sentence provenance; not invented.
        meta = sentence_metadata("ሰላም ነው ።")
        assert meta["url"] is None
        assert meta["date"] is None

    def test_source_recorded(self):
        assert sentence_metadata("ሰላም ነው", source="wiki")["source"] == "wiki"

    def test_length_and_tags(self):
        meta = sentence_metadata("ሰላም <LATIN>Hi</LATIN> ነው")
        assert meta["latin_tag_count"] == 1
        assert meta["sentence_length"] == len("ሰላም <LATIN>Hi</LATIN> ነው")

    def test_serializable(self):
        json.dumps(sentence_metadata("ሰላም ነው ።"))


# ── Accumulator totals ────────────────────────────────────────────────────────


class TestCorpusStatisticsTotals:
    """Running totals across kept and dropped sentences"""

    def test_empty_corpus(self):
        result = CorpusStatistics().to_dict()
        assert result["totals"]["sentences_seen"] == 0
        assert result["totals"]["retention_rate"] is None

    def test_counts_kept_sentences(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")
        stats.add_kept("ኢትዮጵያ ሀገር ናት ።")
        assert stats.to_dict()["totals"]["sentences_kept"] == 2

    def test_counts_dropped_sentences(self):
        stats = CorpusStatistics()
        stats.add_dropped("x", "too_short")
        stats.add_dropped("y", "too_long")
        assert stats.sentences_dropped == 2

    def test_retention_rate(self):
        # 1 kept out of 3 seen
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")
        stats.add_dropped("x", "too_short")
        stats.add_dropped("y", "too_short")
        assert stats.to_dict()["totals"]["retention_rate"] == 0.3333

    def test_retention_rate_simple(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")
        stats.add_dropped("x", "too_short")
        assert stats.to_dict()["totals"]["retention_rate"] == 0.5

    def test_unique_tokens(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ሰላም ነው")
        assert stats.to_dict()["totals"]["unique_tokens"] == 2

    def test_tag_totals(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም <LATIN>Hi</LATIN> ነው")
        stats.add_kept("<URL>www.a.com</URL> ነው")
        totals = stats.to_dict()["totals"]
        assert totals["latin_tags"] == 1
        assert totals["url_tags"] == 1


# ── Drop accounting ───────────────────────────────────────────────────────────


class TestDropAccounting:
    """
    The point of Layer 7: a run that discards most of the corpus must
    decompose into which filter did it.
    """

    def test_all_reasons_present_even_when_zero(self):
        drops = CorpusStatistics().to_dict()["drops"]
        assert set(drops) == set(DROP_REASONS)
        assert all(v == 0 for v in drops.values())

    def test_reasons_counted_separately(self):
        stats = CorpusStatistics()
        stats.add_dropped("a", "too_short")
        stats.add_dropped("b", "too_short")
        stats.add_dropped("c", "boilerplate")
        drops = stats.to_dict()["drops"]
        assert drops["too_short"] == 2
        assert drops["boilerplate"] == 1
        assert drops["too_long"] == 0

    def test_drops_sum_to_total(self):
        stats = CorpusStatistics()
        for reason in DROP_REASONS:
            stats.add_dropped("x", reason)
        assert sum(stats.to_dict()["drops"].values()) == len(DROP_REASONS)


# ── Distributions ─────────────────────────────────────────────────────────────


class TestLengthDistribution:
    """Histogram-backed percentiles, exact and O(1) in corpus size"""

    def test_empty_distribution(self):
        dist = CorpusStatistics().to_dict()["sentence_length_chars"]
        assert dist["count"] == 0
        assert dist["min"] is None
        assert dist["percentiles"]["p50"] is None

    def test_min_mean_max(self):
        stats = CorpusStatistics()
        stats.add_kept("ሀ")       # 1 char
        stats.add_kept("ሀሁ")      # 2 chars
        stats.add_kept("ሀሁሂ")     # 3 chars
        dist = stats.to_dict()["sentence_length_chars"]
        assert dist["min"] == 1
        assert dist["max"] == 3
        assert dist["mean"] == 2.0

    def test_median_of_known_set(self):
        stats = CorpusStatistics()
        for length in range(1, 101):
            stats.add_kept("ሀ" * length)
        dist = stats.to_dict()["sentence_length_chars"]
        assert dist["percentiles"]["p50"] == 50

    def test_percentiles_are_exact(self):
        stats = CorpusStatistics()
        for length in range(1, 101):
            stats.add_kept("ሀ" * length)
        pct = stats.to_dict()["sentence_length_chars"]["percentiles"]
        assert pct["p25"] == 25
        assert pct["p75"] == 75
        assert pct["p99"] == 99

    def test_token_length_distribution_tracked(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")          # 2 Ethiopic word tokens
        stats.add_kept("ሰላም ነው እንዴት ነህ ።")  # 4
        dist = stats.to_dict()["sentence_length_ethiopic_tokens"]
        assert dist["min"] == 2
        assert dist["max"] == 4


class TestEthiopicRatioDistribution:
    """
    The histogram covers every sentence seen, kept or dropped.

    Built from survivors alone it is a constant — the filter admits
    nothing below the threshold, so every kept sentence necessarily
    lands in the top buckets and the distribution describes the filter
    rather than the corpus.
    """

    def test_pure_ethiopic_lands_in_top_bucket(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው")
        hist = stats.to_dict()["ethiopic_ratio"]["histogram_all_sentences"]
        assert hist["0.95-1.00"] == 1

    def test_kept_mean_reported(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው")
        assert stats.to_dict()["ethiopic_ratio"]["kept_mean"] == 1.0

    def test_all_buckets_present(self):
        hist = CorpusStatistics().to_dict()["ethiopic_ratio"][
            "histogram_all_sentences"
        ]
        assert len(hist) == 20

    def test_kept_mean_is_none_when_empty(self):
        assert CorpusStatistics().to_dict()["ethiopic_ratio"]["kept_mean"] is None

    def test_dropped_sentences_appear_in_histogram(self):
        # The whole point: a low-ratio sentence is measured even though
        # it never reaches the corpus.
        stats = CorpusStatistics()
        stats.add_dropped("<LATIN>Hello</LATIN> <LATIN>World</LATIN>", "ethiopic_threshold")
        hist = stats.to_dict()["ethiopic_ratio"]["histogram_all_sentences"]
        assert hist["0.00-0.05"] == 1

    def test_histogram_spans_kept_and_dropped(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው")
        stats.add_dropped("<LATIN>Hello</LATIN>", "ethiopic_threshold")
        hist = stats.to_dict()["ethiopic_ratio"]["histogram_all_sentences"]
        assert sum(hist.values()) == 2

    def test_threshold_recorded(self):
        stats = CorpusStatistics(ethiopic_threshold=0.85)
        assert stats.to_dict()["ethiopic_ratio"]["threshold"] == 0.85


class TestNearThresholdMisses:
    """
    How many dropped sentences fell just below the cutoff — the number
    that says whether the cutoff is in the right place.
    """

    def test_zero_by_default(self):
        assert (
            CorpusStatistics().to_dict()["ethiopic_ratio"][
                "near_threshold_misses"
            ]
            == 0
        )

    def test_counts_sentence_just_below_threshold(self):
        # 9 Ethiopic letters of 10 total is 0.9; 17 of 19 is ~0.895,
        # inside the 0.85-0.90 near-miss band.
        stats = CorpusStatistics(ethiopic_threshold=0.90)
        sentence = "ሰላምነውእንዴትነህወዳጄሆይ <LATIN>ab</LATIN>"
        stats.add_dropped(sentence, "ethiopic_threshold")
        result = stats.to_dict()["ethiopic_ratio"]["near_threshold_misses"]
        assert result == 1

    def test_far_below_threshold_not_counted(self):
        stats = CorpusStatistics(ethiopic_threshold=0.90)
        stats.add_dropped("<LATIN>Hello</LATIN> <LATIN>World</LATIN>", "ethiopic_threshold")
        assert (
            stats.to_dict()["ethiopic_ratio"]["near_threshold_misses"] == 0
        )

    def test_kept_sentences_never_counted(self):
        stats = CorpusStatistics(ethiopic_threshold=0.90)
        stats.add_kept("ሰላም ነው")
        assert (
            stats.to_dict()["ethiopic_ratio"]["near_threshold_misses"] == 0
        )


class TestSourceDistribution:
    """Percentage from each source, per spec"""

    def test_single_source(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው", source="cc100")
        sources = stats.to_dict()["sources"]
        assert sources["cc100"]["sentences"] == 1
        assert sources["cc100"]["percentage"] == 100.0

    def test_two_sources(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው", source="cc100")
        stats.add_kept("ሰላም ነው", source="wiki")
        sources = stats.to_dict()["sources"]
        assert sources["cc100"]["percentage"] == 50.0
        assert sources["wiki"]["percentage"] == 50.0


# ── Top tokens ────────────────────────────────────────────────────────────────


class TestTopTokens:
    """Frequency table feeding the lexicon bootstrap"""

    def test_ordered_by_frequency(self):
        stats = CorpusStatistics()
        stats.add_kept("ሀ ሀ ሀ ለ ለ ሐ")
        top = stats.to_dict()["top_tokens"]
        assert top[0] == {"token": "ሀ", "count": 3}
        assert top[1] == {"token": "ለ", "count": 2}

    def test_respects_top_n(self):
        stats = CorpusStatistics(top_n=2)
        stats.add_kept("ሀ ሀ ሀ ለ ለ ሐ")
        assert len(stats.to_dict()["top_tokens"]) == 2

    def test_tags_excluded_from_frequencies(self):
        stats = CorpusStatistics()
        stats.add_kept("<LATIN>Hi</LATIN> ነው")
        tokens = {t["token"] for t in stats.to_dict()["top_tokens"]}
        assert "<LATIN>" not in tokens
        assert "Hi" in tokens


# ── Flags ─────────────────────────────────────────────────────────────────────


class TestFlags:
    """Sentences worth a second look"""

    def test_high_latin_density_flagged(self):
        stats = CorpusStatistics()
        stats.add_kept("<LATIN>A</LATIN> <LATIN>B</LATIN> ነው")
        assert stats.to_dict()["flags"]["high_latin_density_sentences"] == 1

    def test_normal_sentence_not_flagged(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው እንዴት ነህ ።")
        assert stats.to_dict()["flags"]["high_latin_density_sentences"] == 0

    def test_dropped_sentences_are_counted(self):
        # Counting survivors alone made this flag dead: a sentence that
        # is 30% Latin-tagged cannot pass a 90% Ethiopic threshold, so
        # the population it describes is entirely in the dropped set.
        stats = CorpusStatistics()
        stats.add_dropped("<LATIN>A</LATIN> <LATIN>B</LATIN>", "ethiopic_threshold")
        flags = stats.to_dict()["flags"]
        assert flags["high_latin_density_sentences"] == 1
        assert flags["high_latin_density_kept"] == 0

    def test_kept_count_tracked_separately(self):
        stats = CorpusStatistics()
        stats.add_kept("<LATIN>A</LATIN> <LATIN>B</LATIN> ነው")
        stats.add_dropped("<LATIN>C</LATIN> <LATIN>D</LATIN>", "ethiopic_threshold")
        flags = stats.to_dict()["flags"]
        assert flags["high_latin_density_sentences"] == 2
        assert flags["high_latin_density_kept"] == 1

    def test_encoding_anomaly_flagged(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም � ነው")
        assert stats.to_dict()["flags"]["encoding_anomaly_sentences"] == 1

    def test_examples_are_bounded(self):
        stats = CorpusStatistics()
        for _ in range(50):
            stats.add_kept("ሰላም � ነው")
        assert len(stats.to_dict()["flags"]["examples"]) <= 20


# ── Output ────────────────────────────────────────────────────────────────────


class TestOutput:
    """statistics.json must be writable and valid"""

    def test_report_is_json_serializable(self):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")
        json.dumps(stats.to_dict(), ensure_ascii=False)

    def test_write_json(self, tmp_path):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")
        path = tmp_path / "statistics.json"
        stats.write_json(str(path))

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["totals"]["sentences_kept"] == 1

    def test_amharic_not_escaped(self, tmp_path):
        stats = CorpusStatistics()
        stats.add_kept("ሰላም ነው ።")
        path = tmp_path / "statistics.json"
        stats.write_json(str(path))
        assert "ሰላም" in path.read_text(encoding="utf-8")


# ── Convenience wrapper ───────────────────────────────────────────────────────


class TestComputeStatistics:
    """compute_statistics over an already-filtered list"""

    def test_empty_list(self):
        assert compute_statistics([])["totals"]["sentences_kept"] == 0

    def test_counts_sentences(self):
        result = compute_statistics(["ሰላም ነው ።", "ኢትዮጵያ ሀገር ናት ።"])
        assert result["totals"]["sentences_kept"] == 2

    def test_no_drops_recorded(self):
        result = compute_statistics(["ሰላም ነው ።"])
        assert all(v == 0 for v in result["drops"].values())
