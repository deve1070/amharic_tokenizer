"""
Layer 7 — Quality Check

Corpus-level statistics, reported after the pipeline has run. This is the
last layer and the only one that looks at the corpus as a whole rather
than at one sentence at a time.

Everything here accumulates incrementally, so the full CC-100 run never
holds the corpus in memory. Two design points make that work:

  - Length distributions are kept as histograms keyed by integer length,
    not as lists of lengths. Percentiles come out exact, and memory is
    proportional to the number of distinct lengths (a few thousand)
    rather than to the number of sentences.

  - The token frequency table is the one genuinely large structure. A
    Counter over the Amharic types in CC-100 measures around 640 MB for
    ~6M types, which fits comfortably; it is capped at report time, not
    collection time, so the counts stay exact.

Drop accounting is the part that matters most in practice. The filters
in Layer 5 can discard the majority of a web-crawled corpus, and without
per-reason counts a run that eats 80% of the input reports only a smaller
number with no way to see which filter did it. add_dropped() records the
reason so the report decomposes.

Statistics reported, per spec section 4.3 Layer 7:
  - Total sentences and total Ethiopic word tokens
  - <LATIN> and <URL> tag counts
  - Sentence length distribution (min, mean, max, percentiles)
  - Source distribution
  - Ethiopic content ratio distribution
  - Top N most frequent tokens
  - Flags: high <LATIN> density sentences, encoding anomalies
"""

import json
import re
from collections import Counter

from amharic_morph.filter import (
    DROP_REASONS,
    _count_ethiopic_word_tokens,
    ethiopic_ratio,
    is_likely_tigrinya,
)

# Tags emitted by Layer 4.
_LATIN_TAG_PATTERN = re.compile(r"<LATIN>")
_URL_TAG_PATTERN = re.compile(r"<URL>")
_TAG_PATTERN = re.compile(r"</?(?:LATIN|URL)>")

_ETHIOPIC_RANGE = (0x1200, 0x137F)

# U+FFFD REPLACEMENT CHARACTER marks bytes that failed to decode, and C0
# control characters should not survive Layer 3 whitespace normalization.
# Either one in the output means something upstream is broken.
_ENCODING_ANOMALY_PATTERN = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")

# A sentence more than this fraction Latin-tagged is flagged for review.
#
# Measured over every sentence seen, not just the survivors. Among kept
# sentences this can never fire: a sentence that is 30% Latin-tagged
# cannot also be 90% Ethiopic by letter, so the Layer 5 threshold has
# already removed it. Counting only survivors made this flag structurally
# dead — it reported 0 across 555,408 sentences because it could not
# report anything else.
_HIGH_LATIN_DENSITY = 0.30

_DEFAULT_PERCENTILES = (5, 25, 50, 75, 95, 99)

# Ethiopic ratio histogram resolution. 20 buckets of 0.05 so that the
# band just below a 0.90 threshold (0.85-0.90) is visible on its own —
# that is the population that would be recovered by relaxing the cutoff.
#
# Like the Latin flag, this histogram is built over every sentence seen.
# Restricted to survivors it is a constant: the filter admits nothing
# below the threshold, so every kept sentence lands in the top buckets.
_RATIO_BUCKETS = 20

# Width of the "just missed" band reported below the threshold.
_NEAR_THRESHOLD_BAND = 0.05


def _is_ethiopic_char(ch: str) -> bool:
    return _ETHIOPIC_RANGE[0] <= ord(ch) <= _ETHIOPIC_RANGE[1]


def count_ethiopic_tokens(sentence: str) -> int:
    """
    Count Ethiopic word tokens in a sentence.

    Delegates to Layer 5's counter so that "Ethiopic word token" means
    exactly the same thing in the statistics as it does in the filter
    that enforces the minimum sentence length. Pure punctuation tokens
    do not count — which matters now that Layer 3 pads ። into a token
    of its own.

    Args:
        sentence: A classified sentence.

    Returns:
        Number of Ethiopic word tokens.

    Examples:
        >>> count_ethiopic_tokens("ሰላም ነው ።")
        2
        >>> count_ethiopic_tokens("<LATIN>Biden</LATIN> ተናገሩ")
        1
    """
    return _count_ethiopic_word_tokens(sentence)


def count_latin_tags(sentence: str) -> int:
    """Count <LATIN> tags in a sentence."""
    return len(_LATIN_TAG_PATTERN.findall(sentence))


def count_url_tags(sentence: str) -> int:
    """Count <URL> tags in a sentence."""
    return len(_URL_TAG_PATTERN.findall(sentence))


def has_encoding_anomaly(sentence: str) -> bool:
    """
    True if the sentence contains a replacement character or a control
    character that should not have survived Layer 3.

    Examples:
        >>> has_encoding_anomaly("ሰላም ነው")
        False
        >>> has_encoding_anomaly("ሰላም � ነው")
        True
    """
    return bool(_ENCODING_ANOMALY_PATTERN.search(sentence))


def latin_density(sentence: str) -> float:
    """
    Fraction of a sentence's tokens that are Latin-tagged.

    Args:
        sentence: A classified sentence.

    Returns:
        Ratio in [0.0, 1.0]. Returns 0.0 for an empty sentence.
    """
    total = len(sentence.split())
    if total == 0:
        return 0.0
    return count_latin_tags(sentence) / total


def sentence_metadata(sentence: str, source: str = "cc100", url: str | None = None,
                      date: str | None = None) -> dict:
    """
    Build one metadata.jsonl row for a sentence, per spec section 4.4.

    CC-100 carries no per-sentence url or date, so those default to None
    rather than being invented.

    Args:
        sentence: A kept sentence.
        source:   Corpus identifier.
        url:      Source URL if known.
        date:     Crawl or publication date if known.

    Returns:
        Dict with source, url, date, ethiopic_ratio, latin_tag_count,
        and sentence_length.
    """
    return {
        "source": source,
        "url": url,
        "date": date,
        "ethiopic_ratio": round(ethiopic_ratio(sentence), 4),
        "latin_tag_count": count_latin_tags(sentence),
        "sentence_length": len(sentence),
    }


def _percentiles_from_histogram(
    histogram: Counter, percentiles: tuple[int, ...]
) -> dict:
    """
    Exact percentiles from a histogram keyed by integer value.

    Uses the nearest-rank method: the p-th percentile is the smallest
    value whose cumulative count reaches ceil(p/100 * n).
    """
    total = sum(histogram.values())
    if total == 0:
        return {f"p{p}": None for p in percentiles}

    result = {}
    ordered = sorted(histogram.items())
    for p in percentiles:
        # Nearest-rank, 1-indexed.
        rank = -(-p * total // 100) or 1
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative >= rank:
                result[f"p{p}"] = value
                break
    return result


def _summarize_histogram(histogram: Counter, percentiles: tuple[int, ...]) -> dict:
    """min / mean / max / percentiles for a histogram of integer values."""
    total = sum(histogram.values())
    if total == 0:
        return {"count": 0, "min": None, "mean": None, "max": None,
                "percentiles": {f"p{p}": None for p in percentiles}}

    weighted = sum(value * count for value, count in histogram.items())
    return {
        "count": total,
        "min": min(histogram),
        "mean": round(weighted / total, 2),
        "max": max(histogram),
        "percentiles": _percentiles_from_histogram(histogram, percentiles),
    }


class CorpusStatistics:
    """
    Incremental accumulator for corpus-level statistics.

    Feed kept sentences through add_kept() and rejected ones through
    add_dropped(), then call to_dict() or write_json().

    Examples:
        >>> stats = CorpusStatistics()
        >>> stats.add_kept("ሰላም ነው ።")
        >>> stats.add_dropped("x", "too_short")
        >>> stats.to_dict()["totals"]["sentences_kept"]
        1
    """

    def __init__(
        self,
        top_n: int = 100,
        percentiles: tuple[int, ...] = _DEFAULT_PERCENTILES,
        high_latin_density: float = _HIGH_LATIN_DENSITY,
        ethiopic_threshold: float = 0.90,
    ):
        self.top_n = top_n
        self.percentiles = percentiles
        self.high_latin_density = high_latin_density
        # Needed only to report how many dropped sentences fell just
        # below the cutoff, which is what tells you whether the cutoff
        # is set in the right place.
        self.ethiopic_threshold = ethiopic_threshold

        self.sentences_kept = 0
        self.ethiopic_tokens = 0
        self.latin_tags = 0
        self.url_tags = 0

        self.char_length_histogram: Counter = Counter()
        self.token_length_histogram: Counter = Counter()
        # Over every sentence seen, kept or dropped.
        self.ratio_histogram: Counter = Counter()
        self.near_threshold_misses = 0
        self.ratio_sum = 0.0

        self.token_frequencies: Counter = Counter()
        self.source_counts: Counter = Counter()
        self.drop_counts: Counter = Counter()

        # Over every sentence seen, and over survivors alone. The second
        # is expected to be ~0; that it is 0 confirms the filter works,
        # which is the only thing it can usefully tell you.
        self.high_latin_density_count = 0
        self.high_latin_density_kept = 0
        self.encoding_anomaly_count = 0
        self.tigrinya_suspect_count = 0
        self.flagged_examples: list[dict] = []

    def add_kept(self, sentence: str, source: str = "cc100") -> None:
        """Record a sentence that survived the filters."""
        self.sentences_kept += 1
        self.source_counts[source] += 1

        ethiopic_tokens = count_ethiopic_tokens(sentence)
        self.ethiopic_tokens += ethiopic_tokens

        self.latin_tags += count_latin_tags(sentence)
        self.url_tags += count_url_tags(sentence)

        self.char_length_histogram[len(sentence)] += 1
        self.token_length_histogram[ethiopic_tokens] += 1

        self.ratio_sum += self._observe(sentence, kept=True)

        stripped = _TAG_PATTERN.sub(" ", sentence)
        self.token_frequencies.update(stripped.split())

        self._flag(sentence)

    def add_dropped(self, sentence: str, reason: str) -> None:
        """
        Record a sentence rejected by Layer 5.

        The sentence is still measured. Distributions built only from
        survivors describe the filter rather than the corpus — the
        Ethiopic ratio histogram in particular is a constant if the
        rejected population is excluded from it.

        Args:
            sentence: The rejected sentence.
            reason:   One of the DROP_* constants from filter.py.
        """
        self.drop_counts[reason] += 1
        self._observe(sentence, kept=False)

    def _observe(self, sentence: str, kept: bool) -> float:
        """
        Record the corpus-wide measurements that apply to every sentence,
        kept or dropped. Returns the sentence's Ethiopic ratio.
        """
        ratio = ethiopic_ratio(sentence)
        # Bucket index 0..N-1; a ratio of exactly 1.0 lands in the last.
        bucket = min(int(ratio * _RATIO_BUCKETS), _RATIO_BUCKETS - 1)
        self.ratio_histogram[bucket] += 1

        if not kept and (
            self.ethiopic_threshold - _NEAR_THRESHOLD_BAND
            <= ratio
            < self.ethiopic_threshold
        ):
            self.near_threshold_misses += 1

        if latin_density(sentence) > self.high_latin_density:
            self.high_latin_density_count += 1
            if kept:
                self.high_latin_density_kept += 1

        return ratio

    def _flag(self, sentence: str) -> None:
        """
        Record flags that only matter for surviving sentences, keeping a
        bounded set of examples. Latin density and the Ethiopic ratio are
        handled in _observe() because they apply corpus-wide.
        """
        flags = []
        if latin_density(sentence) > self.high_latin_density:
            flags.append("high_latin_density")
        if has_encoding_anomaly(sentence):
            self.encoding_anomaly_count += 1
            flags.append("encoding_anomaly")
        # Counted, not dropped. The Ethiopic threshold scores Tigrinya at
        # 1.0 and cannot see it, so this is the only place it surfaces
        # unless filtering is explicitly turned on.
        if is_likely_tigrinya(sentence):
            self.tigrinya_suspect_count += 1
            flags.append("tigrinya_suspect")

        if flags and len(self.flagged_examples) < 20:
            self.flagged_examples.append(
                {"flags": flags, "sentence": sentence[:200]}
            )

    @property
    def sentences_dropped(self) -> int:
        """Total sentences rejected across all filters."""
        return sum(self.drop_counts.values())

    def to_dict(self) -> dict:
        """
        Assemble the full statistics report.

        Returns:
            Nested dict covering totals, drop accounting, length and
            ratio distributions, source distribution, top tokens, and
            quality flags.
        """
        seen = self.sentences_kept + self.sentences_dropped
        total_tokens = sum(self.token_frequencies.values())

        return {
            "totals": {
                "sentences_seen": seen,
                "sentences_kept": self.sentences_kept,
                "sentences_dropped": self.sentences_dropped,
                "retention_rate": (
                    round(self.sentences_kept / seen, 4) if seen else None
                ),
                "ethiopic_word_tokens": self.ethiopic_tokens,
                "total_tokens": total_tokens,
                "unique_tokens": len(self.token_frequencies),
                "latin_tags": self.latin_tags,
                "url_tags": self.url_tags,
            },
            "drops": {
                # Every reason appears, including zeros, so the report
                # shape does not change between runs.
                reason: self.drop_counts.get(reason, 0)
                for reason in DROP_REASONS
            },
            "sentence_length_chars": _summarize_histogram(
                self.char_length_histogram, self.percentiles
            ),
            "sentence_length_ethiopic_tokens": _summarize_histogram(
                self.token_length_histogram, self.percentiles
            ),
            "ethiopic_ratio": {
                "kept_mean": (
                    round(self.ratio_sum / self.sentences_kept, 4)
                    if self.sentences_kept
                    else None
                ),
                # Over every sentence seen. Restricted to survivors this
                # is a constant, since the filter admits nothing below
                # the threshold.
                "histogram_all_sentences": {
                    f"{b / _RATIO_BUCKETS:.2f}-{(b + 1) / _RATIO_BUCKETS:.2f}":
                        self.ratio_histogram.get(b, 0)
                    for b in range(_RATIO_BUCKETS)
                },
                "threshold": self.ethiopic_threshold,
                # Dropped sentences within one bucket below the cutoff:
                # what relaxing the threshold would recover.
                "near_threshold_misses": self.near_threshold_misses,
            },
            "sources": {
                source: {
                    "sentences": count,
                    "percentage": (
                        round(count / self.sentences_kept * 100, 2)
                        if self.sentences_kept
                        else None
                    ),
                }
                for source, count in self.source_counts.most_common()
            },
            "top_tokens": [
                {"token": token, "count": count}
                for token, count in self.token_frequencies.most_common(self.top_n)
            ],
            "flags": {
                # Over every sentence seen. The _kept figure is expected
                # to be 0 — a sentence cannot be both 30% Latin-tagged
                # and 90% Ethiopic by letter — so a nonzero value there
                # means the threshold is not doing its job.
                "high_latin_density_sentences": self.high_latin_density_count,
                "high_latin_density_kept": self.high_latin_density_kept,
                "encoding_anomaly_sentences": self.encoding_anomaly_count,
                "tigrinya_suspect_sentences": self.tigrinya_suspect_count,
                "tigrinya_suspect_rate": (
                    round(self.tigrinya_suspect_count / self.sentences_kept, 4)
                    if self.sentences_kept
                    else None
                ),
                "examples": self.flagged_examples,
            },
        }

    def write_json(self, path: str, encoding: str = "utf-8") -> None:
        """
        Write the report to a JSON file.

        Args:
            path:     Destination path for statistics.json.
            encoding: File encoding.
        """
        with open(path, "w", encoding=encoding) as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def compute_statistics(
    sentences: list[str], top_n: int = 100, source: str = "cc100"
) -> dict:
    """
    Convenience wrapper for a list of already-filtered sentences.

    Drop counts are all zero here, because the sentences passed in have
    already survived filtering. Use CorpusStatistics directly during a
    real run so that rejected sentences are accounted for too.

    Args:
        sentences: Kept sentences.
        top_n:     How many frequent tokens to report.
        source:    Corpus identifier.

    Returns:
        The statistics dict.
    """
    stats = CorpusStatistics(top_n=top_n)
    for sentence in sentences:
        stats.add_kept(sentence, source=source)
    return stats.to_dict()
