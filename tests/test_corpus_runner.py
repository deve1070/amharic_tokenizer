"""
Tests for the streaming corpus runner.

The runner must produce the three spec section 4.4 outputs, keep
corpus.txt and metadata.jsonl aligned line for line, and stream rather
than materialize the input.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_tokenizer.corpus_runner import (  # noqa: E402
    _dedup_digest,
    run_corpus,
)

# Five Ethiopic word tokens each, so they clear the minimum length filter.
GOOD_A = "ኢትዮጵያ ትልቅ ሀገር ናት እና ህዝቧ ብዙ ነው።"
GOOD_B = "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"
GOOD_C = "ተማሪዎቹ በጠዋት ወደ ትምህርት ቤት ሄዱ።"


def write_corpus(tmp_path, lines, name="input.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run(tmp_path, lines, **kwargs):
    """Run the pipeline over lines, returning (report, paths)."""
    src = write_corpus(tmp_path, lines)
    out = tmp_path / "corpus.txt"
    meta = tmp_path / "metadata.jsonl"
    stats = tmp_path / "statistics.json"
    report = run_corpus(
        src,
        output_path=str(out),
        metadata_path=str(meta),
        statistics_path=str(stats),
        progress_stream=False,
        **kwargs,
    )
    return report, out, meta, stats


# ── Outputs ───────────────────────────────────────────────────────────────────


class TestOutputFiles:
    """The three spec section 4.4 outputs"""

    def test_writes_corpus_file(self, tmp_path):
        _, out, _, _ = run(tmp_path, [GOOD_A, GOOD_B])
        assert out.exists()
        assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_writes_metadata_file(self, tmp_path):
        _, _, meta, _ = run(tmp_path, [GOOD_A, GOOD_B])
        rows = meta.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 2
        assert json.loads(rows[0])["source"] == "cc100"

    def test_writes_statistics_file(self, tmp_path):
        _, _, _, stats = run(tmp_path, [GOOD_A, GOOD_B])
        loaded = json.loads(stats.read_text(encoding="utf-8"))
        assert loaded["totals"]["sentences_kept"] == 2

    def test_metadata_aligns_with_corpus(self, tmp_path):
        _, out, meta, _ = run(tmp_path, [GOOD_A, GOOD_B, GOOD_C])
        sentences = out.read_text(encoding="utf-8").strip().splitlines()
        rows = [
            json.loads(r)
            for r in meta.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert len(sentences) == len(rows)
        for sentence, row in zip(sentences, rows):
            assert row["sentence_length"] == len(sentence)

    def test_metadata_can_be_skipped(self, tmp_path):
        src = write_corpus(tmp_path, [GOOD_A])
        out = tmp_path / "corpus.txt"
        run_corpus(
            src,
            output_path=str(out),
            metadata_path=None,
            statistics_path=None,
            progress_stream=False,
        )
        assert not (tmp_path / "metadata.jsonl").exists()

    def test_amharic_not_escaped_in_corpus(self, tmp_path):
        _, out, _, _ = run(tmp_path, [GOOD_A])
        assert "ኢትዮጵያ" in out.read_text(encoding="utf-8")

    def test_metadata_carries_no_sentence_text(self, tmp_path):
        # Per spec 4.4 metadata.jsonl is parallel metadata; alignment
        # with corpus.txt is by line number, not by repeating the text.
        _, _, meta, _ = run(tmp_path, [GOOD_A])
        row = json.loads(meta.read_text(encoding="utf-8").strip())
        assert set(row) == {
            "source", "url", "date", "ethiopic_ratio",
            "latin_tag_count", "sentence_length",
        }


# ── Report ────────────────────────────────────────────────────────────────────


class TestReport:
    """The returned statistics dict, plus the run section"""

    def test_returns_statistics(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A, GOOD_B])
        assert report["totals"]["sentences_kept"] == 2

    def test_run_section_records_lines_read(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A, GOOD_B, GOOD_C])
        assert report["run"]["lines_read"] == 3

    def test_run_section_records_settings(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A], min_tokens=3)
        assert report["run"]["min_tokens"] == 3
        assert report["run"]["drop_tigrinya"] is False

    def test_drop_counts_present(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A, "አጭር።"])
        assert report["drops"]["too_short"] == 1


# ── Streaming behavior ────────────────────────────────────────────────────────


class TestStreaming:
    """Correctness properties that streaming must preserve"""

    def test_deduplicates_across_lines(self, tmp_path):
        report, out, _, _ = run(tmp_path, [GOOD_A, GOOD_A, GOOD_A])
        assert report["totals"]["sentences_kept"] == 1
        assert report["drops"]["duplicate"] == 2

    def test_keeps_first_occurrence(self, tmp_path):
        _, out, _, _ = run(tmp_path, [GOOD_A, GOOD_B, GOOD_A])
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("ኢትዮጵያ")
        assert len(lines) == 2

    def test_near_duplicates_collapse(self, tmp_path):
        # Differs only in whitespace, which Layer 5's dedup key ignores.
        report, _, _, _ = run(tmp_path, [GOOD_A, GOOD_A.replace(" ", "  ")])
        assert report["totals"]["sentences_kept"] == 1

    def test_preserves_input_order(self, tmp_path):
        _, out, _, _ = run(tmp_path, [GOOD_A, GOOD_B, GOOD_C])
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("ኢትዮጵያ")
        assert lines[1].startswith("አዲስ")
        assert lines[2].startswith("ተማሪዎቹ")

    def test_limit_stops_early(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A, GOOD_B, GOOD_C], limit=2)
        assert report["run"]["lines_read"] == 2

    def test_empty_input(self, tmp_path):
        report, out, _, _ = run(tmp_path, [])
        assert report["totals"]["sentences_kept"] == 0
        assert out.read_text(encoding="utf-8") == ""

    def test_blank_lines_ignored(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A, "", "   ", GOOD_B])
        assert report["totals"]["sentences_kept"] == 2

    def test_missing_input_raises(self, tmp_path):
        try:
            run_corpus(
                str(tmp_path / "nope.txt"),
                output_path=str(tmp_path / "o.txt"),
                metadata_path=None,
                statistics_path=None,
                progress_stream=False,
            )
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


# ── Dedup digest ──────────────────────────────────────────────────────────────


class TestDedupDigest:
    """Digests stand in for sentences to keep the dedup set bounded"""

    def test_identical_sentences_share_a_digest(self):
        assert _dedup_digest(GOOD_A) == _dedup_digest(GOOD_A)

    def test_different_sentences_differ(self):
        assert _dedup_digest(GOOD_A) != _dedup_digest(GOOD_B)

    def test_matches_layer5_near_duplicate_rule(self):
        # Whitespace and punctuation differences collapse, as in Layer 5.
        assert _dedup_digest("ሰላም ነው።") == _dedup_digest("ሰላም  ነው!")

    def test_digest_is_fixed_width(self):
        assert len(_dedup_digest(GOOD_A)) == 8
        assert len(_dedup_digest("ሀ")) == 8


# ── Tigrinya handling ─────────────────────────────────────────────────────────


TIGRINYA = "እቲ ቤት ትምህርቲ ናይ ግምቢ ዘብአኛ ቤት ትምህርቲ እዩ ነይሩ።"


class TestTigrinyaHandling:
    """Measured by default, dropped only on request"""

    def test_counted_but_kept_by_default(self, tmp_path):
        report, _, _, _ = run(tmp_path, [TIGRINYA])
        assert report["flags"]["tigrinya_suspect_sentences"] == 1
        assert report["totals"]["sentences_kept"] == 1
        assert report["drops"]["tigrinya"] == 0

    def test_dropped_when_requested(self, tmp_path):
        report, _, _, _ = run(tmp_path, [TIGRINYA], drop_tigrinya=True)
        assert report["drops"]["tigrinya"] == 1
        assert report["totals"]["sentences_kept"] == 0

    def test_amharic_unaffected(self, tmp_path):
        report, _, _, _ = run(tmp_path, [GOOD_A, GOOD_B], drop_tigrinya=True)
        assert report["totals"]["sentences_kept"] == 2
        assert report["drops"]["tigrinya"] == 0

    def test_rate_reported(self, tmp_path):
        report, _, _, _ = run(tmp_path, [TIGRINYA, GOOD_A])
        assert report["flags"]["tigrinya_suspect_rate"] == 0.5
