"""
Streaming corpus runner.

Drives the full pipeline over a corpus file and writes the three outputs
defined in spec section 4.4:

    corpus.txt       one clean sentence per line
    metadata.jsonl   one JSON object per line, aligned with corpus.txt
    statistics.json  corpus-level aggregates from Layer 7

Why this exists separately from pipeline.process_file(): that function
calls f.readlines(), which materializes the entire input before doing any
work. CC-100 Amharic is 3.1M lines / 890 MB, and the intermediate sentence
list is larger again, so it does not fit in memory. This module streams
line by line and never holds the corpus.

Memory is dominated by two structures, both bounded and both measured:

  - The dedup set. Sentences are reduced to 8-byte BLAKE2b digests of
    their normalized dedup key rather than kept as strings; at roughly
    70 bytes per entry in a Python set, 15M sentences costs about 1 GB.
  - The token frequency table in Layer 7, around 640 MB for 6M types.

Everything else is histograms and counters. A full CC-100 run fits in
well under 4 GB and takes roughly 17 minutes single-core at the measured
3,000 lines/s, so there is no chunking or multiprocessing here — it would
add failure modes without buying anything.

Deduplication is order-dependent (first occurrence wins), which streaming
preserves naturally: sentences are seen in file order. A second corpus is
deduplicated against the first by preloading its sentences
(dedup_against), so a sentence is kept once across both.

Input is read in bounded records, not lines. A corpus can arrive as
unbroken text (GPAC is one 1 GB line), and reading it as a line would
load it whole. read_records() cuts an over-long line after a sentence
end; ordinary lines pass through unchanged.
"""

import hashlib
import json
import sys
import time

from amharic_morph.filter import DROP_DUPLICATE, _dedup_key, drop_reason
from amharic_morph.pipeline import process_line
from amharic_morph.quality_check import CorpusStatistics, sentence_metadata

# Digest width for the dedup set. 8 bytes gives a collision probability
# around 1 in 10^5 at 20M sentences, which is far below the corpus's own
# near-duplicate rate and costs a quarter of what storing keys would.
_DIGEST_SIZE = 8


def _dedup_digest(sentence: str) -> bytes:
    """
    Reduce a sentence to a fixed-width digest for duplicate detection.

    Hashes Layer 5's dedup key, not the raw sentence, so that
    near-duplicates differing only in whitespace or punctuation collapse
    together exactly as deduplicate() would.
    """
    key = _dedup_key(sentence)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=_DIGEST_SIZE).digest()


# Where an over-long line may be cut: after a sentence end, the old-style
# ፡፡ first so that it is never split into ፡ ፡. A space is the fallback.
_SENTENCE_ENDS = ("፡፡", "።", "፧", "?", "!")
_MAX_RECORD_CHARS = 100_000
_CHUNK_CHARS = 1_000_000


def _cut_point(text: str, max_chars: int) -> int:
    """Where to cut text so the first part is at most max_chars long."""
    window = text[:max_chars]
    best = max((window.rfind(end) + len(end) for end in _SENTENCE_ENDS if end in window), default=0)
    if best:
        return best
    space = window.rfind(" ")
    return space + 1 if space > 0 else max_chars


def read_records(infile, max_chars: int = _MAX_RECORD_CHARS, chunk_chars: int = _CHUNK_CHARS):
    """
    Yield the file's text in records of at most max_chars characters.

    A line that fits is one record, with its newline. A longer line is cut
    after its last sentence end that fits, or else after a space. Joined,
    the records are exactly the file's text.

    Args:
        infile:      An open text file.
        max_chars:   The longest record.
        chunk_chars: How much is read at a time.
    """
    buffer = ""
    while True:
        piece = infile.readline(chunk_chars)
        buffer += piece
        while len(buffer) > max_chars:
            cut = _cut_point(buffer, max_chars)
            yield buffer[:cut]
            buffer = buffer[cut:]
        if not piece or buffer.endswith("\n"):
            if buffer:
                yield buffer
            buffer = ""
            if not piece:
                return


def run_corpus(
    input_path: str,
    output_path: str = "corpus.txt",
    metadata_path: str | None = "metadata.jsonl",
    statistics_path: str | None = "statistics.json",
    source: str = "cc100",
    ethiopic_threshold: float = 0.90,
    min_tokens: int = 5,
    max_chars: int = 512,
    drop_tigrinya: bool = False,
    dedup_against: list[str] | None = None,
    top_n: int = 100,
    limit: int | None = None,
    progress_every: int = 100_000,
    progress_stream=None,
    encoding: str = "utf-8",
) -> dict:
    """
    Stream a corpus file through the full pipeline and write outputs.

    Args:
        input_path:        Raw corpus file (e.g. corpus/am.txt).
        output_path:       Where to write one clean sentence per line.
        metadata_path:     Where to write metadata.jsonl, or None to skip.
        statistics_path:   Where to write statistics.json, or None to skip.
        source:            Corpus identifier recorded in metadata.
        ethiopic_threshold: Minimum Ethiopic letter ratio (Layer 5).
        min_tokens:        Minimum Ethiopic word tokens (Layer 5).
        max_chars:         Maximum sentence length (Layer 5).
        drop_tigrinya:     Reject Tigrinya-looking sentences. Off by
                           default; they are counted either way.
        dedup_against:     Corpus files written by an earlier run. Their
                           sentences count as already seen, so this run
                           keeps only what is new.
        top_n:             How many frequent tokens to report.
        limit:             Stop after this many input records. Useful for
                           validating settings before a full run.
        progress_every:    Emit a progress line every N input lines.
        progress_stream:   Where progress goes; defaults to stderr.
                           Pass False to silence.
        encoding:          File encoding.

    Returns:
        The Layer 7 statistics dict, with an added "run" section
        recording line counts and elapsed time.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if progress_stream is None:
        progress_stream = sys.stderr

    stats = CorpusStatistics(top_n=top_n, ethiopic_threshold=ethiopic_threshold)
    seen: set[bytes] = set()
    for path in dedup_against or ():
        with open(path, encoding=encoding) as existing:
            seen.update(_dedup_digest(line.rstrip("\n")) for line in existing)
    lines_read = 0
    started = time.perf_counter()

    metadata_file = None
    try:
        corpus_file = open(output_path, "w", encoding=encoding)
        if metadata_path:
            metadata_file = open(metadata_path, "w", encoding=encoding)

        with open(input_path, "r", encoding=encoding, errors="replace") as infile:
            for line in read_records(infile):
                if limit is not None and lines_read >= limit:
                    break
                lines_read += 1

                for sentence in process_line(line):
                    reason = drop_reason(
                        sentence,
                        ethiopic_threshold,
                        min_tokens,
                        max_chars,
                        drop_tigrinya,
                    )
                    if reason:
                        stats.add_dropped(sentence, reason)
                        continue

                    digest = _dedup_digest(sentence)
                    if digest in seen:
                        stats.add_dropped(sentence, DROP_DUPLICATE)
                        continue
                    seen.add(digest)

                    stats.add_kept(sentence, source=source)
                    corpus_file.write(sentence + "\n")
                    if metadata_file:
                        row = sentence_metadata(sentence, source=source)
                        metadata_file.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )

                if (
                    progress_stream
                    and progress_every
                    and lines_read % progress_every == 0
                ):
                    _report_progress(
                        progress_stream, lines_read, stats, started
                    )
    finally:
        corpus_file.close()
        if metadata_file:
            metadata_file.close()

    elapsed = time.perf_counter() - started
    report = stats.to_dict()
    report["run"] = {
        "input_path": input_path,
        "lines_read": lines_read,
        "elapsed_seconds": round(elapsed, 2),
        "lines_per_second": round(lines_read / elapsed, 1) if elapsed else None,
        "drop_tigrinya": drop_tigrinya,
        "dedup_against": list(dedup_against or ()),
        "ethiopic_threshold": ethiopic_threshold,
        "min_tokens": min_tokens,
        "max_chars": max_chars,
    }

    if statistics_path:
        with open(statistics_path, "w", encoding=encoding) as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if progress_stream:
        _report_progress(progress_stream, lines_read, stats, started, final=True)

    return report


def _report_progress(
    stream, lines_read: int, stats: CorpusStatistics, started: float, final=False
) -> None:
    """Write a one-line progress update."""
    elapsed = time.perf_counter() - started
    rate = lines_read / elapsed if elapsed else 0
    label = "done" if final else "..."
    stream.write(
        f"{label} {lines_read:,} lines | "
        f"{stats.sentences_kept:,} kept | "
        f"{stats.sentences_dropped:,} dropped | "
        f"{rate:,.0f} lines/s\n"
    )
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    """
    Command line entry point.

    Example:
        python -m amharic_morph.corpus_runner corpus/am.txt \\
            --output corpus/corpus.txt --limit 100000
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream a corpus through the AmharicTokenizer pipeline."
    )
    parser.add_argument("input", help="Raw corpus file")
    parser.add_argument("--output", default="corpus.txt")
    parser.add_argument("--metadata", default="metadata.jsonl")
    parser.add_argument("--statistics", default="statistics.json")
    parser.add_argument("--source", default="cc100")
    parser.add_argument("--ethiopic-threshold", type=float, default=0.90)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument(
        "--drop-tigrinya",
        action="store_true",
        help="Reject Tigrinya-looking sentences (counted either way)",
    )
    parser.add_argument(
        "--dedup-against",
        nargs="*",
        default=None,
        help="Corpus files from earlier runs; their sentences are not kept again",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after N input lines"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Emit a progress line every N input lines (0 to silence)",
    )
    args = parser.parse_args(argv)

    report = run_corpus(
        args.input,
        output_path=args.output,
        metadata_path=args.metadata,
        statistics_path=args.statistics,
        source=args.source,
        ethiopic_threshold=args.ethiopic_threshold,
        min_tokens=args.min_tokens,
        max_chars=args.max_chars,
        drop_tigrinya=args.drop_tigrinya,
        dedup_against=args.dedup_against,
        top_n=args.top_n,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    totals = report["totals"]
    print(
        f"kept {totals['sentences_kept']:,} of {totals['sentences_seen']:,} "
        f"sentences ({totals['retention_rate']:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
