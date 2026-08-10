"""
Corpus Processing Pipeline

Wires together Layers 0, 3, 4, 6, and 5 into a single end-to-end pipeline
that converts raw CC-100 lines into a clean, filtered, deduplicated
list of one-sentence-per-line output.

Processing order:

    raw line
        -> normalize             (Layer 0 - character canonicalization)
        -> normalize_punctuation  (Layer 3 - ፡፡ and Latin period conversion,
                                   punctuation padding, whitespace)
        -> classify_text           (Layer 4 - Latin/URL tagging)
        -> segment                  (Layer 6 - sentence splitting)
        -> filter_sentences          (Layer 5 - quality filter + dedup,
                                      applied once across the full corpus)

Layer 3 runs immediately after character normalization because every
later layer assumes punctuation is space-separated from its word. It is
also what turns sentence-final Latin periods and ፡፡ into ።, without
which the segmenter cannot see those sentence boundaries at all.

Classification runs BEFORE segmentation so that periods inside URLs
are wrapped in <URL> tags before the segmenter ever looks for
sentence-final punctuation. This avoids incorrectly splitting on
periods that are part of a URL.

Filtering runs LAST and operates on the full sentence list at once,
because deduplication requires seeing all sentences together.
"""

from amharic_tokenizer.filter import filter_sentences
from amharic_tokenizer.normalizer import normalize
from amharic_tokenizer.punctuation import normalize_punctuation
from amharic_tokenizer.script_classifier import classify_text
from amharic_tokenizer.segmenter import segment


def process_line(line: str) -> list[str]:
    """
    Run a single raw input line through
    normalize -> normalize_punctuation -> classify -> segment.

    Does NOT apply filtering -- filtering happens once across the full
    corpus in process_lines(), not per-line, because deduplication
    needs the complete sentence list.

    Args:
        line: A single raw line of text (e.g. one line from CC-100,
              which may contain multiple sentences).

    Returns:
        List of classified, segmented sentences extracted from this
        line. May be empty if the line contains no content.

    Examples:
        >>> process_line("ሐሙስ ነው፡፡ ደህና ነህ?")
        ['ሀሙስ ነው።', 'ደህና ነህ?']
    """
    normalized = normalize(line)
    punctuated = normalize_punctuation(normalized)
    classified = classify_text(punctuated)
    return segment(classified)


def process_lines(
    lines: list[str],
    ethiopic_threshold: float = 0.90,
    min_tokens: int = 5,
    max_chars: int = 512,
) -> list[str]:
    """
    Run a full list of raw input lines through the complete pipeline:
    normalize -> classify -> segment for every line, then filter the
    combined sentence list once.

    Args:
        lines: List of raw text lines (e.g. read from a corpus file).
        ethiopic_threshold: Minimum Ethiopic content ratio (Layer 5).
        min_tokens: Minimum Ethiopic word tokens per sentence (Layer 5).
        max_chars: Maximum sentence character length (Layer 5).

    Returns:
        Filtered, deduplicated list of clean sentences ready for
        corpus output.
    """
    all_sentences: list[str] = []
    for line in lines:
        all_sentences.extend(process_line(line))

    return filter_sentences(
        all_sentences,
        ethiopic_threshold=ethiopic_threshold,
        min_tokens=min_tokens,
        max_chars=max_chars,
    )


def process_file(
    input_path: str,
    output_path: str,
    ethiopic_threshold: float = 0.90,
    min_tokens: int = 5,
    max_chars: int = 512,
    encoding: str = "utf-8",
) -> dict:
    """
    Run the full pipeline on an input corpus file and write one clean
    sentence per line to the output file.

    Args:
        input_path: Path to raw input text file (UTF-8).
        output_path: Path to write clean, filtered sentences.
        ethiopic_threshold: Minimum Ethiopic content ratio.
        min_tokens: Minimum Ethiopic word tokens per sentence.
        max_chars: Maximum sentence character length.
        encoding: File encoding (default utf-8).

    Returns:
        A dict with basic run statistics:
            input_lines:      number of raw lines read
            raw_sentences:    number of sentences before filtering
            output_sentences: number of sentences written after
                               filtering and deduplication

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    with open(input_path, "r", encoding=encoding) as f:
        lines = f.readlines()

    raw_sentences: list[str] = []
    for line in lines:
        raw_sentences.extend(process_line(line))

    final_sentences = filter_sentences(
        raw_sentences,
        ethiopic_threshold=ethiopic_threshold,
        min_tokens=min_tokens,
        max_chars=max_chars,
    )

    with open(output_path, "w", encoding=encoding) as f:
        f.writelines(sentence + "\n" for sentence in final_sentences)

    return {
        "input_lines": len(lines),
        "raw_sentences": len(raw_sentences),
        "output_sentences": len(final_sentences),
    }
