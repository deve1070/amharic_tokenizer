"""
Train the default subword model, the morpheme-aware unigram model.

Reproduces the shipped model from a corpus file built by corpus_runner
(one sentence per line, normalized). The held-out lines are chosen by a
hash of the line, so every run holds out the same ones.

    python scripts/train_bpe.py corpus/combined.txt out/ --vocab 16000

The corpus is reduced to a table of distinct segments and their counts,
which is what the model trains on, so every training line is used and a
few gigabytes of memory are enough.
"""

import argparse
import hashlib
import time
from pathlib import Path

from amharic_morph import AmharicTokenizer
from amharic_morph.bpe import SubwordModel, count_segments, train, write_segment_counts

HELD_OUT = 3  # lines whose md5 starts with a byte below this, about 1.2%


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plain", action="store_true", help="train the plain baseline instead")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    kind = "plain" if args.plain else "morph"
    counts_path = args.out / f"{kind}_segments.tsv"
    held_out = args.out / "heldout.txt"
    tok = AmharicTokenizer(cache_size=300_000)

    def training_lines(held):
        for line in corpus:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if hashlib.md5(line.encode("utf-8")).digest()[0] < HELD_OUT:
                held.write(line + "\n")
                continue
            yield line

    start = time.time()
    with open(args.corpus, encoding="utf-8") as corpus, open(held_out, "w", encoding="utf-8") as held:
        counts = count_segments(training_lines(held), tok, morphemes=not args.plain)
    write_segment_counts(counts, counts_path)
    print(f"{len(counts):,} distinct segments, {sum(counts.values()):,} in all, in {time.time() - start:.0f}s")

    prefix = args.out / f"{kind}_{args.vocab // 1000}k"
    start = time.time()
    train(counts_path, prefix, args.vocab, num_threads=args.threads)
    print(f"trained {prefix}.model in {time.time() - start:.0f}s")

    model = SubwordModel(f"{prefix}.model", tok, morphemes=not args.plain)
    lines = held_out.read_text(encoding="utf-8").splitlines()[:5000]
    bad = sum(1 for line in lines if model.decode(model.encode(line)) != line)
    words = sum(len(line.split()) for line in lines)
    pieces = sum(len(model.encode(line)) for line in lines)
    print(f"held-out: {pieces / words:.3f} pieces per word, {bad} of {len(lines)} lines fail the round trip")


if __name__ == "__main__":
    main()
