"""
Train the default subword model, the morpheme-aware unigram model.

Reproduces the shipped model from a corpus file built by corpus_runner
(one sentence per line, normalized). The held-out lines are chosen by a
hash of the line, so every run holds out the same ones.

    python scripts/train_bpe.py corpus/corpus.txt out/ --vocab 16000 --lines 0

--lines 0 trains on every training line. Each 300K lines needs about
1.1 GB of memory.
"""

import argparse
import hashlib
import time
from pathlib import Path

from amharic_morph import AmharicTokenizer
from amharic_morph.bpe import SubwordModel, train, training_line

HELD_OUT = 3  # lines whose md5 starts with a byte below this, about 1.2%


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--lines", type=int, default=0, help="training lines sampled; 0 for all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plain", action="store_true", help="train the plain baseline instead")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    kind = "plain" if args.plain else "morph"
    text = args.out / f"{kind}.txt"
    held_out = args.out / "heldout.txt"
    tok = AmharicTokenizer(cache_size=300_000)

    start = time.time()
    written = 0
    with open(args.corpus, encoding="utf-8") as corpus, open(text, "w", encoding="utf-8") as out, \
            open(held_out, "w", encoding="utf-8") as held:
        for line in corpus:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if hashlib.md5(line.encode("utf-8")).digest()[0] < HELD_OUT:
                held.write(line + "\n")
                continue
            out.write(training_line(line, tok, morphemes=not args.plain) + "\n")
            written += 1
    print(f"{written} training lines in {time.time() - start:.0f}s")

    prefix = args.out / f"{kind}_{args.vocab // 1000}k"
    start = time.time()
    train(text, prefix, args.vocab, input_sentence_size=args.lines, num_threads=args.threads)
    print(f"trained {prefix}.model in {time.time() - start:.0f}s")

    model = SubwordModel(f"{prefix}.model", tok, morphemes=not args.plain)
    lines = held_out.read_text(encoding="utf-8").splitlines()[:5000]
    bad = sum(1 for line in lines if model.decode(model.encode(line)) != line)
    words = sum(len(line.split()) for line in lines)
    pieces = sum(len(model.encode(line)) for line in lines)
    print(f"held-out: {pieces / words:.3f} pieces per word, {bad} of {len(lines)} lines fail the round trip")


if __name__ == "__main__":
    main()
