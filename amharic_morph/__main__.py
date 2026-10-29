"""
Try the tokenizer on text: `python -m amharic_morph "ቤቶቹን አየሁ።"`.

With no argument the text is read from standard input, so a file can be
piped in. Every word is shown at all four levels, with each suffix's
readings beside it.
"""

import sys

from amharic_morph.frequency import is_ethiopic_word
from amharic_morph.tokenizer import AmharicTokenizer


def _morphemes(analysis) -> str:
    parts = []
    for part in analysis.parts:
        text = part.text
        if part.readings:
            text += f" ({', '.join(part.readings)})"
        parts.append(text)
    return " + ".join(parts)


def describe(tok: AmharicTokenizer, word: str) -> list[str]:
    """The lines printed for one token."""
    if not is_ethiopic_word(word):
        return [word]
    analysis = tok.segment(word)
    root = tok.root(word)
    status = "" if analysis.known else "   [unknown: not segmented]"
    return [
        f"{word}{status}",
        f"    morphemes  {_morphemes(analysis)}",
        f"    fidel      {' '.join(tok.tokenize(word, level='fidel'))}",
        f"    sound      {' '.join(tok.tokenize(word, level='sound'))}",
        f"    root       {'-'.join(root) if root else '—'}",
    ]


def main(argv: list[str]) -> int:
    text = " ".join(argv) if argv else sys.stdin.read()
    tok = AmharicTokenizer()
    words = tok.words(text)
    known = 0
    for word in words:
        print("\n".join(describe(tok, word)))
        if is_ethiopic_word(word) and tok.segment(word).known:
            known += 1
    ethiopic = sum(1 for w in words if is_ethiopic_word(w))
    print(f"\n{known} of {ethiopic} words analysed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
