# Changelog

## 0.1.0 (unreleased)

First release.

- Four linguistic levels for every word: morpheme, fidel, sound and root.
- Level 4: lexicon-anchored morpheme analysis over 915 reviewed entries.
  - Fused suffixes, the possessive and object series, the plural ኣት.
  - The በ-series ("on" a person, as in አለበት).
  - Perfective, imperfective, jussive and gerund verb forms for 44
    reviewed verbs.
  - The plural-adjective infix.
  - Readings for every affix.
- Level 1: native-speaker-reviewed roots.
- Subword level: a SentencePiece unigram model over lossless sound text,
  cut at morpheme boundaries; `encode()` and `decode()` round-trip exactly.
- Text normalization: spelling variants, punctuation, Latin and URL
  tagging.
- The `amharic-morph` command for trying the tokenizer on text.
