# Changelog

## 0.1.0 (unreleased)

First release.

- Four linguistic levels for every word: morpheme, fidel, sound and root.
- Level 4: lexicon-anchored morpheme analysis over 948 reviewed entries.
  - Fused suffixes, the possessive and object series, the plural ኣት.
  - The በ-series ("on" a person, as in አለበት).
  - Perfective, imperfective, jussive and gerund verb forms for 70
    reviewed verbs, and the present perfect (ሆኗል, ሆነዋል, አድርጓል).
  - Prefixes fused with a stem's initial አ (ያለው, ያውቃል, የሚያደርግ).
  - The plural-adjective infix.
  - The relative ም for subjects other than the 3rd person (የምፈልገው).
  - Readings for every affix.
- Level 1: native-speaker-reviewed roots.
- Subword level: a SentencePiece unigram model over lossless sound text,
  cut at morpheme boundaries; `encode()` and `decode()` round-trip exactly.
  The shipped 16K model was trained on 3.5M sentences from CC-100 and GPAC
  and keeps 100% of morpheme boundaries; a plain model on the same data
  keeps 21%. Needs the `bpe` extra.
- Text normalization: spelling variants, punctuation, Latin and URL
  tagging.
- The `amharic-morph` command for trying the tokenizer on text.
