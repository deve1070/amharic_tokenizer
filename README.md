# amharic-morph

Morphologically aware tokenization for Amharic. Every word is available at four linguistic levels (morpheme, fidel, sound and root), plus a subword level for language models that never cuts across a morpheme.

```python
>>> from amharic_morph import AmharicTokenizer
>>> tok = AmharicTokenizer()
>>> tok.tokenize("ቤቶች", level="morpheme")
['ቤት', 'ኦች']
>>> tok.tokenize("ቤቶች", level="fidel")
['ቤ', 'ቶ', 'ች']
>>> tok.tokenize("ቤቶች", level="sound")
['ብ', 'ኤ', 'ት', 'ኦ', 'ች']
>>> tok.tokenize("ቤቶች", level="root")
['ብ', 'ት']
```

> **Status: pre-release (0.1.0).** All five levels are complete and tested, and the subword model ships with the package.

## Why

Amharic builds words by attaching prefixes and suffixes to a stem, and some suffixes fuse into the stem's last letter. ቤቶች ("houses") is ቤት ("house") with the plural ኦች, whose vowel ኦ rides on the stem's ት and makes ቶ. The same stem then appears in the following words, and many more:

    ቤቱ  ቤቶቹ  ቤቶቻችንን  በቤታችን  የቤቱ  ለቤቶቹ

The usual ways of tokenizing Amharic do not see this structure:

- **Word-level tokenization** treats every inflected form as an unrelated vocabulary entry. In the 42.9M-word corpus used here, reaching 90% token coverage takes 93,699 distinct word forms; folding them to their stems takes 38,005.
- **Subword models (BPE or unigram) over raw or decomposed fidel** learn frequent letter sequences, wherever the morpheme boundaries fall. Trained on the same data as this package, such a model keeps only 15% to 40% of the morpheme boundaries, fewer as its vocabulary grows (see [Measurements](#measurements)). Its pieces often join a stem to part of a suffix: `▁አልነበር ኡም`, `▁ይሰራል ኡ`.
- **Many existing subword tokenizers do not decode back to the text they were given.** Punctuation and digits become `<unk>`, or a decomposed syllable is recomposed wrongly. On 10,000 corpus sentences, one published BPE tokenizer over decomposed fidel returned only 2.9% of them unchanged.

amharic-morph finds the morphemes first, from a lexicon of words reviewed by a native speaker, and builds every other level on that analysis.

## Install

```bash
pip install amharic-morph            # the four linguistic levels, no dependencies
pip install "amharic-morph[bpe]"     # adds the subword level (sentencepiece)
```

Python 3.10 to 3.13.

## The levels

| level | name | ቤቶች | ይሆናል "it will be" | what it is |
|---|---|---|---|---|
| 4 | `morpheme` | ቤት + ኦች | ይ + ሆነ + ኣል | the stem in its dictionary form, and its affixes |
| 3 | `fidel` | ቤ ቶ ች | ይ ሆ ና ል | the letters as written |
| 2 | `sound` | ብ ኤ ት ኦ ች | ይ ህ ኦ ን ኣ ል | each letter split into consonant and vowel |
| 1 | `root` | ብ-ት | ህ-ው-ን | the consonantal root of the stem, from the lexicon |
| | `bpe` | ▁ቤት ኦች | ▁ይ ሆን ኣል | subword pieces, cut at the morpheme boundaries |

Level 4 is computed first, because the other levels depend on it: the root belongs to the stem alone (ቤት), not to the whole word (ቤቶች would give ብ-ት-ች, with the plural's consonant inside the root).

## Usage

### A word

```python
>>> analysis = tok.segment("ቤቶቻችንን")
>>> analysis.morphemes
['ቤት', 'ኦች', 'ኣችን', 'ን']
>>> [(part.text, part.readings) for part in analysis.parts]
[('ቤት', None), ('ኦች', ('plural',)), ('ኣችን', ('possessive_1pl',)), ('ን', ('accusative',))]
```

Each affix carries its readings. A suffix can mean different things on a noun and on a verb: ኡ is the definite article in ቤቱ ("the house") and the plural subject in ሆኑ ("they became").

Verbs are analysed through their reviewed stems, and the aspect is reported:

```python
>>> tok.segment("ይሆናል").morphemes, tok.segment("ይሆናል").aspect
(['ይ', 'ሆነ', 'ኣል'], 'imperfective')
>>> [(p.text, p.readings) for p in tok.segment("አልነበረም").parts]
[('አል', ('negative',)), ('ነበረ', None), ('ም', ('negative',))]
```

All levels at once:

```python
>>> tok.analyze("ቤቶቹን")
{'word': 'ቤቶቹን', 'morpheme': ['ቤት', 'ኦች', 'ኡ', 'ን'], 'fidel': ['ቤ', 'ቶ', 'ቹ', 'ን'],
 'sound': ['ብ', 'ኤ', 'ት', 'ኦ', 'ች', 'ኡ', 'ን'], 'root': ['ብ', 'ት'], 'known': True, 'aspect': None}
```

### Text

`tokenize` accepts a sentence or a passage. Text is normalized first (spelling variants such as ሐ and ሀ are unified, punctuation is separated, Latin text is tagged), so the same word always gets the same analysis.

```python
>>> tok.tokenize("ልጆቹ ትምህርት ቤት ሄዱ።")
['ልጅ', 'ኦች', 'ኡ', 'ትምህርት', 'ቤት', 'ሄደ', 'ኡ', '።']
```

### Subwords for language models

```python
>>> tok = AmharicTokenizer()      # needs: pip install "amharic-morph[bpe]"
>>> tok.tokenize("ቤቶቻችንን የሚገባቸው ሰዎች", level="bpe")
['▁ቤት', 'ኦች', 'ኣችን', 'ን', '▁የ', 'ሚ', 'ገብ', 'ኣቸው', '▁ሰው', 'ኦች']
>>> ids = tok.encode("ቤቶቻችንን የሚገባቸው ሰዎች")
>>> tok.decode(ids)
'ቤቶቻችንን የሚገባቸው ሰዎች'
```

The shipped model has a 16,000-piece vocabulary and was trained on 3.5M sentences sampled from CC-100 and a second Amharic news corpus. It runs over sound text, where a fused suffix separates cleanly from its stem (ቤቶች is ብኤት | ኦች), and every word is cut at its morpheme boundaries before the model sees it. As a result:

- no piece crosses a morpheme boundary;
- `decode(encode(text))` returns the normalized text exactly, every character included;
- pieces display as readable fidel.

### Command line

```
$ amharic-morph "ልጆቹ ትምህርት ቤት ሄዱ።"
ልጆቹ
    morphemes  ልጅ + ኦች (plural) + ኡ (definite, possessive_3ms)
    fidel      ል ጆ ቹ
    sound      ል ጅ ኦ ች ኡ
    root       ል-ጅ
...
```

## How it works

**The lexicon.** Every Amharic word has exactly one independent morpheme, and affixes never appear alone, so a lexicon of independent words is enough to anchor the analysis. The lexicon has 1,146 entries drawn from the most frequent stems and function words in the corpus. A native speaker reviewed each entry's category and root, and the stems of its 93 verbs.

**Level 4 is constructive.** Affixes are removed to build candidate stems, and the lexicon decides which candidates are real. Fused suffixes are undone with fidel arithmetic: ቶ is restored to ት, and the candidate ቤት is looked up. The stem's category gates which affixes may attach, and a fixed affix order rules out impossible stacks. When several analyses survive, the one with the fewest morphemes wins. A word with no lexicon anchor is returned whole and marked unknown; it is never split on a guess.

**Level 1 is reviewed, not computed.** A root cannot be read off the spelling: መሆን is spelled ም-ህ-ን, and its root is that of ሆነ. Roots come from the lexicon, and a stem without a reviewed root gives `None`.

**The subword level** is a SentencePiece unigram model trained on lossless sound text with morpheme boundaries marked. During training no piece may cross a boundary, and when encoding each morpheme is encoded on its own.

## Measurements

Corpus: CC-100 Amharic after cleaning and deduplication, 2.9M sentences and 42.9M words.

**Level 4.**

- **Coverage:** 65% of corpus tokens are analysed; the remaining words are returned whole.
- **Agreement with a hand-annotated reference:** 91.3% on the inflectional affixes, over 26.7M compared tokens. The reference is a set of hand-annotated Amharic lexicons; it also marks derivation, which this package keeps inside the stem by design: it writes the causative ያደርጋል as ያ_ደርግ_አል and the passive ይታወቃል as ይ_ት_አወቅ_አል, where this package gives ይ + አደረገ + ኣል and ይ + ታወቀ + ኣል.

**Level 1.** Agreement with the reference's roots: 93.0% of tokens.

**Subword level**, on 20,000 held-out sentences (294,454 Amharic words) that neither model was trained on:

| method | vocabulary | pieces per word | morpheme boundaries kept | words with every boundary kept | exact round trip |
|---|---|---|---|---|---|
| unigram over decomposed fidel | 16K | 1.49 | 21.0% | 12.3% | 100% |
| **amharic-morph** | **16K** | 1.80 | **100%** | **100%** | **100%** |

The decomposed-fidel model follows the approach existing Amharic subword tokenizers take. It was trained on exactly the same 5.9M sentences, with the same settings and the same lossless encoding, so the difference is the morpheme boundaries alone. Keeping every boundary costs about 0.3 pieces per word.

The plain model's boundary rate falls as its vocabulary grows: on a smaller training sample it kept 40% of the boundaries at 4K, 30% at 8K, 22% at 16K and 15% at 32K. A larger vocabulary learns longer letter sequences, which span more morpheme boundaries.

## Limitations

- **Coverage.** About a third of corpus tokens come back unknown: names, rarer stems, and verbs not yet reviewed. Reaching 90% coverage needs closer to 10,000 stems.
- **Verbs.** Verb forms are analysed for the 93 verbs whose stems have been reviewed. For a verb beginning with አ, the first person singular is written without its prefix: አውቃለሁ "I know" gets the right morphemes (አወቀ + ኣል + ሁ) but is labelled as the present perfect, and the negative አላውቅም "I don't know" is not analysed. Some irregular jussives (ይወቅ, ይይ) are not covered yet.
- **Prepositions written as separate words** (ወደ, ስለ when written apart from the next word) are not yet analysed; they are recognised only as prefixes.
- **Ambiguity.** A word with two analyses (ትል, "worm" or "she says") gets one from `segment()`; `segment_all()` returns them all. Choosing between them needs context, which a word-level analyser does not have.
- **Gemination** is not represented: Amharic spelling does not mark it, so pairs such as ይሰጣል "gives" and "is given" are written alike, and `segment()` picks one reading.
- **Compounds** keep their written form. A compound written as one word (ቤተክርስቲያን) is one entry; written apart or with a hyphen (ቤተ ክርስቲያን, ቤተ-ክርስቲያን) each part is analysed, the construct form as its noun plus ኧ (ቤት + ኧ).

## Data

- **CC-100** Amharic, from the CC-Net monolingual corpora (Common Crawl). Used for the lexicon's candidate words and for training the subword model.
- **GPAC**, an Amharic news corpus, after the same cleaning and deduplication against CC-100 (3.0M new sentences). Used for training the subword model only.
- **The reference annotations** used in the measurements are the stem-based and root-based annotated lexicons published as "Morphologically Annotated Amharic Text Corpora" (SIGIR 2021). They are used for evaluation only and are not distributed with this package.

## License

MIT, for the code and the lexicon.
