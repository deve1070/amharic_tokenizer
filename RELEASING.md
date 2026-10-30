# Releasing

Steps 1 and 2 are kept here so the model can be retrained whenever Level 4
changes, since the model is trained on Level 4's cuts.

## 1. Train the subword model

The corpus is reduced to a table of distinct segments and their counts, and
the model trains on that table, so every line is used and a few gigabytes of
memory are enough. Start from the raw corpora, `corpus/am.txt` (CC-100) and
`corpus/GPAC.txt`, both gitignored.

On Windows, one script does all of it, with a log per step in `logs\`:

```bat
scripts\train_all.bat
```

On Linux, the same steps:

```bash
python -m amharic_morph.corpus_runner corpus/am.txt --output corpus/corpus.txt \
    --metadata corpus/metadata.jsonl --statistics corpus/statistics.json --drop-tigrinya
python -m amharic_morph.corpus_runner corpus/GPAC.txt --output corpus/gpac/corpus.txt \
    --metadata corpus/gpac/metadata.jsonl --statistics corpus/gpac/statistics.json \
    --source gpac --drop-tigrinya --dedup-against corpus/corpus.txt
cat corpus/corpus.txt corpus/gpac/corpus.txt > corpus/combined.txt
python scripts/train_bpe.py corpus/combined.txt corpus/bpe_full --vocab 16000 --threads 8
python scripts/train_bpe.py corpus/combined.txt corpus/bpe_full --vocab 16000 --threads 8 --plain
```

The last line each training prints must say `0 of 5000 lines fail the round
trip`. The plain model is the comparison baseline and is not shipped.

## 2. Ship the model

```bash
mkdir -p src/amharic_morph/models
cp corpus/bpe_full/morph_16k.model src/amharic_morph/models/
```

Then:

- add a check to `tests/test_installed_package.py` that the wheel carries
  `amharic_morph/models/morph_16k.model` and that `level="bpe"` works once
  installed;
- replace the *preliminary* subword figures in `README.md` with the
  full-corpus measurements, and remove the pre-release note;
- update `CHANGELOG.md`.

## 3. Merge

Push the branch, open a pull request into `main`, and merge once the tests
workflow passes on Python 3.10 to 3.13.

## 4. Register Trusted Publishing (once, on your accounts)

On both https://test.pypi.org and https://pypi.org, add a *pending
publisher* (Account settings, Publishing):

| field | value |
|---|---|
| project name | `amharic-morph` |
| owner | `deve1070` |
| repository | `amharic_tokenizer` |
| workflow | `release.yml` |
| environment | `testpypi` on TestPyPI, `pypi` on PyPI |

In the GitHub repository settings, create the two environments, `testpypi`
and `pypi`.

## 5. Release candidate on TestPyPI

```bash
git tag v0.1.0rc1 && git push origin v0.1.0rc1
```

Set `__version__ = "0.1.0rc1"` in `src/amharic_morph/__init__.py` first, so
the tag and the version agree. Then install it in a clean environment:

```bash
python -m venv /tmp/rc && /tmp/rc/bin/pip install \
    --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
    "amharic-morph[bpe]==0.1.0rc1"
/tmp/rc/bin/amharic-morph "ልጆቹ ትምህርት ቤት ሄዱ።"
```

## 6. Release

Set `__version__ = "0.1.0"`, commit, then:

```bash
git tag v0.1.0 && git push origin v0.1.0
```
