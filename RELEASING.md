# Releasing

The release is blocked on one thing: the subword model trained on the full
corpus. Everything else below is ready.

## 1. Train the subword model (16 GB machine)

The two corpora together are 5.95M sentences: CC-100 2,918,058 and the
GPAC sentences not already in CC-100, 3,030,231. Training needs about
1.1 GB of memory per 300K lines sampled, so all of them (about 22 GB) do
not fit in 16 GB. Sample 3.5M lines, about 13 GB:

```bash
pip install -e ".[dev]"

# One corpus file, CC-100 then the GPAC sentences not already in it.
cat corpus/corpus.txt corpus/gpac/corpus.txt > corpus/combined.txt

python scripts/train_bpe.py corpus/combined.txt corpus/bpe_full --vocab 16000 --lines 3500000 --threads 8
```

Writing the training text takes about an hour (about 1,700 lines per
second) and needs about 2 GB of disk. If training runs out of memory, use `--lines 3000000`. The last
line printed must say `0 of 5000 lines fail the round trip`.

The files to copy to the 16 GB machine are the repository and
`corpus/corpus.txt` and `corpus/gpac/corpus.txt` (1.3 GB together; both
are gitignored).

Rebuilding `corpus/gpac/corpus.txt` from the raw file, if needed:

```bash
python -m amharic_morph.corpus_runner corpus/GPAC.txt \
    --output corpus/gpac/corpus.txt --metadata corpus/gpac/metadata.jsonl \
    --statistics corpus/gpac/statistics.json --source gpac --drop-tigrinya \
    --dedup-against corpus/corpus.txt
```

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
