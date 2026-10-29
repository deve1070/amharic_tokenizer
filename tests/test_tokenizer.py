"""
Tests for the AmharicTokenizer API.

The worked example throughout is the spec's, section 5.4:

    tokenize('ቤቶች', level='morpheme')  ->  ['ቤት', 'ኦች']
    tokenize('ቤቶች', level='fidel')     ->  ['ቤ', 'ቶ', 'ች']
    tokenize('ቤቶች', level='sound')     ->  ['ብ', 'ኤ', 'ት', 'ኦ', 'ች']
    tokenize('ቤቶች', level='root')      ->  ['ብ', 'ት']
"""

import json

import pytest


from amharic_morph import AmharicTokenizer
from amharic_morph.tokenizer import LEVELS


@pytest.fixture
def tok():
    return AmharicTokenizer()


@pytest.fixture
def small(tmp_path):
    """
    A tokenizer over a small lexicon: ቤት with a reviewed root, ሰው not
    validated, መሆን validated with its root unreviewed, and ላይ reviewed
    as having no root.
    """
    entries = [
        {
            "word": "ቤት",
            "category": "noun",
            "gender": None,
            "allows_prefix": ["የ"],
            "allows_suffix": ["ኦች", "ኡ"],
            "allows_infix": None,
            "root_consonants": ["ብ", "ት"],
            "root_source": "native_speaker",
            "source": "test",
            "validated": True,
            "notes": "",
        },
        {
            "word": "ሰው",
            "category": None,
            "gender": None,
            "allows_prefix": [],
            "allows_suffix": ["ኦች"],
            "allows_infix": None,
            "root_consonants": ["ስ", "ው"],
            "root_source": "skeleton",
            "source": "test",
            "validated": False,
            "notes": "",
        },
        {
            "word": "መሆን",
            "category": "noun",
            "gender": None,
            "allows_prefix": [],
            "allows_suffix": ["ኡ"],
            "allows_infix": None,
            "root_consonants": ["ም", "ህ", "ን"],
            "root_source": "skeleton",
            "source": "test",
            "validated": True,
            "notes": "",
        },
        {
            "word": "ላይ",
            "category": "preposition",
            "gender": None,
            "allows_prefix": ["በ"],
            "allows_suffix": [],
            "allows_infix": None,
            "root_consonants": None,
            "root_source": "native_speaker",
            "source": "test",
            "validated": True,
            "notes": "",
        },
    ]
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return AmharicTokenizer(str(path))


@pytest.fixture
def small_path(tmp_path, small):
    return str(tmp_path / "lexicon.json")


# ── The spec's worked example ────────────────────────────────────────────────


class TestSpecExample:
    def test_morpheme(self, tok):
        assert tok.tokenize("ቤቶች", level="morpheme") == ["ቤት", "ኦች"]

    def test_fidel(self, tok):
        assert tok.tokenize("ቤቶች", level="fidel") == ["ቤ", "ቶ", "ች"]

    def test_sound(self, tok):
        assert tok.tokenize("ቤቶች", level="sound") == ["ብ", "ኤ", "ት", "ኦ", "ች"]

    def test_root(self, small):
        assert small.tokenize("ቤቶች", level="root") == ["ብ", "ት"]

    def test_morpheme_is_the_default_level(self, tok):
        assert tok.tokenize("ቤቶች") == ["ቤት", "ኦች"]


class TestLevelsAreDifferentViews:
    """
    Levels 3 and 2 describe the surface word; Level 1 describes the
    independent morpheme, which is why it needs Level 4 first.
    """

    def test_fidel_round_trips_the_surface_form(self, tok):
        assert "".join(tok.tokenize("ቤቶች", level="fidel")) == "ቤቶች"

    def test_root_excludes_the_suffix_consonant(self, small):
        # Run on the surface form, ቤቶች would give ብ-ት-ች.
        assert "ች" not in small.tokenize("ቤቶች", level="root")

    def test_root_of_the_bare_stem_matches(self, small):
        assert small.tokenize("ቤት", level="root") == small.tokenize("ቤቶች", level="root")

    def test_sound_is_never_shorter_than_fidel(self, tok):
        for word in ("ቤቶች", "ሰላም", "አለ"):
            fidel = tok.tokenize(word, level="fidel")
            assert len(tok.tokenize(word, level="sound")) >= len(fidel)


# ── Text, not just single words ──────────────────────────────────────────────


class TestWords:
    def test_splits_a_sentence(self, tok):
        assert tok.words("ቤቶች ናቸው።") == ["ቤቶች", "ናቸው", "።"]

    def test_normalizes_first(self, tok):
        # ሐ normalizes to ሀ, and ፡፡ becomes ።
        assert tok.words("ሐሙስ ነው፡፡") == ["ሀሙስ", "ነው", "።"]

    def test_empty_text(self, tok):
        assert tok.words("") == []


class TestTokenizeText:
    def test_flattens_across_words(self, tok):
        assert tok.tokenize("ስለሰዎች ቤቶች") == ["ስለ", "ሰው", "ኦች", "ቤት", "ኦች"]

    def test_punctuation_survives_as_a_token(self, tok):
        assert "።" in tok.tokenize("ቤቶች ናቸው።")

    def test_root_level_skips_words_without_one(self, small):
        # A root of None is not a token, so it contributes nothing.
        assert small.tokenize("ቤቶች ኮምፒውተር", level="root") == ["ብ", "ት"]


# ── Unknown words ────────────────────────────────────────────────────────────


class TestUnknownWords:
    def test_returned_whole_at_the_morpheme_level(self, tok):
        assert tok.tokenize("ኮምፒውተር", level="morpheme") == ["ኮምፒውተር"]

    def test_root_is_none(self, tok):
        assert tok.root("ኮምፒውተር") is None

    def test_flagged_in_the_analysis(self, tok):
        assert tok.analyze("ኮምፒውተር")["known"] is False

    def test_surface_levels_still_work(self, tok):
        # Levels 3 and 2 need no lexicon.
        assert tok.tokenize("ኮምፒውተር", level="fidel") == list("ኮምፒውተር")


# ── Lexicon loading ──────────────────────────────────────────────────────────


class TestLexiconLoading:
    def test_packaged_lexicon_loads(self, tok):
        assert len(tok.lexicon) > 0

    def test_unvalidated_entries_are_excluded(self, small):
        assert "ሰው" not in small.lexicon
        assert "ቤት" in small.lexicon

    def test_unvalidated_entries_do_not_anchor_segmentation(self, small):
        assert small.tokenize("ሰዎች") == ["ሰዎች"]

    def test_can_include_unvalidated_explicitly(self, small_path):
        loose = AmharicTokenizer(small_path, validated_only=False)
        assert loose.tokenize("ሰዎች") == ["ሰው", "ኦች"]

    def test_custom_lexicon_path(self, small):
        assert small.tokenize("ቤቶች") == ["ቤት", "ኦች"]


# ── analyze() ────────────────────────────────────────────────────────────────


class TestAnalyze:
    def test_reports_every_level(self, tok):
        assert set(tok.analyze("ቤቶች")) == {
            "word",
            "morpheme",
            "fidel",
            "sound",
            "root",
            "known",
            "aspect",
        }

    def test_agrees_with_tokenize(self, small):
        analysis = small.analyze("ቤቶች")
        for level in ("morpheme", "fidel", "sound", "root"):
            assert analysis[level] == small.tokenize("ቤቶች", level=level)


# ── Level 1 reads the reviewed root ──────────────────────────────────────────


class TestLevel1ReadsTheLexicon:
    """
    The root is a ruling stored on the entry, not computed from the
    stem's spelling. An unreviewed root is withheld, as an unknown word's
    segmentation is.
    """

    def test_reviewed_root_comes_from_the_entry(self, small):
        assert small.root("የቤቶች") == ["ብ", "ት"]

    def test_unreviewed_root_is_none(self, small):
        # The stored skeleton ም-ህ-ን keeps the infinitive's መ; it is not
        # reported as a root.
        assert small.segment("መሆኑ").known is True
        assert small.root("መሆኑ") is None
        assert small.analyze("መሆኑ")["root"] is None

    def test_reviewed_none_is_none(self, small):
        assert small.root("በላይ") is None

    def test_root_level_skips_unreviewed_roots(self, small):
        assert small.tokenize("ቤቶች መሆኑ በላይ", level="root") == ["ብ", "ት"]


# ── Level names ──────────────────────────────────────────────────────────────


class TestLevelNames:
    def test_unknown_level_raises(self, tok):
        with pytest.raises(ValueError, match="unknown level"):
            tok.tokenize("ቤቶች", level="syllable")

    def test_bpe_without_a_model_names_the_missing_file(self, tmp_path):
        tok = AmharicTokenizer(subword_model=tmp_path / "missing.model")
        with pytest.raises(FileNotFoundError, match="missing.model"):
            tok.tokenize("ቤቶች", level="bpe")

    def test_other_levels_never_load_the_subword_model(self, tmp_path):
        tok = AmharicTokenizer(subword_model=tmp_path / "missing.model")
        assert tok.tokenize("ቤቶች") == ["ቤት", "ኦች"]

    def test_every_advertised_level_is_reachable(self, tok):
        for level in LEVELS:
            if level == "bpe":
                continue
            assert isinstance(tok.tokenize("ቤቶች", level=level), list)


# ── Suffix readings surface through the API ──────────────────────────────────


class TestSuffixReadingsThroughTheAPI:
    def test_definite_on_a_noun(self, small):
        parts = small.segment("ቤቱ").parts
        assert [(p.text, p.readings) for p in parts] == [
            ("ቤት", None),
            ("ኡ", ("definite", "possessive_3ms")),
        ]


# ── Per-word cache ───────────────────────────────────────────────────────────
#
# A cache hit hands back the Segmentation object computed the first time; a
# miss builds a new one. Identity therefore shows which of the two happened.


class TestSegmentCache:
    def test_repeated_word_is_served_from_the_cache(self, tok):
        assert tok.segment("ቤቶች") is tok.segment("ቤቶች")

    def test_size_zero_disables_it(self):
        tok = AmharicTokenizer(cache_size=0)
        first, second = tok.segment("ቤቶች"), tok.segment("ቤቶች")
        assert first == second
        assert first is not second

    def test_least_recently_used_word_is_evicted(self):
        tok = AmharicTokenizer(cache_size=1)
        first = tok.segment("ቤቶች")
        tok.segment("ቤቱ")
        assert tok.segment("ቤቶች") is not first

    def test_cached_output_matches_uncached(self, tok):
        text = "ቤቶች ቤቱ ቤቶች ናቸው። ቤቶች ኮምፒውተር ቤቱ"
        uncached = AmharicTokenizer(cache_size=0)
        for level in ("morpheme", "root"):
            assert tok.tokenize(text, level=level) == uncached.tokenize(text, level=level)

    def test_negative_size_raises(self):
        with pytest.raises(ValueError):
            AmharicTokenizer(cache_size=-1)
