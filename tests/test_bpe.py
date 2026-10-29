"""
Tests for the BPE level: cutting a word's sound text at its morpheme
boundaries, the training text, and encoding.
"""

import pytest

from amharic_morph.bpe import sound_segments
from amharic_morph.levels import sound_text
from amharic_morph.tokenizer import AmharicTokenizer


@pytest.fixture(scope="module")
def tok():
    return AmharicTokenizer()


def cut(tok, word):
    return sound_segments(tok.segment(word))


class TestSoundSegments:
    """
    In sound text a fused suffix separates cleanly from its stem: the
    carrier's vowel belongs to the suffix. ቤቶች is ብኤት | ኦች.
    """

    def test_fused_plural(self, tok):
        assert cut(tok, "ቤቶች") == ["ብኤት", "ኦች"]

    def test_restored_carrier_is_cut_as_written(self, tok):
        # ከተማ + ኦች: the written stem is ከተም.
        assert cut(tok, "ከተሞች") == ["ክኧትኧም", "ኦች"]

    def test_imperfective(self, tok):
        assert cut(tok, "ይሆናል") == ["ይ", "ህኦን", "ኣል"]

    def test_perfective_subject_on_an_order6_carrier(self, tok):
        assert cut(tok, "ሄድኩ") == ["ህኤድ", "ክኡ"]

    def test_linking_vowel_goes_with_the_suffix(self, tok):
        # IRIT የ_ሚ_ኖር_ኧ_ው.
        assert cut(tok, "የሚኖረው") == ["ይኧ", "ምኢ", "ንኦር", "ኧው"]

    def test_stem_vowel(self, tok):
        assert cut(tok, "የሚገባ") == ["ይኧ", "ምኢ", "ግኧብ", "ኣ"]

    def test_order8_carrier(self, tok):
        assert cut(tok, "ቤቷ") == ["ብኤት", "ዋ"]

    def test_labiovelar_stays_whole(self, tok):
        assert cut(tok, "ቋንቋዎች") == ["ቅʷኣንቅʷኣ", "ውኦች"]

    def test_infix_splits_the_stem(self, tok):
        # ትላልቅ: ላ is the copy inserted into ትልቅ.
        assert cut(tok, "ትላልቅ") == ["ት", "ልኣ", "ልቅ"]

    def test_unknown_word_is_whole(self, tok):
        assert cut(tok, "ሆስፒታል") == [sound_text("ሆስፒታል")]

    def test_segments_rebuild_the_word(self, tok):
        for word in ("ቤቶቹን", "ለቤቶቻችን", "አልነበረም", "ይገባዋል", "ትላልቆች", "ብሎ"):
            assert "".join(cut(tok, word)) == sound_text(word)


# ── Training text, training and encoding ─────────────────────────────────────

from amharic_morph.bpe import (  # noqa: E402
    DELIMITER,
    SubwordModel,
    training_line,
    train,
)

LINES = [
    "ቤቶቹን አየሁ ።",
    "ይሆናል ብሎ ነበር ።",
    "ቤቶች የሚገባ ይሆናል ።",
    "ቋንቋዎች <LATIN>mast</LATIN> ቤቶች ።",
]


class TestTrainingLine:
    def test_morpheme_aware_line(self, tok):
        assert training_line("ቤቶች ይሆናል", tok, morphemes=True) == (
            f" ብኤት{DELIMITER}ኦች ይ{DELIMITER}ህኦን{DELIMITER}ኣል"
        )

    def test_plain_line(self, tok):
        assert training_line("ቤቶች ይሆናል", tok, morphemes=False) == " ብኤትኦች ይህኦንኣል"

    def test_non_ethiopic_tokens_pass_through(self, tok):
        assert training_line("<LATIN>mast</LATIN> ።", tok, morphemes=True) == " <LATIN>mast</LATIN> ።"


@pytest.fixture(scope="module")
def model_files(tok, tmp_path_factory):
    pytest.importorskip("sentencepiece")
    out = tmp_path_factory.mktemp("models")
    files = {}
    for morphemes in (True, False):
        text = out / f"{morphemes}.txt"
        text.write_text("\n".join(training_line(l, tok, morphemes) for l in LINES * 50) + "\n", encoding="utf-8")
        prefix = out / ("morph" if morphemes else "plain")
        train(text, prefix, vocab_size=400)
        files[morphemes] = f"{prefix}.model"
    return files


@pytest.fixture(scope="module")
def models(tok, model_files):
    return {m: SubwordModel(path, tok, morphemes=m) for m, path in model_files.items()}


class TestSubwordModel:
    def test_pieces_never_cross_a_morpheme_boundary(self, models, tok):
        model = models[True]
        for word in ("ቤቶቹን", "ይሆናል", "ቋንቋዎች"):
            pieces = model.encode_word(word)
            boundaries = set()
            position = 0
            for segment in sound_segments(tok.segment(word)):
                position += len(segment)
                boundaries.add(position)
            position = 0
            ends = set()
            for piece in pieces:
                position += len(piece.lstrip("▁"))
                ends.add(position)
            assert boundaries <= ends, (word, pieces)

    def test_only_the_first_piece_marks_the_word(self, models):
        pieces = models[True].encode_word("ቤቶች")
        assert pieces[0].startswith("▁")
        assert not any(p.startswith("▁") for p in pieces[1:])

    def test_round_trip(self, models):
        for model in models.values():
            for line in LINES + ["ሆስፒታል ገባ ።"]:
                assert model.decode(model.encode(line)) == line

    def test_ids_round_trip(self, models):
        model = models[True]
        assert model.decode_ids(model.encode_ids("ቤቶቹን አየሁ ።")) == "ቤቶቹን አየሁ ።"

    def test_readable_pieces(self, models):
        # Shown as fidel: ▁ቤት + ኦች, not ▁ብኤት + ኦች.
        assert SubwordModel.readable(["▁ብኤት", "ኦች"]) == ["▁ቤት", "ኦች"]
        assert SubwordModel.readable(["▁ይ", "ህኦን", "ኣል"]) == ["▁ይ", "ሆን", "ኣል"]

    def test_latin_tag_is_one_piece(self, models):
        assert "<LATIN>" in models[True].encode("<LATIN>mast</LATIN>")


class TestTokenizerSubwordLevel:
    """level="bpe", encode() and decode() on AmharicTokenizer."""

    @pytest.fixture(scope="class")
    def sub_tok(self, model_files):
        return AmharicTokenizer(subword_model=model_files[True])

    def test_bpe_level_gives_readable_pieces(self, sub_tok):
        pieces = sub_tok.tokenize("ቤቶቹን አየሁ።", level="bpe")
        assert pieces[0].startswith("▁")
        assert "".join(pieces).replace("▁", "") == "".join(SubwordModel.readable(
            sub_tok.subword_model.encode("ቤቶቹን አየሁ ።")
        )).replace("▁", "")

    def test_encode_decode_round_trip(self, sub_tok):
        # The text comes back in its normalized form, words joined by one
        # space and punctuation padded, as tokenize() sees it.
        text = "ቤቶቹን አየሁ ። ይሆናል ብሎ ነበር ።"
        assert sub_tok.decode(sub_tok.encode(text)) == text

    def test_encode_normalizes_first(self, sub_tok):
        assert sub_tok.decode(sub_tok.encode("ቤቶቹን አየሁ።")) == "ቤቶቹን አየሁ ።"
