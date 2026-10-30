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

    def test_fused_prefix(self, tok):
        # ያ is ይ + the stem's ኣ: the cut is after the prefix's consonant.
        assert cut(tok, "ያለው") == ["ይ", "ኣልኧ", "ው"]
        assert cut(tok, "ያውቃል") == ["ይ", "ኣውቅ", "ኣል"]
        assert cut(tok, "የሚያውቅ") == ["ይኧ", "ምኢ", "ይ", "ኣውቅ"]
        assert cut(tok, "እስካሁን") == ["እስክ", "ኣህኡን"]

    def test_e_fusion_keeps_the_prefix_vowel(self, tok):
        # በዚህ = በ + እዚህ: the preposition is written whole, the እ is gone.
        assert cut(tok, "በዚህ") == ["ብኧ", "ዝኢህ"]

    def test_e_dropped_from_another_prefix(self, tok):
        # በንደዚህ: በ drops the እ of እንደ, and እንደ drops the እ of እዚህ.
        assert cut(tok, "በንደዚህ") == ["ብኧ", "ንድኧ", "ዝኢህ"]

    def test_fused_onto_another_prefix(self, tok):
        # ካልሆነም: ከ absorbs the አ of the negative አል, not of the stem.
        assert cut(tok, "ካልሆነም") == ["ክ", "ኣል", "ህኦንኧ", "ም"]

    def test_unknown_word_is_whole(self, tok):
        assert cut(tok, "ሆስፒታል") == [sound_text("ሆስፒታል")]

    def test_segments_rebuild_the_word(self, tok):
        for word in ("ቤቶቹን", "ለቤቶቻችን", "አልነበረም", "ይገባዋል", "ትላልቆች", "ብሎ"):
            assert "".join(cut(tok, word)) == sound_text(word)


# ── Training text, training and encoding ─────────────────────────────────────

from amharic_morph.bpe import (  # noqa: E402
    TAGS,
    SubwordModel,
    count_segments,
    train,
    word_segments,
    write_segment_counts,
)

LINES = [
    "ቤቶቹን አየሁ ።",
    "ይሆናል ብሎ ነበር ።",
    "ቤቶች የሚገባ ይሆናል ።",
    "ቋንቋዎች <LATIN>mast</LATIN> ቤቶች ።",
]


class TestTrainingSegments:
    """
    Training sees what encoding encodes: each segment on its own, the
    word's first one marked with a leading space (▁).
    """

    def test_morpheme_aware_word(self, tok):
        assert word_segments("ቤቶች", tok, morphemes=True) == [" ብኤት", "ኦች"]

    def test_plain_word(self, tok):
        assert word_segments("ቤቶች", tok, morphemes=False) == [" ብኤትኦች"]

    def test_non_ethiopic_token_is_one_segment(self, tok):
        assert word_segments("<LATIN>mast</LATIN>", tok, morphemes=True) == [" <LATIN>mast</LATIN>"]

    def test_counts(self, tok):
        counts = count_segments(["ቤቶች ይሆናል", "ቤቶች"], tok, morphemes=True)
        assert counts == {" ብኤት": 2, "ኦች": 2, " ይ": 1, "ህኦን": 1, "ኣል": 1}


@pytest.fixture(scope="module")
def model_files(tok, tmp_path_factory):
    pytest.importorskip("sentencepiece")
    out = tmp_path_factory.mktemp("models")
    files = {}
    for morphemes in (True, False):
        counts = out / f"{morphemes}.tsv"
        write_segment_counts(count_segments(LINES * 50, tok, morphemes), counts)
        prefix = out / ("morph" if morphemes else "plain")
        train(counts, prefix, vocab_size=400, min_symbol_count=10)
        files[morphemes] = f"{prefix}.model"
    return files


@pytest.fixture(scope="module")
def models(tok, model_files):
    return {m: SubwordModel(path, tok, morphemes=m) for m, path in model_files.items()}


class TestSubwordModel:
    def test_frequent_punctuation_is_one_piece_with_its_word_mark(self, models):
        # " ።" is its own row of the count table, so training rarely proposes
        # ▁። as a piece; frequent one-character words are given as pieces.
        assert models[True].encode("ቤቶች ።")[-1:] == ["▁።"]

    def test_no_learned_piece_crosses_a_segment(self, model_files, tok):
        # Every piece is part of one training segment: training can never
        # learn ▁ይኣልኧው when ያለው is always cut ይ | ኣልኧ | ው.
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model_files[True])
        segments = list(count_segments(LINES, tok, morphemes=True))
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i).replace("▁", " ")
            if sp.is_control(i) or sp.is_byte(i) or sp.is_unknown(i) or piece.strip() in TAGS:
                continue
            assert any(piece in s for s in segments), piece

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


class TestShippedModel:
    """The model in the package, found without being given a path."""

    def test_default_tokenizer_finds_it(self):
        pytest.importorskip("sentencepiece")
        tok = AmharicTokenizer()
        assert tok.tokenize("ቤቶቻችንን", level="bpe") == ["▁ቤት", "ኦች", "ኣችን", "ን"]

    def test_round_trip_with_the_shipped_model(self):
        pytest.importorskip("sentencepiece")
        tok = AmharicTokenizer()
        for text in ("ቤቶቻችንን የሚገባቸው ሰዎች አልነበሩም።", "ቋንቋዎች በ2015 ተጀመሩ።", "COVID-19 ወረርሽኝ"):
            # decode() returns the text as words() normalizes it.
            assert tok.decode(tok.encode(text)) == " ".join(tok.words(text))


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
