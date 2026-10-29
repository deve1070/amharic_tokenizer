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
