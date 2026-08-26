"""
Tests for the representation levels.

The spec's worked example is ቤቶች:
    Level 3 fidel   ቤ ቶ ች
    Level 2 sound   ብ ኤ ት ኦ ች
    Level 1 root    ብ ት
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_tokenizer.levels import (  # noqa: E402
    decompose_fidel,
    fidel_order,
    is_ethiopic,
    is_glottal,
    level1_root,
    level2_sound,
    level3_fidel,
    to_order,
    to_order6,
)


# ── Fidel arithmetic ──────────────────────────────────────────────────────────


class TestFidelOrder:
    """Families are 8-aligned, so order = codepoint % 8 + 1"""

    def test_all_seven_orders(self):
        for order, char in enumerate("ተቱቲታቴትቶ", start=1):
            assert fidel_order(char) == order

    def test_labialized_is_order8(self):
        assert fidel_order("ሏ") == 8

    def test_non_ethiopic(self):
        assert fidel_order("a") == 0
        assert fidel_order("") == 0


class TestToOrder:
    """The consonant is kept and the vowel replaced"""

    def test_walks_a_family(self):
        for order, char in enumerate("ተቱቲታቴትቶ", start=1):
            assert to_order("ት", order) == char

    def test_reaches_the_labialized_form(self):
        assert to_order("ል", 8) == "ሏ"

    def test_starting_order_does_not_matter(self):
        # A fused vowel overwrites the stem's own, so the carrier's
        # order says nothing about where it came from.
        for char in "ተቱቲታቴትቶ":
            assert to_order(char, 4) == "ታ"

    def test_verb_stem_endings(self):
        assert to_order("ደ", 4) == "ዳ"
        assert to_order("ነ", 4) == "ና"

    def test_non_ethiopic_unchanged(self):
        assert to_order("a", 3) == "a"

    def test_rejects_order_out_of_range(self):
        for bad in (0, 9, -1):
            try:
                to_order("ት", bad)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for order {bad}")


class TestToOrder6:
    """Order 6 is the bare consonant"""

    def test_every_order_reduces_to_the_same_consonant(self):
        for char in "ተቱቲታቴትቶ":
            assert to_order6(char) == "ት"

    def test_labialized_reduces(self):
        assert to_order6("ሏ") == "ል"

    def test_non_ethiopic_unchanged(self):
        assert to_order6("a") == "a"


class TestIsGlottal:
    """The አ family carries the glottal stop"""

    def test_whole_family(self):
        assert all(is_glottal(c) for c in "አኡኢኣኤእኦኧ")

    def test_other_families_are_not(self):
        assert not any(is_glottal(c) for c in "ቤትችልም")

    def test_non_ethiopic(self):
        assert is_glottal("a") is False


class TestIsEthiopic:
    def test_ethiopic(self):
        assert is_ethiopic("ቤ") is True

    def test_latin(self):
        assert is_ethiopic("a") is False

    def test_empty(self):
        assert is_ethiopic("") is False


# ── Decomposition table ───────────────────────────────────────────────────────


class TestDecomposeFidel:
    """
    Order 1 contributes ኧ, not አ.

    That reservation is what lets Level 1 tell a glottal consonant from
    a vowel: an አ-family letter in a decomposition can only have come
    from the surface word.
    """

    def test_order1_uses_ea_not_glottal_a(self):
        assert decompose_fidel("ተ") == ("ት", "ኧ")

    def test_order2(self):
        assert decompose_fidel("ቱ") == ("ት", "ኡ")

    def test_order3(self):
        assert decompose_fidel("ቲ") == ("ት", "ኢ")

    def test_order4(self):
        assert decompose_fidel("ታ") == ("ት", "ኣ")

    def test_order5(self):
        assert decompose_fidel("ቴ") == ("ት", "ኤ")

    def test_order6_has_no_vowel(self):
        assert decompose_fidel("ት") == ("ት", "")

    def test_order7(self):
        assert decompose_fidel("ቶ") == ("ት", "ኦ")

    def test_labialized_uses_wa(self):
        assert decompose_fidel("ሏ") == ("ል", "ዋ")

    def test_order1_never_produces_glottal_a(self):
        # If order 1 produced አ, a decomposed vowel would be
        # indistinguishable from a surface glottal consonant.
        for char in "ተለመሰረ":
            assert decompose_fidel(char)[1] != "አ"


# ── Level 3 ───────────────────────────────────────────────────────────────────


class TestLevel3Fidel:
    """Surface script as written"""

    def test_spec_example(self):
        assert level3_fidel("ቤቶች") == ["ቤ", "ቶ", "ች"]

    def test_single_character(self):
        assert level3_fidel("ቤ") == ["ቤ"]

    def test_empty(self):
        assert level3_fidel("") == []


# ── Level 2 ───────────────────────────────────────────────────────────────────


class TestLevel2Sound:
    """Full fidel decomposition"""

    def test_spec_example(self):
        assert level2_sound("ቤቶች") == ["ብ", "ኤ", "ት", "ኦ", "ች"]

    def test_spec_example_as_string(self):
        assert "".join(level2_sound("ቤቶች")) == "ብኤትኦች"

    def test_bare_consonants_add_no_vowel(self):
        assert level2_sound("ት") == ["ት"]

    def test_order1_word(self):
        # ለ is order 1, so it contributes ኧ.
        assert level2_sound("ለ") == ["ል", "ኧ"]

    def test_glottal_initial_word(self):
        assert level2_sound("አለ") == ["እ", "ኧ", "ል", "ኧ"]

    def test_internal_glottal_is_bare(self):
        # እ is order 6, so it contributes no vowel of its own.
        assert level2_sound("ጣእም") == ["ጥ", "ኣ", "እ", "ም"]

    def test_labialized(self):
        assert level2_sound("ሏ") == ["ል", "ዋ"]

    def test_empty(self):
        assert level2_sound("") == []


# ── Level 1 ───────────────────────────────────────────────────────────────────


class TestLevel1Root:
    """
    Both rules together: decomposition vowels never enter the root, and
    among surface አ-family letters only a word-initial one counts.
    """

    def test_spec_example(self):
        assert level1_root("ቤት") == ["ብ", "ት"]

    def test_decomposition_vowels_excluded(self):
        # ኤ and ኦ appear in ብኤትኦች but never existed in the surface form.
        root = level1_root("ቤት")
        assert "ኤ" not in root and "ኦ" not in root

    def test_word_initial_glottal_is_a_consonant(self):
        assert level1_root("አለ") == ["እ", "ል"]

    def test_word_internal_glottal_is_a_template_vowel(self):
        assert level1_root("ጣእም") == ["ጥ", "ም"]

    def test_vowels_stripped_from_every_order(self):
        assert level1_root("ሰላም") == ["ስ", "ል", "ም"]

    def test_labialized_consonant_extracted(self):
        assert level1_root("ሏ") == ["ል"]

    def test_single_glottal(self):
        assert level1_root("አ") == ["እ"]

    def test_empty_returns_none(self):
        assert level1_root("") is None

    def test_non_ethiopic_returns_none(self):
        assert level1_root("abc") is None


class TestLevel1RequiresAMorpheme:
    """
    Level 1 takes the independent morpheme, not the surface word. Given
    a whole word it returns a root polluted by affix consonants, which
    is exactly why the computational order is 4 -> 3 -> 2 -> 1.
    """

    def test_surface_word_gives_a_polluted_root(self):
        # The ች of the plural suffix ends up inside the root.
        assert level1_root("ቤቶች") == ["ብ", "ት", "ች"]

    def test_the_morpheme_gives_the_correct_root(self):
        assert level1_root("ቤት") == ["ብ", "ት"]


# ── Cross-level consistency ───────────────────────────────────────────────────


class TestLevelsAgree:
    """The levels are different views of the same word"""

    def test_level2_consonants_match_level1_for_a_clean_stem(self):
        consonants = [c for c in level2_sound("ቤት") if c not in "ኧኡኢኣኤኦዋ"]
        assert consonants == level1_root("ቤት")

    def test_level3_roundtrips(self):
        assert "".join(level3_fidel("ቤቶች")) == "ቤቶች"

    def test_level2_is_never_shorter_than_level3(self):
        for word in ("ቤቶች", "ሰላም", "አለ", "ጣእም"):
            assert len(level2_sound(word)) >= len(level3_fidel(word))
