"""
Tests for Layer 0 — Character Normalization

Covers:
  - Order-level homophone normalization for ሀ, ሰ, አ, ፀ families
  - Labialized character unification (e.g. ሩዋ → ሯ)
  - Single character alternates (ቊ → ቁ, ኵ → ኩ)
  - Edge cases: empty string, Latin, numerals, punctuation, mixed script

Run with: python -m pytest tests/test_normalizer.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amharic_tokenizer.normalizer import normalize, normalize_lines

# ── ሀ family ─────────────────────────────────────────────────────────────────


class TestHaFamily:
    """All order variants of the hä family normalize to the ሀ series"""

    def test_order1_ha_variant_1(self):
        assert normalize("ሃ") == "ሀ"

    def test_order1_ha_variant_2(self):
        assert normalize("ኃ") == "ሀ"

    def test_order1_ha_variant_3(self):
        assert normalize("ሐ") == "ሀ"

    def test_order1_ha_variant_4(self):
        assert normalize("ሓ") == "ሀ"

    def test_order1_ha_variant_5(self):
        assert normalize("ኻ") == "ሀ"

    def test_order2_hu(self):
        assert normalize("ሑ") == "ሁ"
        assert normalize("ኁ") == "ሁ"

    def test_order3_hi(self):
        assert normalize("ሒ") == "ሂ"
        assert normalize("ኂ") == "ሂ"

    def test_order4_he(self):
        assert normalize("ሔ") == "ሄ"
        assert normalize("ኌ") == "ሄ"

    def test_order5_h(self):
        assert normalize("ሕ") == "ህ"

    def test_order6_ho(self):
        assert normalize("ሖ") == "ሆ"
        assert normalize("ኆ") == "ሆ"
        assert normalize("ኾ") == "ሆ"

    def test_ha_in_word(self):
        # ሐሙስ (Thursday) — ሐ and ሀ represent the same sound
        assert normalize("ሐሙስ") == "ሀሙስ"

    def test_ha_variants_become_identical(self):
        # Different spellings of the same word become identical
        assert normalize("ሃይሌ") == normalize("ሀይሌ")


# ── ሰ family ─────────────────────────────────────────────────────────────────


class TestSeFamily:
    """All order variants of the se family normalize to the ሰ series"""

    def test_order1_se(self):
        assert normalize("ሠ") == "ሰ"

    def test_order2_su(self):
        assert normalize("ሡ") == "ሱ"

    def test_order3_si(self):
        assert normalize("ሢ") == "ሲ"

    def test_order4_sa(self):
        assert normalize("ሣ") == "ሳ"

    def test_order5_se_long(self):
        assert normalize("ሤ") == "ሴ"

    def test_order6_s(self):
        assert normalize("ሥ") == "ስ"

    def test_order7_so(self):
        assert normalize("ሦ") == "ሶ"

    def test_selam_normalization(self):
        assert normalize("ሠላም") == normalize("ሰላም")

    def test_se_in_word(self):
        assert normalize("ሠርቶ") == "ሰርቶ"


# ── አ family ─────────────────────────────────────────────────────────────────


class TestAaFamily:
    """All order variants of the ā family normalize to the አ series"""

    def test_order1_a_variant_1(self):
        assert normalize("ዓ") == "አ"

    def test_order1_a_variant_2(self):
        assert normalize("ኣ") == "አ"

    def test_order1_a_variant_3(self):
        assert normalize("ዐ") == "አ"

    def test_order2_u(self):
        assert normalize("ዑ") == "ኡ"

    def test_order3_i(self):
        assert normalize("ዒ") == "ኢ"

    def test_order4_e(self):
        assert normalize("ዔ") == "ኤ"

    def test_order5_schwa(self):
        assert normalize("ዕ") == "እ"

    def test_order6_o(self):
        assert normalize("ዖ") == "ኦ"

    def test_ayin_in_word(self):
        assert normalize("ዐይን") == "አይን"

    def test_year_normalization(self):
        assert normalize("ዓመት") == normalize("አመት")


# ── ጸ family ─────────────────────────────────────────────────────────────────


class TestTseFamily:
    """
    ፀ series normalizes to ጸ series.
    ጸ is canonical per spec section 2.5, which is also the rare → common
    direction: in CC-100 the ጸ series consistently outnumbers the ፀ series.
    """

    def test_order1_tse(self):
        assert normalize("ፀ") == "ጸ"

    def test_order2_tsu(self):
        assert normalize("ፁ") == "ጹ"

    def test_order3_tsi(self):
        assert normalize("ፂ") == "ጺ"

    def test_order4_tsa(self):
        assert normalize("ፃ") == "ጻ"

    def test_order5_tse_long(self):
        assert normalize("ፄ") == "ጼ"

    def test_order6_ts(self):
        assert normalize("ፅ") == "ጽ"

    def test_order7_tso(self):
        assert normalize("ፆ") == "ጾ"

    def test_sun_word(self):
        # ፀሀይ → ጸሀይ  (the ፀ gets normalized to ጸ)
        assert normalize("ፀሀይ") == "ጸሀይ"

    def test_canonical_form_is_stable(self):
        # ጸ is canonical, so it must survive normalization unchanged
        assert normalize("ጸሀይ") == "ጸሀይ"


# ── Labialized characters ─────────────────────────────────────────────────────


class TestLabializedUnification:
    """
    Sequences of order-2 character + ዋ/አ should unify to
    the single labialized (order-8) character.
    """

    def test_lu_wa(self):
        assert normalize("ሉዋ") == "ሏ"

    def test_lu_a(self):
        assert normalize("ሉአ") == "ሏ"

    def test_mu_wa(self):
        assert normalize("ሙዋ") == "ሟ"

    def test_tu_wa(self):
        assert normalize("ቱዋ") == "ቷ"

    def test_ru_wa(self):
        assert normalize("ሩዋ") == "ሯ"

    def test_su_wa(self):
        assert normalize("ሱዋ") == "ሷ"

    def test_shu_wa(self):
        assert normalize("ሹዋ") == "ሿ"

    def test_qu_wa(self):
        assert normalize("ቁዋ") == "ቋ"

    def test_bu_wa(self):
        assert normalize("ቡዋ") == "ቧ"

    def test_chu_wa(self):
        assert normalize("ቹዋ") == "ቿ"

    def test_hu_wa(self):
        assert normalize("ሁዋ") == "ኋ"

    def test_nu_wa(self):
        assert normalize("ኑዋ") == "ኗ"

    def test_nyu_wa(self):
        assert normalize("ኙዋ") == "ኟ"

    def test_ku_wa(self):
        assert normalize("ኩዋ") == "ኳ"

    def test_zu_wa(self):
        assert normalize("ዙዋ") == "ዟ"

    def test_gu_wa(self):
        assert normalize("ጉዋ") == "ጓ"

    def test_du_wa(self):
        assert normalize("ዱዋ") == "ዷ"

    def test_de_wa_is_not_labialized(self):
        # ደ is order 1, not order 2, and ደዋ is a legitimate sequence in
        # ordinary words. Collapsing it corrupts real text.
        assert normalize("ድሬደዋ") == "ድሬደዋ"
        assert normalize("ወስደዋል") == "ወስደዋል"

    def test_tu2_wa(self):
        assert normalize("ጡዋ") == "ጧ"

    def test_chu2_wa(self):
        assert normalize("ጩዋ") == "ጯ"

    def test_fu_wa(self):
        assert normalize("ፉዋ") == "ፏ"


# ── Single character alternates ───────────────────────────────────────────────


class TestSingleCharacterAlternates:
    """Some characters have rarely-used alternate codepoints"""

    def test_qe_alternate(self):
        # ቊ is an alternate form of ቁ
        assert normalize("ቊ") == "ቁ"

    def test_ku_alternate(self):
        # ኵ is an alternate form of ኩ
        assert normalize("ኵ") == "ኩ"


# ── ው family and ጎ / ጐ ────────────────────────────────────────────────────────


class TestWuFamily:
    """ዉ (U+12C9) normalizes to ው (U+12CD), per spec section 2.5"""

    def test_wu_normalizes(self):
        assert normalize("ዉ") == "ው"

    def test_wu_in_word(self):
        assert normalize("ሰዉ") == "ሰው"

    def test_canonical_wu_unchanged(self):
        assert normalize("ሰው") == "ሰው"


class TestGoGwa:
    """
    ጐ (U+1310, gwa) normalizes to ጎ (U+130E, go).

    The spec table lists this pair in the opposite direction, but that maps
    the common form onto the rare one — in CC-100 ጎ outnumbers ጐ roughly
    16 to 1. Normalizing rare → common, consistent with every other rule.
    """

    def test_gwa_normalizes_to_go(self):
        assert normalize("ጐ") == "ጎ"

    def test_canonical_go_unchanged(self):
        assert normalize("ጎበዘ") == "ጎበዘ"


# ── ኀ / ኅ order disambiguation ────────────────────────────────────────────────


class TestXaSeriesOrders:
    """
    ኀ (U+1280) is the order-1 form and normalizes to ሀ.
    ኅ (U+1285) is the order-6 form and normalizes to ህ.

    These are distinct characters and must not be conflated — an earlier
    version mapped ኅ to ሀ, which left ኀ unnormalized entirely.
    """

    def test_order1_xa_to_ha(self):
        assert normalize("ኀ") == "ሀ"

    def test_order6_xe_to_h(self):
        assert normalize("ኅ") == "ህ"

    def test_both_in_one_string(self):
        assert normalize("ኀኅ") == "ሀህ"


# ── Multiple normalizations in one string ─────────────────────────────────────


class TestCombined:
    """Multiple normalization rules applied in a single string"""

    def test_multiple_families(self):
        result = normalize("ሠላም ሐሙስ ዓመት ፀሀይ")
        assert result == "ሰላም ሀሙስ አመት ጸሀይ"

    def test_real_sentence(self):
        # A realistic sentence containing multiple variant characters
        result = normalize("ሐሙስ ዓ.ም ሠርቶ ዐደገ")
        assert result == "ሀሙስ አ.ም ሰርቶ አደገ"


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Normalization must handle edge cases without breaking"""

    def test_empty_string(self):
        assert normalize("") == ""

    def test_latin_unchanged(self):
        assert normalize("Hello") == "Hello"

    def test_arabic_numerals_unchanged(self):
        assert normalize("2024") == "2024"

    def test_ethiopic_punctuation_unchanged(self):
        assert normalize("።") == "።"
        assert normalize("፣") == "፣"
        assert normalize("፤") == "፤"
        assert normalize("፡") == "፡"

    def test_mixed_script_latin_untouched(self):
        result = normalize("Daniel ሃይሌ")
        assert result == "Daniel ሀይሌ"

    def test_already_canonical_unchanged(self):
        # These words contain only canonical characters — normalize must not change them
        canonical = "ሰላም ቤት ሰው ልጅ ቀን እናት አባት"
        assert normalize(canonical) == canonical

    def test_newline_preserved(self):
        assert normalize("ሃይሌ\n") == "ሀይሌ\n"

    def test_whitespace_preserved(self):
        assert normalize("  ሃ  ") == "  ሀ  "

    def test_tab_preserved(self):
        assert normalize("\tሃ\t") == "\tሀ\t"


# ── normalize_lines ───────────────────────────────────────────────────────────


class TestNormalizeLines:
    """normalize_lines processes each line independently"""

    def test_empty_list(self):
        assert normalize_lines([]) == []

    def test_single_line(self):
        assert normalize_lines(["ሃይሌ"]) == ["ሀይሌ"]

    def test_multiple_lines(self):
        result = normalize_lines(["ሠላም", "ሐሙስ", "Hello"])
        assert result == ["ሰላም", "ሀሙስ", "Hello"]

    def test_output_length_matches_input(self):
        lines = ["ሠ", "ሐ", "ዓ", "ጸ", "ዉ"]
        assert len(normalize_lines(lines)) == len(lines)

    def test_empty_lines_preserved(self):
        # Empty lines in the middle should be preserved
        result = normalize_lines(["ሠ", "", "ሐ"])
        assert result == ["ሰ", "", "ሀ"]
