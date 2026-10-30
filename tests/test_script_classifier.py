"""
Tests for Layer 4 — Script Classification

Covers:
  - Pure Ethiopic tokens (unchanged)
  - Pure Latin tokens (wrapped in <LATIN>)
  - Pure numeric tokens (unchanged)
  - URL detection and wrapping
  - Mixed Ethiopic+Latin split at script boundary
  - Mixed numeric+Ethiopic split at boundary
  - Latin+numeric merged into single <LATIN> unit
  - Hashtags
  - Sentence-level classification via classify_text

Run with: python -m pytest tests/test_script_classifier.py -v
"""


from amharic_morph.script_classifier import classify_text, classify_token

# ── Pure Ethiopic ─────────────────────────────────────────────────────────────


class TestPureEthiopic:
    """All-Ethiopic tokens must be returned unchanged"""

    def test_simple_word(self):
        assert classify_token("ሰላም") == "ሰላም"

    def test_longer_word(self):
        assert classify_token("ኢትዮጵያ") == "ኢትዮጵያ"

    def test_word_with_punctuation_attached(self):
        # Punctuation should already be space-padded by Layer 3,
        # but verify a bare Ethiopic token with no punctuation works
        assert classify_token("መጣ") == "መጣ"


# ── Pure Latin ────────────────────────────────────────────────────────────────


class TestPureLatin:
    """All-Latin tokens must be wrapped in <LATIN> tags"""

    def test_simple_english_word(self):
        assert classify_token("Hello") == "<LATIN>Hello</LATIN>"

    def test_proper_noun(self):
        assert classify_token("Ethiopia") == "<LATIN>Ethiopia</LATIN>"

    def test_acronym(self):
        assert classify_token("BBC") == "<LATIN>BBC</LATIN>"

    def test_lowercase_word(self):
        assert classify_token("press") == "<LATIN>press</LATIN>"


# ── Pure numeric ──────────────────────────────────────────────────────────────


class TestPureNumeric:
    """All-numeric tokens must be returned unchanged (Arabic numerals kept as-is)"""

    def test_year(self):
        assert classify_token("2024") == "2024"

    def test_large_number(self):
        assert classify_token("165") == "165"

    def test_single_digit(self):
        assert classify_token("5") == "5"


# ── URL detection ─────────────────────────────────────────────────────────────


class TestUrlDetection:
    """URLs must be wrapped in <URL> tags"""

    def test_https_url(self):
        result = classify_token("https://example.com")
        assert result == "<URL>https://example.com</URL>"

    def test_http_url(self):
        result = classify_token("http://example.com")
        assert result == "<URL>http://example.com</URL>"

    def test_www_url(self):
        result = classify_token("www.bbc.com")
        assert result == "<URL>www.bbc.com</URL>"

    def test_bare_domain(self):
        result = classify_token("ethiopianreporter.com")
        assert result == "<URL>ethiopianreporter.com</URL>"

    def test_domain_with_path(self):
        result = classify_token("example.com/amharic/news")
        assert result == "<URL>example.com/amharic/news</URL>"

    def test_et_domain(self):
        result = classify_token("ebc.et")
        assert result == "<URL>ebc.et</URL>"


# ── Mixed Ethiopic + Latin split ───────────────────────────────────────────────


class TestMixedEthiopicLatinSplit:
    """
    Mixed-script tokens must split at the script boundary.
    Prefix morphemes (Ethiopic) separate from foreign words (Latin).
    """

    def test_prefix_plus_latin_word(self):
        result = classify_token("የEthiopia")
        assert result == "የ <LATIN>Ethiopia</LATIN>"

    def test_latin_then_ethiopic(self):
        result = classify_token("ABCኢትዮጵያ")
        assert result == "<LATIN>ABC</LATIN> ኢትዮጵያ"

    def test_suffix_after_latin(self):
        # Ethiopic suffix attached after a Latin word
        result = classify_token("Ethiopiaን")
        assert result == "<LATIN>Ethiopia</LATIN> ን"


# ── Mixed numeric + Ethiopic split ─────────────────────────────────────────────


class TestMixedNumericEthiopicSplit:
    """Numeric + Ethiopic suffix must split at the boundary"""

    def test_ordinal_suffix(self):
        # 100ኛ means "100th"
        result = classify_token("100ኛ")
        assert result == "100 ኛ"

    def test_number_then_ethiopic_word(self):
        result = classify_token("2024ዓም")
        assert result == "2024 ዓም"


# ── Latin + numeric merge ──────────────────────────────────────────────────────


class TestLatinNumericMerge:
    """Latin+numeric combinations like COVID-19 stay as one <LATIN> unit"""

    def test_covid_19(self):
        result = classify_token("COVID-19")
        assert result == "<LATIN>COVID-19</LATIN>"

    def test_g7(self):
        result = classify_token("G7")
        assert result == "<LATIN>G7</LATIN>"

    def test_covid_then_ethiopic(self):
        # The Latin+numeric unit still splits correctly from
        # a following Ethiopic span
        result = classify_token("COVID-19ወረርሽኝ")
        assert result == "<LATIN>COVID-19</LATIN> ወረርሽኝ"


# ── Hashtags ──────────────────────────────────────────────────────────────────


class TestHashtags:
    """Hashtags: # kept separate, remainder classified normally"""

    def test_amharic_hashtag(self):
        result = classify_token("#ኢትዮጵያ")
        assert result == "# ኢትዮጵያ"

    def test_latin_hashtag(self):
        result = classify_token("#Ethiopia")
        assert result == "# <LATIN>Ethiopia</LATIN>"

    def test_mixed_hashtag(self):
        result = classify_token("#BBCAmharic")
        assert result == "# <LATIN>BBCAmharic</LATIN>"


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Classification must handle degenerate input gracefully"""

    def test_empty_token(self):
        assert classify_token("") == ""

    def test_single_ethiopic_char(self):
        assert classify_token("ሀ") == "ሀ"

    def test_single_latin_char(self):
        assert classify_token("A") == "<LATIN>A</LATIN>"

    def test_punctuation_only_token(self):
        # A bare punctuation token (already space-padded by Layer 3)
        # has no Ethiopic/Latin/numeric content — passed through.
        assert classify_token("።") == "።"
        assert classify_token("?") == "?"


# ── Sentence-level classification ──────────────────────────────────────────────


class TestClassifyText:
    """classify_text applies classification across an entire sentence"""

    def test_empty_string(self):
        assert classify_text("") == ""

    def test_pure_ethiopic_sentence(self):
        result = classify_text("ሰላም ነው እንዴት ነህ")
        assert result == "ሰላም ነው እንዴት ነህ"

    def test_sentence_with_one_latin_word(self):
        result = classify_text("ፕሬዚዳንቱ Biden ተናገሩ")
        assert result == "ፕሬዚዳንቱ <LATIN>Biden</LATIN> ተናገሩ"

    def test_sentence_with_number(self):
        result = classify_text("ከ 165 ኪሎ ሜትር በላይ")
        assert result == "ከ 165 ኪሎ ሜትር በላይ"

    def test_real_corpus_example(self):
        # Drawn from the actual CC-100 sample
        result = classify_text("ከአዲስ አበባ 165 ኪሎ ሜትር በምትርቅ ከተማ")
        assert result == "ከአዲስ አበባ 165 ኪሎ ሜትር በምትርቅ ከተማ"

    def test_sentence_with_mixed_token(self):
        result = classify_text("የ Ethiopia ህዝብ")
        assert result == "የ <LATIN>Ethiopia</LATIN> ህዝብ"

    def test_multiple_spaces_collapsed_to_single_tokens(self):
        # classify_text splits on single space; empty tokens from
        # double spaces are dropped
        result = classify_text("ሰላም  ነው")
        assert result == "ሰላም ነው"


# ── Abbreviation preservation ────────────────────────────────────────────────


class TestAbbreviationsStayAtomic:
    """
    Amharic abbreviations contain internal periods, which are script
    boundaries. Without an explicit guard, span splitting shreds ዓ.ም into
    "ዓ . ም" and the abbreviation is lost before it ever reaches the
    tokenizer levels.
    """

    def test_year_abbreviation(self):
        assert classify_token("ዓ.ም") == "ዓ.ም"

    def test_year_abbreviation_with_trailing_period(self):
        assert classify_token("ዓ.ም.") == "ዓ.ም."

    def test_gregorian_calendar_abbreviation(self):
        assert classify_token("እ.ኤ.አ") == "እ.ኤ.አ"

    def test_company_abbreviation(self):
        assert classify_token("ኃ.የተ.የግ.ማ") == "ኃ.የተ.የግ.ማ"

    def test_broadcaster_abbreviation(self):
        assert classify_token("ኤፍ.ቢ.ሲ") == "ኤፍ.ቢ.ሲ"

    def test_unit_abbreviation(self):
        assert classify_token("ኪ.ሜ") == "ኪ.ሜ"

    def test_abbreviation_in_sentence(self):
        # በ2010 is mixed Ethiopic+Numeric and splits at the script
        # boundary per spec; the abbreviation beside it stays whole.
        result = classify_text("በ2010 ዓ.ም ተጀመረ")
        assert result == "በ 2010 ዓ.ም ተጀመረ"

    def test_decimal_number_still_intact(self):
        assert classify_token("3.5") == "3.5"

    def test_ordinary_word_still_classified(self):
        # Not an abbreviation, so normal handling applies
        assert classify_token("ሰላም") == "ሰላም"

    def test_latin_still_tagged(self):
        assert classify_token("Biden") == "<LATIN>Biden</LATIN>"


class TestSlashAndHyphen:
    def test_slash_abbreviation_stays_whole(self):
        # Splitting it left ዶ and ር as "words" (27,277 times in the corpus).
        assert classify_text("ዶ/ር አብይ ጽ/ቤት") == "ዶ/ር አብይ ጽ/ቤት"

    def test_hyphen_between_ethiopic_words_is_its_own_token(self):
        # A compound written with a hyphen then analyses as the spaced one:
        # ቤተ - ክርስቲያን = ቤት + ኧ | - | ክርስቲያን. Ruled 2026-10-30.
        assert classify_text("ቤተ-ክርስቲያን") == "ቤተ - ክርስቲያን"
        assert classify_text("ጸረ-ሙስና") == "ጸረ - ሙስና"
