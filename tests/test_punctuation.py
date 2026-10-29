"""
Tests for Layer 3 — Punctuation and Whitespace Normalization.

The Latin period cases follow the four non-sentence uses of "." in
Amharic text: numbers, abbreviations, ellipsis, and URLs.
"""


from amharic_morph.punctuation import (
    convert_latin_periods,
    normalize_punctuation,
    normalize_punctuation_lines,
    normalize_whitespace,
    pad_punctuation,
)


# ── Doubled wordspace ─────────────────────────────────────────────────────────


class TestDoubleWordspace:
    """፡፡ is the old-style full stop and converts to ።"""

    def test_double_wordspace_becomes_full_stop(self):
        assert normalize_punctuation("ሰላም ነው፡፡") == "ሰላም ነው ።"

    def test_double_wordspace_mid_text(self):
        result = normalize_punctuation("ሰላም ነው፡፡ እንዴት ነህ፡፡")
        assert result == "ሰላም ነው ። እንዴት ነህ ።"

    def test_double_wordspace_survives_padding(self):
        # Regression: padding ፡ before converting ፡፡ would produce "፡ ፡"
        # and destroy the sentence boundary.
        assert "፡ ፡" not in normalize_punctuation("ነው፡፡")

    def test_single_wordspace_is_not_a_full_stop(self):
        assert normalize_punctuation("ቤት፡ ሰው") == "ቤት ፡ ሰው"


# ── ፡ as punctuation ──────────────────────────────────────────────────────────


class TestWordspacePadding:
    """
    Modern Amharic uses the space as word separator and ፡ as punctuation.
    Padding handles both the modern and the residual old-style usage.
    """

    def test_attached_wordspace_is_detached(self):
        assert normalize_punctuation("ቤት፡") == "ቤት ፡"

    def test_already_isolated_wordspace_unchanged(self):
        assert normalize_punctuation("ቤት ፡ ሰው") == "ቤት ፡ ሰው"

    def test_old_style_separator_splits_words(self):
        # Where ፡ genuinely separates two words, splitting is correct.
        assert normalize_punctuation("ቤት፡ሰው") == "ቤት ፡ ሰው"


# ── Latin period: sentence ends ───────────────────────────────────────────────


class TestLatinPeriodSentenceEnd:
    """A period after an Ethiopic character normally ends a sentence"""

    def test_simple_sentence_end(self):
        assert convert_latin_periods("ኢትዮጵያ ሀገር ናት.") == "ኢትዮጵያ ሀገር ናት።"

    def test_two_sentences(self):
        result = convert_latin_periods("ሰላም ነው. እንዴት ነህ.")
        assert result == "ሰላም ነው። እንዴት ነህ።"

    def test_period_after_closing_paren(self):
        assert convert_latin_periods("(ሰላም).") == "(ሰላም)።"

    def test_glued_sentences_are_split(self):
        # No space after the period; still a sentence boundary.
        assert convert_latin_periods("ነው.እንዴት") == "ነው።እንዴት"

    def test_full_layer_separates_glued_sentences(self):
        assert normalize_punctuation("ነው.እንዴት") == "ነው ። እንዴት"

    def test_period_after_latin_not_converted(self):
        # Spec restricts conversion to Ethiopic or closing punctuation.
        assert convert_latin_periods("Hello world.") == "Hello world."

    def test_list_numbering_not_converted(self):
        assert convert_latin_periods("1. ሰላም") == "1. ሰላም"


# ── Latin period: numbers ─────────────────────────────────────────────────────


class TestLatinPeriodNumbers:
    """Amharic uses Arabic numerals, so decimal points must survive"""

    def test_decimal_preserved(self):
        assert convert_latin_periods("ከ3.5 ቢሊዮን") == "ከ3.5 ቢሊዮን"

    def test_decimal_alone(self):
        assert convert_latin_periods("4.0") == "4.0"

    def test_decimal_survives_full_layer(self):
        assert normalize_punctuation("ከ3.5 ቢሊዮን በላይ") == "ከ3.5 ቢሊዮን በላይ"

    def test_decimal_and_sentence_end_together(self):
        assert convert_latin_periods("ከ3.5 ቢሊዮን በላይ ነው.") == "ከ3.5 ቢሊዮን በላይ ነው።"


# ── Latin period: abbreviations ───────────────────────────────────────────────


class TestLatinPeriodAbbreviations:
    """
    Abbreviations such as አ.ል.ማ (አማራ ልማት ማህበር) keep their periods.

    This is the case the spec's "preceded by Ethiopic" rule gets wrong
    on its own, since abbreviations also have Ethiopic before the period.
    """

    def test_year_abbreviation(self):
        assert convert_latin_periods("በ2010 ዓ.ም ተጀመረ") == "በ2010 ዓ.ም ተጀመረ"

    def test_year_abbreviation_with_trailing_period(self):
        assert convert_latin_periods("በ2010 ዓ.ም. ተጀመረ") == "በ2010 ዓ.ም. ተጀመረ"

    def test_gregorian_calendar_abbreviation(self):
        assert convert_latin_periods("እ.ኤ.አ 2020") == "እ.ኤ.አ 2020"

    def test_three_letter_abbreviation(self):
        assert convert_latin_periods("አ.ል.ማ") == "አ.ል.ማ"

    def test_two_character_segments(self):
        assert convert_latin_periods("ኃ.የተ.የግ.ማ") == "ኃ.የተ.የግ.ማ"

    def test_abbreviation_in_parentheses(self):
        assert convert_latin_periods("(ኤፍ.ቢ.ሲ)") == "(ኤፍ.ቢ.ሲ)"

    def test_unit_abbreviation(self):
        assert convert_latin_periods("50 ኪ.ሜ") == "50 ኪ.ሜ"

    def test_single_character_abbreviation(self):
        assert convert_latin_periods("ቁ. 5") == "ቁ. 5"

    def test_known_multi_character_abbreviation(self):
        assert convert_latin_periods("ወዘተ.") == "ወዘተ."

    def test_abbreviation_survives_full_layer(self):
        assert normalize_punctuation("በ2010 ዓ.ም ተጀመረ") == "በ2010 ዓ.ም ተጀመረ"

    def test_ordinary_word_is_not_an_abbreviation(self):
        # ነው is two characters but has no internal period, so it is a
        # sentence end rather than an abbreviation.
        assert convert_latin_periods("ነው.") == "ነው።"

    def test_long_word_is_not_an_abbreviation(self):
        assert convert_latin_periods("ይችላሉ.") == "ይችላሉ።"


# ── Latin period: ellipsis ────────────────────────────────────────────────────


class TestLatinPeriodEllipsis:
    """Three dots mark continuation, not a sentence end"""

    def test_ellipsis_preserved(self):
        assert convert_latin_periods("ሰላም...") == "ሰላም..."

    def test_two_dots_preserved(self):
        assert convert_latin_periods("ሰላም..") == "ሰላም.."

    def test_unicode_ellipsis_preserved(self):
        assert convert_latin_periods("ሰላም…") == "ሰላም…"

    def test_ellipsis_survives_full_layer(self):
        assert normalize_punctuation("ሰላም...") == "ሰላም..."


# ── Latin period: URLs ────────────────────────────────────────────────────────


class TestLatinPeriodUrls:
    """Periods inside URLs are used as in English and stay untouched"""

    def test_http_url(self):
        text = "https://example.com/page"
        assert convert_latin_periods(text) == text

    def test_www_url(self):
        assert convert_latin_periods("www.example.com") == "www.example.com"

    def test_bare_domain(self):
        assert convert_latin_periods("ecadforum.com") == "ecadforum.com"

    def test_url_with_sentence_after(self):
        result = convert_latin_periods("ይህ www.example.com ነው.")
        assert result == "ይህ www.example.com ነው።"

    def test_url_survives_full_layer(self):
        text = "https://example.com/page"
        assert normalize_punctuation(text) == text


# ── Padding ───────────────────────────────────────────────────────────────────


class TestPadPunctuation:
    """Each of ። ፣ ፤ ፥ ፦ ፡ ? ! # gets a space on both sides"""

    def test_full_stop(self):
        assert normalize_punctuation("ሰላም ነው።") == "ሰላም ነው ።"

    def test_comma(self):
        assert normalize_punctuation("ቤት፣ ሰው") == "ቤት ፣ ሰው"

    def test_semicolon(self):
        assert normalize_punctuation("ቤት፤ ሰው") == "ቤት ፤ ሰው"

    def test_colon(self):
        assert normalize_punctuation("ቤት፥ ሰው") == "ቤት ፥ ሰው"

    def test_preface_colon(self):
        assert normalize_punctuation("ቤት፦ ሰው") == "ቤት ፦ ሰው"

    def test_question_mark(self):
        assert normalize_punctuation("እንዴት ነህ?") == "እንዴት ነህ ?"

    def test_exclamation_mark(self):
        assert normalize_punctuation("ሰላም!") == "ሰላም !"

    def test_hash(self):
        assert normalize_punctuation("#ኢትዮጵያ") == "# ኢትዮጵያ"

    def test_pad_does_not_collapse_whitespace(self):
        # pad_punctuation leaves cleanup to normalize_whitespace
        assert pad_punctuation("ቤት፣") == "ቤት ፣ "

    def test_multiple_punctuation(self):
        assert normalize_punctuation("ቤት፣ ሰው፤ ውሻ።") == "ቤት ፣ ሰው ፤ ውሻ ።"


# ── Whitespace ────────────────────────────────────────────────────────────────


class TestNormalizeWhitespace:
    """Collapse runs, normalize line endings, trim"""

    def test_collapse_multiple_spaces(self):
        assert normalize_whitespace("ሰላም    ነው") == "ሰላም ነው"

    def test_collapse_tabs(self):
        assert normalize_whitespace("ሰላም\t\tነው") == "ሰላም ነው"

    def test_strip_leading_and_trailing(self):
        assert normalize_whitespace("   ሰላም ነው   ") == "ሰላም ነው"

    def test_crlf_normalized(self):
        assert normalize_whitespace("ሰላም\r\nነው") == "ሰላም\nነው"

    def test_cr_normalized(self):
        assert normalize_whitespace("ሰላም\rነው") == "ሰላም\nነው"

    def test_spaces_around_newline_removed(self):
        assert normalize_whitespace("ሰላም  \n  ነው") == "ሰላም\nነው"

    def test_newlines_preserved(self):
        assert normalize_whitespace("ሰላም\nነው") == "ሰላም\nነው"


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Empty and degenerate input"""

    def test_empty_string(self):
        assert normalize_punctuation("") == ""

    def test_whitespace_only(self):
        assert normalize_punctuation("   ") == ""

    def test_no_punctuation(self):
        assert normalize_punctuation("ሰላም ነው") == "ሰላም ነው"

    def test_punctuation_only(self):
        assert normalize_punctuation("።") == "።"

    def test_idempotent(self):
        text = "ኢትዮጵያ ሀገር ናት. በ2010 ዓ.ም ተጀመረ፡፡"
        once = normalize_punctuation(text)
        assert normalize_punctuation(once) == once

    def test_latin_text_untouched(self):
        assert normalize_punctuation("Hello world") == "Hello world"


# ── Batch helper ──────────────────────────────────────────────────────────────


class TestNormalizePunctuationLines:
    """normalize_punctuation_lines processes each line independently"""

    def test_empty_list(self):
        assert normalize_punctuation_lines([]) == []

    def test_multiple_lines(self):
        result = normalize_punctuation_lines(["ሰላም ነው።", "ቤት፣ ሰው"])
        assert result == ["ሰላም ነው ።", "ቤት ፣ ሰው"]

    def test_output_length_matches_input(self):
        lines = ["ሰላም ነው።", "", "ቤት፣ ሰው"]
        assert len(normalize_punctuation_lines(lines)) == len(lines)


# ── Real corpus shapes ────────────────────────────────────────────────────────


class TestRealCorpusPatterns:
    """Patterns taken from the actual CC-100 Amharic text"""

    def test_year_with_following_clause(self):
        result = normalize_punctuation("በ2010 ዓ.ም. በአዲስ አበባ ተጀመረ፡፡")
        assert result == "በ2010 ዓ.ም. በአዲስ አበባ ተጀመረ ።"

    def test_company_abbreviation(self):
        result = normalize_punctuation("ኃ.የተ.የግ.ማ ነው፡፡")
        assert result == "ኃ.የተ.የግ.ማ ነው ።"

    def test_mixed_punctuation_sentence(self):
        result = normalize_punctuation("ቤት፣ ሰው፣ ውሻ ወዘተ. አሉ፡፡")
        assert result == "ቤት ፣ ሰው ፣ ውሻ ወዘተ. አሉ ።"
