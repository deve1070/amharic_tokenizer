"""
Tests for Layer 6 — Sentence Segmentation

Covers:
  - Standard ። splitting
  - ፡፡ (double wordspace) normalized and split as ።
  - ? and ! as sentence boundaries
  - Combined punctuation (?!)
  - Single ፡ NOT treated as a boundary
  - Edge cases: empty input, no boundary, multiple consecutive boundaries

Run with: python -m pytest tests/test_segmenter.py -v
"""


from amharic_morph.segmenter import segment, segment_lines

# ── Basic ። splitting ─────────────────────────────────────────────────────────


class TestBasicPeriodSplit:
    """Standard ። sentence boundary splitting"""

    def test_two_sentences(self):
        result = segment("ሰላም ነው። እንዴት ነህ።")
        assert result == ["ሰላም ነው።", "እንዴት ነህ።"]

    def test_single_sentence(self):
        result = segment("ሰላም ነው።")
        assert result == ["ሰላም ነው።"]

    def test_three_sentences(self):
        result = segment("አንድ። ሁለት። ሶስት።")
        assert result == ["አንድ።", "ሁለት።", "ሶስት።"]

    def test_no_trailing_period(self):
        # Last sentence without terminal punctuation is still kept
        result = segment("ሰላም ነው። እንዴት ነህ")
        assert result == ["ሰላም ነው።", "እንዴት ነህ"]


# ── ፡፡ normalization and splitting ────────────────────────────────────────────


class TestDoubleWordspaceSplit:
    """
    ፡፡ (two U+1361 characters) is an alternate full-stop convention
    found in CC-100 and other corpora. It must be normalized to ።
    and split exactly like ።.
    """

    def test_double_wordspace_as_terminator(self):
        result = segment("ናት፡፡")
        assert result == ["ናት።"]

    def test_double_wordspace_two_sentences(self):
        result = segment("ናት፡፡ ሄደ፡፡")
        assert result == ["ናት።", "ሄደ።"]

    def test_mixed_period_and_double_wordspace(self):
        # Same text may mix ። and ፡፡ across different sentences
        result = segment("አንደኛው ነው። ሁለተኛው ናት፡፡")
        assert result == ["አንደኛው ነው።", "ሁለተኛው ናት።"]

    def test_real_corpus_example(self):
        # Drawn from the actual CC-100 sample
        text = "ይህን የሰሙት ጥዱም ብለው መለሱላቸው ይባላል፡፡"
        result = segment(text)
        assert result == ["ይህን የሰሙት ጥዱም ብለው መለሱላቸው ይባላል።"]


# ── Single ፡ is NOT a sentence boundary ────────────────────────────────────────


class TestSingleWordspaceNotBoundary:
    """
    Per Phase 0 policy: single ፡ is a clarification mark, not a
    sentence boundary or word separator. It must not cause a split.
    """

    def test_single_wordspace_no_split(self):
        result = segment("ሰላም፡ ነው።")
        # Only one sentence — the single ፡ does not split
        assert result == ["ሰላም፡ ነው።"]

    def test_single_wordspace_preserved_in_output(self):
        result = segment("ቃል፡ ፍቺ ይከተላል።")
        assert len(result) == 1
        assert "፡" in result[0]


# ── Question marks and exclamation marks ──────────────────────────────────────


class TestQuestionAndExclamation:
    """? and ! are valid sentence-final boundaries"""

    def test_question_mark(self):
        result = segment("እንዴት ነህ? ደህና ነኝ።")
        assert result == ["እንዴት ነህ?", "ደህና ነኝ።"]

    def test_exclamation_mark(self):
        result = segment("ና! ና!")
        assert result == ["ና!", "ና!"]

    def test_combined_question_exclamation(self):
        # "ምን ሆነ?!" should stay as ONE sentence, not split
        # between ? and !
        result = segment("ምን ሆነ?!")
        assert result == ["ምን ሆነ?!"]

    def test_mixed_boundary_types(self):
        result = segment("ሄደ። መጣ? ሮጠ!")
        assert result == ["ሄደ።", "መጣ?", "ሮጠ!"]


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Segmentation must handle degenerate input gracefully"""

    def test_empty_string(self):
        assert segment("") == []

    def test_whitespace_only(self):
        assert segment("   ") == []

    def test_no_boundary_at_all(self):
        # Text with no terminal punctuation is returned as one sentence
        result = segment("ምንም ምልክት የለውም")
        assert result == ["ምንም ምልክት የለውም"]

    def test_only_punctuation(self):
        # Edge case: text that is only punctuation
        result = segment("።")
        assert result == ["።"]

    def test_consecutive_boundaries_collapsed(self):
        # Multiple periods in a row should not create empty sentences
        result = segment("ሰላም።። እንዴት ነህ።")
        assert result == ["ሰላም።።", "እንዴት ነህ።"]

    def test_leading_whitespace_stripped(self):
        result = segment("   ሰላም ነው።")
        assert result == ["ሰላም ነው።"]

    def test_trailing_whitespace_stripped(self):
        result = segment("ሰላም ነው።   ")
        assert result == ["ሰላም ነው።"]

    def test_extra_spaces_between_sentences(self):
        result = segment("ሰላም ነው።     እንዴት ነህ።")
        assert result == ["ሰላም ነው።", "እንዴት ነህ።"]


# ── Mixed script content (Latin words present) ─────────────────────────────────


class TestMixedScriptContent:
    """Segmentation must work correctly even with Latin content present"""

    def test_latin_word_in_sentence(self):
        result = segment("ፕሬዚዳንቱ Biden ተናገሩ። ቀጣዩ ዜና ነው።")
        assert result == ["ፕሬዚዳንቱ Biden ተናገሩ።", "ቀጣዩ ዜና ነው።"]

    def test_real_corpus_mixed_example(self):
        # Drawn from the actual CC-100 sample with the embedded English title
        text = "የዳንኤል እይታዎች Daniel Kibret's views: ‹‹ታች በሌ›› ናት።"
        result = segment(text)
        assert len(result) == 1
        assert result[0].endswith("።")


# ── segment_lines ─────────────────────────────────────────────────────────────


class TestSegmentLines:
    """segment_lines flattens multiple input lines into one sentence list"""

    def test_empty_list(self):
        assert segment_lines([]) == []

    def test_single_line_multiple_sentences(self):
        result = segment_lines(["ሰላም ነው። እንዴት ነህ።"])
        assert result == ["ሰላም ነው።", "እንዴት ነህ።"]

    def test_multiple_lines(self):
        result = segment_lines(["አንድ።", "ሁለት። ሶስት።"])
        assert result == ["አንድ።", "ሁለት።", "ሶስት።"]

    def test_lines_with_no_sentences_skipped(self):
        # A blank line contributes nothing to the output
        result = segment_lines(["አንድ።", "", "ሁለት።"])
        assert result == ["አንድ።", "ሁለት።"]

    def test_output_is_flat_list(self):
        result = segment_lines(["አንድ። ሁለት።", "ሶስት። አራት።"])
        assert result == ["አንድ።", "ሁለት።", "ሶስት።", "አራት።"]
