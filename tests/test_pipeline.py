"""
Tests for the end-to-end corpus processing pipeline.

Covers:
  - process_line: normalize -> classify -> segment on a single line
  - process_lines: full corpus list processing including filtering
  - process_file: file-to-file pipeline run with statistics

Run with: python -m pytest tests/test_pipeline.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amharic_tokenizer.pipeline import process_file, process_line, process_lines

# ── process_line ──────────────────────────────────────────────────────────────


class TestProcessLine:
    """
    Single-line normalize -> punctuation -> classify -> segment chain.

    Layer 3 space-pads sentence-final punctuation, so output sentences
    end with " ።" rather than a ። glued to the last word. That is the
    point: it keeps ይባላል and ይባላል። from counting as two distinct types.
    """

    def test_simple_sentence(self):
        result = process_line("ሰላም ነው።")
        assert result == ["ሰላም ነው ።"]

    def test_double_wordspace_normalized_to_period(self):
        # ፡፡ becomes ። in Layer 3, before padding can split it
        result = process_line("ናት፡፡")
        assert result == ["ናት ።"]

    def test_character_normalization_applied(self):
        # ሐ should normalize to ሀ before segmentation
        result = process_line("ሐሙስ ነው።")
        assert result == ["ሀሙስ ነው ።"]

    def test_two_sentences_in_one_line(self):
        result = process_line("ሰላም ነው። እንዴት ነህ?")
        assert result == ["ሰላም ነው ።", "እንዴት ነህ ?"]

    def test_latin_period_becomes_sentence_boundary(self):
        # Without Layer 3 the Latin period never becomes ። and these two
        # sentences would run together as one.
        result = process_line("ኢትዮጵያ ሀገር ናት. ትልቅ ናት.")
        assert result == ["ኢትዮጵያ ሀገር ናት ።", "ትልቅ ናት ።"]

    def test_abbreviation_does_not_split_the_sentence(self):
        # Layer 0 runs first, so ዓ has already normalized to አ by the
        # time the abbreviation reaches Layer 3.
        result = process_line("በ2010 ዓ.ም በአዲስ አበባ ተጀመረ።")
        assert result == ["በ 2010 አ.ም በአዲስ አበባ ተጀመረ ።"]

    def test_latin_word_tagged_before_segmentation(self):
        result = process_line("ፕሬዚዳንቱ Biden ተናገሩ።")
        assert result == ["ፕሬዚዳንቱ <LATIN>Biden</LATIN> ተናገሩ ።"]

    def test_url_not_split_on_internal_period(self):
        # The period in "bbc.com" must NOT be treated as a sentence
        # boundary -- classification must wrap it as a URL before
        # segmentation ever sees it.
        result = process_line("ድረ ገጹን www.bbc.com ይጎብኙ።")
        assert len(result) == 1
        assert "<URL>www.bbc.com</URL>" in result[0]

    def test_empty_line(self):
        assert process_line("") == []

    def test_whitespace_only_line(self):
        assert process_line("   ") == []

    def test_real_corpus_sample(self):
        # Drawn directly from the actual CC-100 sample provided
        line = "ይህን የሰሙት ጥዱም ብለው መለሱላቸው ይባላል፡፡"
        result = process_line(line)
        assert result == ["ይህን የሰሙት ጥዱም ብለው መለሱላቸው ይባላል ።"]

    def test_normalization_and_classification_compose_correctly(self):
        # ሐ normalizes AND Latin gets tagged in the same line
        result = process_line("ሐኪሙ Doctor ነው።")
        assert result == ["ሀኪሙ <LATIN>Doctor</LATIN> ነው ።"]


# ── process_lines ─────────────────────────────────────────────────────────────


class TestProcessLines:
    """Full list processing including end-of-pipeline filtering"""

    def test_empty_list(self):
        assert process_lines([]) == []

    def test_filters_short_sentences(self):
        lines = [
            "ይህ ጥሩ ረዘም ያለ የአማርኛ ዓረፍተ ነገር ምሳሌ ነው።",  # passes min length
            "ሰላም ነው።",  # too short, dropped
        ]
        result = process_lines(lines)
        assert len(result) == 1
        assert result[0].startswith("ይህ ጥሩ")

    def test_deduplicates_across_lines(self):
        lines = [
            "ይህ ጥሩ ረዘም ያለ የአማርኛ ዓረፍተ ነገር ምሳሌ ነው።",
            "ይህ ጥሩ ረዘም ያለ የአማርኛ ዓረፍተ ነገር ምሳሌ ነው።",
        ]
        result = process_lines(lines)
        assert len(result) == 1

    def test_custom_min_tokens(self):
        lines = ["ሰላም ነው እንዴት ነህ።"]
        # 4 ethiopic tokens -- fails default min of 5
        assert process_lines(lines) == []
        # passes with a lower custom minimum
        result = process_lines(lines, min_tokens=3)
        assert len(result) == 1

    def test_multi_sentence_lines_flattened(self):
        lines = ["አንድ ሁለት ሶስት አራት አምስት። ስድስት ሰባት ስምንት ዘጠኝ አስር።"]
        result = process_lines(lines, min_tokens=3)
        assert len(result) == 2


# ── process_file ──────────────────────────────────────────────────────────────


class TestProcessFile:
    """File-to-file pipeline run"""

    def test_basic_run(self, tmp_path):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"

        input_file.write_text(
            "ይህ ጥሩ ረዘም ያለ የአማርኛ ዓረፍተ ነገር ምሳሌ ነው።\n"
            "ሰላም ነው።\n",  # too short, will be filtered
            encoding="utf-8",
        )

        stats = process_file(str(input_file), str(output_file))

        assert stats["input_lines"] == 2
        assert stats["output_sentences"] == 1

        output_content = output_file.read_text(encoding="utf-8")
        assert "ይህ ጥሩ" in output_content
        assert output_content.count("\n") == 1

    def test_stats_keys_present(self, tmp_path):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("ሰላም ነው።\n", encoding="utf-8")

        stats = process_file(str(input_file), str(output_file))

        assert "input_lines" in stats
        assert "raw_sentences" in stats
        assert "output_sentences" in stats

    def test_raises_on_missing_input(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        output_file = tmp_path / "output.txt"
        with pytest.raises(FileNotFoundError):
            process_file(str(missing), str(output_file))

    def test_empty_input_file(self, tmp_path):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("", encoding="utf-8")

        stats = process_file(str(input_file), str(output_file))

        assert stats["input_lines"] == 0
        assert stats["output_sentences"] == 0
        assert output_file.read_text(encoding="utf-8") == ""
