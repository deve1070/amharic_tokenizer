"""
Tests for `python -m amharic_morph`, the command for trying the
tokenizer on text before release.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(*args, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "amharic_morph", *args],
        input=stdin, capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
    )


class TestCommand:
    def test_text_argument(self):
        result = run("ቤቶች")
        assert result.returncode == 0
        assert "ቤት + ኦች" in result.stdout

    def test_every_level_is_shown(self):
        out = run("ቤቶች").stdout
        for label in ("morphemes", "fidel", "sound", "root"):
            assert label in out

    def test_readings_are_shown(self):
        assert "plural" in run("ቤቶች").stdout

    def test_root_is_shown(self):
        assert "ብ-ት" in run("ቤቶች").stdout

    def test_unknown_word_is_marked(self):
        assert "unknown" in run("ኮምፒውተር").stdout

    def test_reads_standard_input(self):
        result = run(stdin="ቤቶች\n")
        assert "ቤት + ኦች" in result.stdout

    def test_punctuation_is_listed_without_analysis(self):
        out = run("ቤቶች ናቸው።").stdout
        assert "።" in out

    def test_summary_line(self):
        assert "words analysed" in run("ቤቶች ኮምፒውተር").stdout
