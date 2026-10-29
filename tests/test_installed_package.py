"""
Tests that the built wheel works once installed.

The other tests import the package from the checkout, so a wheel missing
lexicon.json would pass all of them and fail for every user at
AmharicTokenizer(). These build the wheel, install it into a fresh
virtual environment, and run it from outside the repository.

They are slow and need the build tools (pip install -e ".[dev]"), so the
default run leaves them out. Run them with: pytest -m packaging
"""

import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

ROOT = Path(__file__).parent.parent

# Working files kept beside lexicon.json in the checkout. None of them
# belongs in the distribution.
WORKING_FILES = ("candidates.json", "category_review.tsv", "root_review.tsv", "stem_review.tsv")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(ROOT)],
        check=True, capture_output=True,
    )
    return next(out.glob("amharic_morph-*.whl"))


@pytest.fixture(scope="module")
def installed_with_bpe(wheel, tmp_path_factory):
    # The [bpe] extra installs sentencepiece from the package index.
    env = tmp_path_factory.mktemp("env_bpe")
    venv.create(env, with_pip=True)
    python = env / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", f"{wheel}[bpe]"],
        check=True, capture_output=True,
    )
    return python


@pytest.fixture(scope="module")
def installed_python(wheel, tmp_path_factory):
    env = tmp_path_factory.mktemp("env")
    venv.create(env, with_pip=True)
    python = env / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        check=True, capture_output=True,
    )
    return python


def run_installed(python, code, tmp_path):
    # From a directory outside the repository, so the checkout cannot be
    # imported in place of the installed copy. The snippet prints Amharic,
    # and on Windows a pipe defaults to cp1252, so its output is set to
    # UTF-8; the library itself prints nothing.
    result = subprocess.run(
        [str(python), "-c", code], cwd=tmp_path, check=True, capture_output=True,
        text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.stdout


class TestWheelContents:
    def test_carries_the_lexicon(self, wheel):
        assert "amharic_morph/lexicon/lexicon.json" in zipfile.ZipFile(wheel).namelist()

    def test_carries_the_subword_model(self, wheel):
        assert "amharic_morph/models/morph_16k.model" in zipfile.ZipFile(wheel).namelist()

    def test_leaves_out_the_working_files(self, wheel):
        names = zipfile.ZipFile(wheel).namelist()
        assert not [n for n in names if n.rsplit("/", 1)[-1] in WORKING_FILES]


class TestInstalledTokenizer:
    def test_imports_the_installed_copy(self, installed_python, tmp_path):
        path = run_installed(installed_python, "import amharic_morph; print(amharic_morph.__file__)", tmp_path)
        assert str(ROOT) not in path

    def test_four_levels(self, installed_python, tmp_path):
        code = (
            "import json\n"
            "from amharic_morph import AmharicTokenizer\n"
            "t = AmharicTokenizer()\n"
            "print(json.dumps({lv: t.tokenize('ቤቶች', level=lv)"
            " for lv in ('morpheme', 'fidel', 'sound', 'root')}, ensure_ascii=False))\n"
        )
        levels = json.loads(run_installed(installed_python, code, tmp_path))
        assert levels == {
            "morpheme": ["ቤት", "ኦች"],
            "fidel": ["ቤ", "ቶ", "ች"],
            "sound": ["ብ", "ኤ", "ት", "ኦ", "ች"],
            "root": ["ብ", "ት"],
        }

    def test_command_is_installed(self, installed_python, tmp_path):
        command = installed_python.parent / ("amharic-morph.exe" if sys.platform == "win32" else "amharic-morph")
        result = subprocess.run(
            [str(command), "ቤቶች"], cwd=tmp_path, check=True, capture_output=True,
            text=True, encoding="utf-8",
        )
        assert "ቤት + ኦች" in result.stdout

    def test_reviewed_verb_stems_are_shipped(self, installed_python, tmp_path):
        code = (
            "from amharic_morph import AmharicTokenizer\n"
            "print(' '.join(AmharicTokenizer().tokenize('ይሆናል')))\n"
        )
        assert run_installed(installed_python, code, tmp_path).split() == ["ይ", "ሆነ", "ኣል"]


class TestInstalledSubwordLevel:
    def test_bpe_level_with_the_shipped_model(self, installed_with_bpe, tmp_path):
        code = (
            "from amharic_morph import AmharicTokenizer\n"
            "t = AmharicTokenizer()\n"
            "print(' '.join(t.tokenize('ቤቶቻችንን', level='bpe')))\n"
            "print(t.decode(t.encode('ቤቶቻችንን የሚገባቸው ሰዎች ።')))\n"
        )
        pieces, decoded = run_installed(installed_with_bpe, code, tmp_path).splitlines()
        assert pieces.split() == ["▁ቤት", "ኦች", "ኣችን", "ን"]
        assert decoded == "ቤቶቻችንን የሚገባቸው ሰዎች ።"

    def test_without_the_extra_the_error_names_it(self, installed_python, tmp_path):
        code = (
            "from amharic_morph import AmharicTokenizer\n"
            "try:\n"
            "    AmharicTokenizer().tokenize('ቤቶች', level='bpe')\n"
            "except ImportError as e:\n"
            "    print(e)\n"
        )
        assert "amharic-morph[bpe]" in run_installed(installed_python, code, tmp_path)
