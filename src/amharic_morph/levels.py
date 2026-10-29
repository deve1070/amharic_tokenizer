"""
The representation levels.

Level 3 (Fidel) and Level 2 (Sound) live here, along with the fidel
arithmetic they share and the consonant skeleton that proposes Level 1
roots. Level 1 itself is a reviewed field on each lexicon entry
(lexicon_builder.entry_root), because a root cannot be read off the
spelling: መሆን has the skeleton ም-ህ-ን and the root ህ-ን. Level 4 (Morpheme) is
lexicon-anchored and lives with the segmenter once a validated lexicon
exists — nothing here depends on it.

── Computational order ──────────────────────────────────────────────────

The spec computes 4 -> 3 -> 2 -> 1. Level 1 takes the root of the
INDEPENDENT morpheme only, so it needs morpheme boundaries first: run it
on the raw surface form ቤቶች and you get ብ-ት-ች, with the suffix consonant
polluting the root. The functions here are therefore written to be
applied to a single morpheme, and consonant_skeleton() documents that contract.

── Fidel arithmetic ─────────────────────────────────────────────────────

Ethiopic syllable families are 8-aligned: seven vowel orders plus a
labialized form in the eighth slot. So order = codepoint % 8 + 1, and
the bare consonant (order 6) is always family_base + 5.

Labiovelars are the exception. ቋ, ኳ, ጓ, ኋ and their relatives sit
in blocks of their own, each placed right after its plain family (ቈ
after ቀ, ኰ after ከ). The arithmetic still gives the right vowel order
inside such a block, but it gives the block's own bare form, ቍ, never
the plain ቅ. Each block also fills only five of its eight slots (orders
1, 3, 4, 5, 6). They are therefore decomposed through a table, like
order 8:

    ቈ  ቅ + ወ     ቊ  ቅ + ዊ     ቋ  ቅ + ዋ     ቌ  ቅ + ዌ     ቍ  ቅ + ው

ቋ is ቅ + ዋ exactly as ሏ is ል + ዋ, and at Level 1 the glide is
dropped from both.

── Level 2 decomposition ────────────────────────────────────────────────

Each fidel splits into (bare consonant, vowel):

    order 1   ሀ    ህ + ኧ
    order 2   ሁ    ህ + ኡ
    order 3   ሂ    ህ + ኢ
    order 4   ሃ    ህ + ኣ
    order 5   ሄ    ህ + ኤ
    order 6   ህ    ህ          bare, no vowel
    order 7   ሆ    ህ + ኦ

Order 1 carries ኧ, not አ. This matters: it reserves አ for the glottal
CONSONANT, so an አ-family letter appearing in a decomposition can only
have come from the surface word. That is what makes the Level 1 rule
below decidable. (Spec section 2.1 writes order 1 as "ህ + አ", which
would make the two indistinguishable.)

── The consonant skeleton ───────────────────────────────────────────────

Two rules, applied together:

1. Provenance. Only characters present in the surface morpheme can enter
   the root. Vowels produced by decomposition never do — the ኤ and ኦ in
   ብኤትኦች came from ቤ and ቶ, so ቤት gives ብ-ት.

2. Position. Among surface አ-family letters, a word-initial one is a
   glottal consonant and is kept; a word-internal one is a template
   vowel and is stripped. Amharic vowels sit at the front of a word far
   more often than inside it, which is what makes this reliable.

    አለ    -> እ-ል     initial አ is a consonant
    ጣእም   -> ጥ-ም     internal እ is a template vowel
    ቤቶች   -> ብ-ት     ኤ and ኦ never existed in the surface form
"""

import unicodedata
from functools import lru_cache

_FAMILY_SIZE = 8
_ORDER6 = 6

_ETHIOPIC_FIRST = "ሀ"
_ETHIOPIC_LAST = "፿"
# The syllables proper: the regular families, ሀ to ፗ. After them come
# ፘ ፙ ፚ, which belong to no family, then combining marks, punctuation
# (። ፣) and numerals (፩ ፲). None of those has a vowel order.
_SYLLABLE_LAST = "ፗ"

# Vowel letter contributed by each order at Level 2. Order 6 is the bare
# consonant and contributes nothing.
_ORDER_VOWELS = {
    1: "ኧ",
    2: "ኡ",
    3: "ኢ",
    4: "ኣ",
    5: "ኤ",
    6: "",
    7: "ኦ",
    # Labialized forms (ሏ ሟ ቷ …) are order 8. Spec section 2.4.3 reserves
    # an extended designation for them; /wa/ is written ዋ, which is also
    # what the Layer 0 normalizer uses when it unifies ሉዋ into ሏ.
    8: "ዋ",
}

# Labiovelar blocks, keyed by codepoint base, mapped to the plain
# consonant they labialize. The ኀ block maps to ህ rather than ኅ because
# Layer 0 merges the ኀ family into ሀ, and a decomposition must not bring
# back a letter the normalizer removed.
_LABIOVELAR_CONSONANTS = {
    0x1248: "ቅ",  # ቈ ቊ ቋ ቌ ቍ
    0x1288: "ህ",  # ኈ ኊ ኋ ኌ ኍ
    0x12B0: "ክ",  # ኰ ኲ ኳ ኴ ኵ
    0x12C0: "ኽ",  # ዀ ዂ ዃ ዄ ዅ
    0x1310: "ግ",  # ጐ ጒ ጓ ጔ ጕ
}

# The w-syllable each labiovelar contributes at Level 2, by vowel order.
# Orders 2, 7 and 8 have no labiovelar letter.
_LABIOVELAR_GLIDES = {1: "ወ", 3: "ዊ", 4: "ዋ", 5: "ዌ", 6: "ው"}

# The አ family (U+12A0-U+12A7) carries the glottal stop.
_GLOTTAL_BASE = ord("አ") - ord("አ") % _FAMILY_SIZE


# Cached: the answer depends on the character alone, and the search asks
# about the same few hundred characters millions of times.
@lru_cache(maxsize=None)
def is_ethiopic(char: str) -> bool:
    """
    True if char is an assigned Ethiopic character.

    The block has unassigned codepoints, among them the empty slots of
    each labiovelar block (U+1249) and U+12D7. They reach the corpus
    through broken encodings, and treating them as syllables gives
    the fidel arithmetic a vowel order for a letter that does not exist.

    Examples:
        >>> is_ethiopic("ቤ")
        True
        >>> is_ethiopic("\\u1249")
        False
    """
    return (
        bool(char)
        and _ETHIOPIC_FIRST <= char <= _ETHIOPIC_LAST
        and unicodedata.category(char) != "Cn"
    )


@lru_cache(maxsize=None)
def is_syllable(char: str) -> bool:
    """
    True if char is an Ethiopic syllable of a regular family.

    Punctuation, numerals and the three family-less letters ፘ ፙ ፚ are
    Ethiopic but not syllables, so the fidel arithmetic does not apply to
    them.

    Examples:
        >>> is_syllable("ቤ")
        True
        >>> is_syllable("።")
        False
    """
    return is_ethiopic(char) and char <= _SYLLABLE_LAST


@lru_cache(maxsize=None)
def fidel_order(char: str) -> int:
    """
    Vowel order of an Ethiopic syllable, 1-8.

    Order 6 is the bare consonant; order 8 is the labialized form.

    Args:
        char: A single character.

    Returns:
        1-8, or 0 if the character is not an Ethiopic syllable.

    Examples:
        >>> fidel_order("ቤ")
        5
        >>> fidel_order("ት")
        6
        >>> fidel_order("ቶ")
        7
    """
    if not is_syllable(char):
        return 0
    return ord(char) % _FAMILY_SIZE + 1


def _block_base(char: str) -> int:
    return ord(char) - ord(char) % _FAMILY_SIZE


def is_labiovelar(char: str) -> bool:
    """
    True if char is in one of the labiovelar blocks (ቈ ኈ ኰ ዀ ጐ).

    Examples:
        >>> is_labiovelar("ቋ")
        True
        >>> is_labiovelar("ሏ")
        False
    """
    return is_ethiopic(char) and _block_base(char) in _LABIOVELAR_CONSONANTS


def to_order(char: str, order: int) -> str:
    """
    Rewrite an Ethiopic syllable at another vowel order of its own family.

    The consonant is kept and the vowel replaced. Families are 8-aligned,
    so the target is family_base + order - 1. A labiovelar stays in its
    own block (ቋ at order 6 is ቍ), which is what fusion arithmetic needs.

    Args:
        char:  A single character.
        order: Target vowel order, 1-8.

    Returns:
        The character at that order, or unchanged if not an Ethiopic
        syllable.

    Raises:
        ValueError: If order is outside 1-8, or char is a labiovelar and
                    its block has no letter at that order.

    Examples:
        >>> to_order("ት", 7)
        'ቶ'
        >>> to_order("ደ", 4)
        'ዳ'
        >>> to_order("ቤ", 6)
        'ብ'
    """
    if not 1 <= order <= _FAMILY_SIZE:
        raise ValueError(f"vowel order must be 1-{_FAMILY_SIZE}, got {order}")
    if not is_syllable(char):
        return char
    if is_labiovelar(char) and order not in _LABIOVELAR_GLIDES:
        raise ValueError(f"{char} has no order-{order} form")
    return chr(_block_base(char) + order - 1)


def to_order6(char: str) -> str:
    """
    Bare-consonant (order 6) form of an Ethiopic syllable.

    Args:
        char: A single character.

    Returns:
        The order-6 form, or the character unchanged if not Ethiopic.

    Examples:
        >>> to_order6("ቶ")
        'ት'
        >>> to_order6("ቤ")
        'ብ'
    """
    return to_order(char, _ORDER6)


def bare_consonant(char: str) -> str:
    """
    The plain consonant of an Ethiopic syllable.

    The order-6 form for every family except the labiovelars, which
    give the consonant they labialize.

    Args:
        char: A single character.

    Returns:
        The consonant, or the character unchanged if not Ethiopic.

    Examples:
        >>> bare_consonant("ቶ")
        'ት'
        >>> bare_consonant("ቋ")
        'ቅ'
    """
    if is_labiovelar(char):
        return _LABIOVELAR_CONSONANTS[_block_base(char)]
    return to_order6(char)


def is_glottal(char: str) -> bool:
    """
    True if char belongs to the አ family, which carries the glottal stop.

    Examples:
        >>> is_glottal("አ")
        True
        >>> is_glottal("ቤ")
        False
    """
    if not is_ethiopic(char):
        return False
    return ord(char) - ord(char) % _FAMILY_SIZE == _GLOTTAL_BASE


def decompose_fidel(char: str) -> tuple[str, str]:
    """
    Split one fidel into its bare consonant and its vowel.

    Args:
        char: A single character.

    Returns:
        (consonant, vowel). The vowel is "" for order 6, which is the
        bare consonant already, and for anything that is not a syllable,
        which is returned whole as the consonant.

    Examples:
        >>> decompose_fidel("ቤ")
        ('ብ', 'ኤ')
        >>> decompose_fidel("ት")
        ('ት', '')
        >>> decompose_fidel("ቶ")
        ('ት', 'ኦ')
        >>> decompose_fidel("ቋ")
        ('ቅ', 'ዋ')
    """
    if not is_syllable(char):
        return (char, "")
    if is_labiovelar(char):
        return (bare_consonant(char), _LABIOVELAR_GLIDES[fidel_order(char)])
    return (to_order6(char), _ORDER_VOWELS[fidel_order(char)])


# ── Level 3 — Fidel ──────────────────────────────────────────────────────────


def level3_fidel(word: str) -> list[str]:
    """
    Level 3: the surface script as written, one fidel per token.

    Args:
        word: A surface word or morpheme.

    Returns:
        List of characters.

    Examples:
        >>> level3_fidel("ቤቶች")
        ['ቤ', 'ቶ', 'ች']
    """
    return list(word)


# ── Level 2 — Sound ──────────────────────────────────────────────────────────


def level2_sound(word: str) -> list[str]:
    """
    Level 2: full fidel decomposition into consonants and vowels.

    Gemination is not represented, per spec section 2.4.1.

    Args:
        word: A surface word or morpheme.

    Returns:
        Flat list of consonant and vowel characters.

    Examples:
        >>> level2_sound("ቤቶች")
        ['ብ', 'ኤ', 'ት', 'ኦ', 'ች']
        >>> "".join(level2_sound("ቤቶች"))
        'ብኤትኦች'
    """
    units: list[str] = []
    for char in word:
        consonant, vowel = decompose_fidel(char)
        units.append(consonant)
        if vowel:
            units.append(vowel)
    return units


# ── Lossless sound text ──────────────────────────────────────────────────────
#
# Level 2 is a display level and loses two distinctions: ቍ and ቅው both
# give ቅ ው, and ቋ and ቇ both give ቅ ዋ. The BPE level trains on sound text
# and must give back exactly what it was given, so sound_text() writes a
# labiovelar as its consonant, the marker ʷ, and its vowel: ቋ -> ቅʷኣ.
#
# Decoding is unambiguous because every syllable starts with its bare
# consonant (an order-6 letter), and no vowel symbol is an order-6 letter.
# A real glottal letter starts with its consonant too: ኡ is እ + ኡ.

_LABIOVELAR_MARK = "ʷ"
_VOWEL_ORDERS = {vowel: order for order, vowel in _ORDER_VOWELS.items() if vowel}
_LABIOVELAR_BLOCKS = {consonant: base for base, consonant in _LABIOVELAR_CONSONANTS.items()}


def sound_text(word: str) -> str:
    """
    The word as lossless sound text, for the BPE level.

    As Level 2, except that a labiovelar's glide is written ʷ plus its
    vowel, so that from_sound_text() can restore every word exactly.

    Examples:
        >>> sound_text("ቤቶች")
        'ብኤትኦች'
        >>> sound_text("ቋንቋ")
        'ቅʷኣንቅʷኣ'
    """
    units = []
    for char in word:
        if is_labiovelar(char):
            units.append(bare_consonant(char) + _LABIOVELAR_MARK + _ORDER_VOWELS[fidel_order(char)])
        else:
            units.extend(decompose_fidel(char))
    return "".join(units)


def from_sound_text(text: str) -> str:
    """
    Restore the written text from sound_text().

    Examples:
        >>> from_sound_text("ብኤትኦች")
        'ቤቶች'
        >>> from_sound_text("ቅʷኣንቅʷኣ")
        'ቋንቋ'
    """
    out = []
    i = 0
    while i < len(text):
        char = text[i]
        i += 1
        if not (is_syllable(char) and fidel_order(char) == _ORDER6 and not is_labiovelar(char)):
            out.append(char)
            continue
        base = _block_base(char)
        if i < len(text) and text[i] == _LABIOVELAR_MARK and char in _LABIOVELAR_BLOCKS:
            base = _LABIOVELAR_BLOCKS[char]
            i += 1
            order = _ORDER6
            if i < len(text) and text[i] in _VOWEL_ORDERS:
                order = _VOWEL_ORDERS[text[i]]
                i += 1
            out.append(chr(base + order - 1))
            continue
        if i < len(text) and text[i] in _VOWEL_ORDERS:
            out.append(chr(base + _VOWEL_ORDERS[text[i]] - 1))
            i += 1
        else:
            out.append(char)
    return "".join(out)


# ── Consonant skeleton ───────────────────────────────────────────────────────


def consonant_skeleton(morpheme: str) -> list[str] | None:
    """
    The consonant skeleton of an INDEPENDENT morpheme.

    This is the root proposal for an entry with no reviewed root, not
    Level 1 itself. It equals the root for an underived word (ቤት ብ-ት)
    and keeps derivational consonants for a derived one (መሆን ም-ህ-ን,
    root ህ-ን). See docs/dev_plans/level1_root_redesign.md.

    Must be given a single morpheme, not a surface word. Level 4 finds
    the independent morpheme; running this on ቤቶች directly would return
    ብ-ት-ች, with the plural suffix's consonant inside the root.

    Applies both rules described in the module docstring: decomposition
    vowels never enter the root, and among surface አ-family letters only
    a word-initial one counts as a consonant. A labiovelar gives its
    plain consonant (ቋንቋ -> ቅ-ን-ቅ).

    Args:
        morpheme: A single independent morpheme.

    Returns:
        List of order-6 consonants, or None when no root can be
        extracted. Per spec section 2.2, Level 1 is valid only for
        Semitic content morphemes — it returns None for grammatical
        morphemes, loanwords, and ambiguous standalone vowel words.
        Detecting those needs the lexicon; what is detectable here is
        the case where nothing survives extraction.

    Examples:
        >>> consonant_skeleton("ቤት")
        ['ብ', 'ት']
        >>> consonant_skeleton("አለ")
        ['እ', 'ል']
        >>> consonant_skeleton("ጣእም")
        ['ጥ', 'ም']
        >>> consonant_skeleton("አ")
        ['እ']
    """
    root: list[str] = []
    for index, char in enumerate(morpheme):
        if not is_ethiopic(char):
            continue
        # Rule 2: an internal glottal is a template vowel, not a root
        # consonant. An initial one is the consonant itself.
        if is_glottal(char) and index > 0:
            continue
        root.append(bare_consonant(char))

    return root or None
