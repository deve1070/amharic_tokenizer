"""
The representation levels.

Level 3 (Fidel), Level 2 (Sound), and Level 1 (Consonantal Root) live
here, along with the fidel arithmetic they share. Level 4 (Morpheme) is
lexicon-anchored and lives with the segmenter once a validated lexicon
exists — nothing here depends on it.

── Computational order ──────────────────────────────────────────────────

The spec computes 4 -> 3 -> 2 -> 1. Level 1 takes the root of the
INDEPENDENT morpheme only, so it needs morpheme boundaries first: run it
on the raw surface form ቤቶች and you get ብ-ት-ች, with the suffix consonant
polluting the root. The functions here are therefore written to be
applied to a single morpheme, and level1_root() documents that contract.

── Fidel arithmetic ─────────────────────────────────────────────────────

Ethiopic syllable families are 8-aligned: seven vowel orders plus a
labialized form in the eighth slot. So order = codepoint % 8 + 1, and
the bare consonant (order 6) is always family_base + 5.

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

── Level 1 root extraction ──────────────────────────────────────────────

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

_FAMILY_SIZE = 8
_ORDER6 = 6

_ETHIOPIC_FIRST = "ሀ"
_ETHIOPIC_LAST = "፿"

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

# The አ family (U+12A0-U+12A7) carries the glottal stop.
_GLOTTAL_BASE = ord("አ") - ord("አ") % _FAMILY_SIZE


def is_ethiopic(char: str) -> bool:
    """True if char is an Ethiopic character."""
    return bool(char) and _ETHIOPIC_FIRST <= char <= _ETHIOPIC_LAST


def fidel_order(char: str) -> int:
    """
    Vowel order of an Ethiopic syllable, 1-8.

    Order 6 is the bare consonant; order 8 is the labialized form.

    Args:
        char: A single character.

    Returns:
        1-8, or 0 if the character is not Ethiopic.

    Examples:
        >>> fidel_order("ቤ")
        5
        >>> fidel_order("ት")
        6
        >>> fidel_order("ቶ")
        7
    """
    if not is_ethiopic(char):
        return 0
    return ord(char) % _FAMILY_SIZE + 1


def to_order(char: str, order: int) -> str:
    """
    Rewrite an Ethiopic syllable at another vowel order of its own family.

    The consonant is kept and the vowel replaced. Families are 8-aligned,
    so the target is family_base + order - 1.

    Args:
        char:  A single character.
        order: Target vowel order, 1-8.

    Returns:
        The character at that order, or unchanged if not Ethiopic.

    Raises:
        ValueError: If order is outside 1-8.

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
    if not is_ethiopic(char):
        return char
    base = ord(char) - ord(char) % _FAMILY_SIZE
    return chr(base + order - 1)


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
        char: A single Ethiopic character.

    Returns:
        (consonant, vowel). The vowel is "" for order 6, which is the
        bare consonant already.

    Examples:
        >>> decompose_fidel("ቤ")
        ('ብ', 'ኤ')
        >>> decompose_fidel("ት")
        ('ት', '')
        >>> decompose_fidel("ቶ")
        ('ት', 'ኦ')
    """
    if not is_ethiopic(char):
        return (char, "")
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


# ── Level 1 — Consonantal Root ───────────────────────────────────────────────


def level1_root(morpheme: str) -> list[str] | None:
    """
    Level 1: the consonantal root of an INDEPENDENT morpheme.

    Must be given a single morpheme, not a surface word. Level 4 finds
    the independent morpheme; running this on ቤቶች directly would return
    ብ-ት-ች, with the plural suffix's consonant inside the root.

    Applies both rules described in the module docstring: decomposition
    vowels never enter the root, and among surface አ-family letters only
    a word-initial one counts as a consonant.

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
        >>> level1_root("ቤት")
        ['ብ', 'ት']
        >>> level1_root("አለ")
        ['እ', 'ል']
        >>> level1_root("ጣእም")
        ['ጥ', 'ም']
        >>> level1_root("አ")
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
        root.append(to_order6(char))

    return root or None
