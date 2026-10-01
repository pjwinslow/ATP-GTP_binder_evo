"""Reduced amino-acid alphabets for ames.

ames draws every residue (initial random sequences, substitutions, insertions)
from the alphabet of its ``Evolver``. Alphabets are the keys of
``Evolver.protein`` (``uniform``, ``uniprot``, ``codonrates``) and there is no
working way to add one from the command line (``-ed/--evoldict`` is parsed but
never handed to ``Evolver``). ``register_alphabet`` adds named alphabets to
``Evolver.protein`` at runtime so that ``-pa1 <name>`` selects them.

Rate normalisation
------------------
In ames every residue in the alphabet gets weight 1 and the other mutation
operators have fixed weights (insertion 1, deletion 1, duplication 0.4, ...), all
drawn from one pool. With a 5-letter alphabet the substitution share of the pool
is 5/7.9 = 63 % instead of 20/22.9 = 87 % for 20 letters, so smaller alphabets
would also get ~3x more indels/duplications. That is a confound when comparing
alphabets. By default each letter therefore gets weight ``20 / N`` so the total
substitution weight is 20 for every alphabet, the mix of operator types is
identical across alphabets, and ``ALL20`` equals ames' stock ``uniform``.
"""

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Nested: each alphabet contains the previous one.
ALPHABETS = {
    "GADVP": "GADVP",
    "GADVPSELT": "GADVPSELT",
    "GADVPSELTRIQN": "GADVPSELTRIQN",
    "ALL20": STANDARD_AA,
}

RAW_SUFFIX = "-raw"  # key suffix for un-normalised weights, recorded in progress.log


def resolve_alphabet(spec: str) -> str:
    """Name from ``ALPHABETS`` or an explicit letter set -> validated letters."""
    spec = spec.strip().upper()
    letters = ALPHABETS.get(spec, spec)
    bad = sorted(set(letters) - set(STANDARD_AA))
    if bad:
        raise ValueError(f"alphabet {spec!r}: {''.join(bad)} not standard amino acid letters")
    if len(set(letters)) != len(letters):
        raise ValueError(f"alphabet {spec!r}: repeated letters")
    if len(letters) < 2:
        raise ValueError(f"alphabet {spec!r}: needs at least 2 letters")
    return letters


def alphabet_weights(letters: str, normalize: bool = True) -> dict:
    """Per-letter mutation weights; see the module docstring for ``normalize``."""
    w = len(STANDARD_AA) / len(letters) if normalize else 1.0
    return {aa: w for aa in letters}


def register_alphabet(evolver_cls, spec: str, normalize: bool = True) -> str:
    """Add the alphabet to ``evolver_cls.protein``; returns the key for ``-pa1``."""
    letters = resolve_alphabet(spec)
    key = spec.strip().upper() + ("" if normalize else RAW_SUFFIX)
    evolver_cls.protein[key] = alphabet_weights(letters, normalize)
    return key


def letters_from_key(key: str) -> str:
    """Inverse of ``register_alphabet`` for a ``protein_alphabet1`` log header value."""
    key = key.strip()
    if key.endswith(RAW_SUFFIX):
        key = key[: -len(RAW_SUFFIX)]
    if key.lower() == "uniform":
        return STANDARD_AA
    return resolve_alphabet(key)
