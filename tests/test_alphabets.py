"""Alphabet registration tests. Needs ames importable (python >= 3.12 env).

    python tests/test_alphabets.py      # or: pytest tests
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ames_alphabets import ALPHABETS, STANDARD_AA, letters_from_key, register_alphabet, resolve_alphabet

try:
    import ames  # noqa: F401  puts ames' source dir on sys.path
    import evolution
except ImportError:  # pragma: no cover
    evolution = None


def _evolver(alphabet, normalize=True, mutations="npm"):
    key = register_alphabet(evolution.Evolver, alphabet, normalize)
    return key, evolution.Evolver(protein_alphabet=key, protein_mutations=mutations)


def test_nested_alphabets():
    names = list(ALPHABETS)
    for small, big in zip(names, names[1:]):
        assert set(ALPHABETS[small]) < set(ALPHABETS[big])
    assert [len(ALPHABETS[n]) for n in names] == [5, 9, 13, 20]


def test_closure_under_all_mutation_operators():
    """No operator (point, indel, duplication, permutation, random insert/reset)
    may introduce a residue outside the alphabet."""
    random.seed(1)
    for name, letters in ALPHABETS.items():
        for mutations in ("npm", "pmo"):
            _, ev = _evolver(name, mutations=mutations)
            seq = ev.randomseq("protein", 65)
            assert set(seq) <= set(letters)
            ops = set()
            for _ in range(20000):
                seq, info = ev.mutate("protein", seq)
                ops.add(next((c for c in info if c in "+-*/%pdr"), "."))
                assert set(seq) <= set(letters), (name, mutations, info)
                if len(seq) > 300:  # full duplications grow without bound
                    seq = ev.randomseq("protein", 65)
            assert {".", "+", "-"} <= ops
            if mutations == "npm":
                assert {"*", "/", "%", "p", "d"} <= ops


def test_operator_mix_identical_across_alphabets():
    """Substitution share of the mutation pool must not depend on alphabet size."""
    shares = {}
    for name, letters in ALPHABETS.items():
        _, ev = _evolver(name)
        shares[name] = sum(ev.protein_weigths[: len(letters)])
    assert max(shares.values()) - min(shares.values()) < 1e-12, shares

    _, raw = _evolver("GADVP", normalize=False)
    assert sum(raw.protein_weigths[:5]) < shares["GADVP"] - 0.1  # native ames weights differ


def test_all20_equals_stock_uniform():
    key, ev = _evolver("ALL20")
    stock = evolution.Evolver(protein_alphabet="uniform")
    assert ev.protein_alphabet == stock.protein_alphabet
    assert all(abs(a - b) < 1e-12 for a, b in zip(ev.protein_weigths, stock.protein_weigths))


def test_resolve_and_header_roundtrip():
    assert resolve_alphabet("gadvp") == "GADVP"
    assert resolve_alphabet("STQ") == "STQ"
    for bad in ("GADVPX", "GGA", "G"):
        try:
            resolve_alphabet(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted")
    assert letters_from_key("GADVPSELT") == ALPHABETS["GADVPSELT"]
    assert letters_from_key("GADVP-raw") == "GADVP"
    assert letters_from_key("uniform") == STANDARD_AA


if __name__ == "__main__":
    if evolution is None:
        sys.exit("ames is not importable; activate the env where it is installed")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
