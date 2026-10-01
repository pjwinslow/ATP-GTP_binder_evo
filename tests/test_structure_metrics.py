"""summarize_matrix.structure_metrics on hand-built geometries. Needs biotite (ames dependency).

    python tests/test_structure_metrics.py      # or: pytest tests
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from summarize_matrix import (BASE, N_HEAVY, NUCLEOTIDES, PHOSPHATE, RIBOSE, parse_ligands,  # noqa: E402
                              structure_metrics)
from synthetic import _fmt, build_complex, nucleotide_template  # noqa: E402

BASE_ATOM = {"ATP": "N6", "GTP": "O6"}  # an exocyclic base atom to put a residue next to


def _mg_and_away(pos):
    away = (pos["O2B"] - pos["PB"]) / np.linalg.norm(pos["O2B"] - pos["PB"])
    return pos["O2B"] + 2.1 * away, away


def _scenario(nucleotide="ATP", low_plddt_tyr=40.0, extra=()):
    """Nucleotide at the origin, one residue per situation, Mg near O2B.

      1 LYS  NZ  2.8 A from PG                     -> phosphate contact
      2 ALA  CB  25 A away                         -> no contact
      3 TYR  OH  3.0 A from the base               -> base contact, removed by the pLDDT filter (40)
      4 ASP  OD1 2.1 A from Mg                     -> Mg coordination AND a nucleotide contact:
                                                      Mg is 2.1 A from O2B, so OD1 is <= 4.2 A from it
    `extra` = additional (resname, atom, xyz) residues, numbered from 5.
    """
    nuc = nucleotide_template(nucleotide)
    pos = {n: x for n, _, x in nuc}
    mg, away = _mg_and_away(pos)
    lines = [
        _fmt(1, "ATOM", "NZ", "LYS", "A", 1, pos["PG"] + [2.8, 0, 0], 85.0, "N"),
        _fmt(2, "ATOM", "CB", "ALA", "A", 2, [25.0, 25.0, 25.0], 85.0, "C"),
        _fmt(3, "ATOM", "OH", "TYR", "A", 3, pos[BASE_ATOM[nucleotide]] + [0, 3.0, 0], low_plddt_tyr, "O"),
        _fmt(4, "ATOM", "OD1", "ASP", "A", 4, mg + 2.1 * away, 85.0, "O"),
    ]
    for k, (resname, atom, xyz) in enumerate(extra, start=5):
        lines.append(_fmt(k, "ATOM", atom, resname, "A", k, xyz, 85.0, atom[0]))
    for i, (n, e, x) in enumerate(nuc, start=100):
        lines.append(_fmt(i, "HETATM", n, nucleotide, "B", 1, x, 75.0, e))
    lines.append(_fmt(999, "HETATM", "MG", "MG", "C", 1, mg, 75.0, "Mg"))
    return "\n".join(lines) + "\n"


def test_contacts_and_plddt_filter():
    for nuc in NUCLEOTIDES:
        m = structure_metrics(_scenario(nuc), nuc)
        assert m["nuc_atoms"] == N_HEAVY[nuc] and m["nuc_atoms_unrecognized"] == 0, nuc
        assert m["nuc_contact_res"] == 2, nuc  # LYS + ASP; TYR removed by pLDDT filter, ALA too far
        assert m["phosphate_contact_res"] == 2 and m["base_contact_res"] == 0, nuc
        assert abs(m["nuc_lcd"] - 2 / N_HEAVY[nuc]) < 1e-12, nuc
        assert m["contact_composition"]["K"] == 1 and m["contact_composition"]["D"] == 1
        assert sum(m["contact_composition"].values()) == 2

        # brute-force check of the fraction of nucleotide atoms within 4.5 A of a confident protein atom
        pos = {n: x for n, _, x in nucleotide_template(nuc)}
        mg, away = _mg_and_away(pos)
        protein = [pos["PG"] + [2.8, 0, 0], mg + 2.1 * away]  # NZ, OD1 (the confident ones)
        near = sum(any(np.linalg.norm(x - p) < 4.5 for p in protein) for x in pos.values())
        assert abs(m["nuc_atom_contact_frac"] - near / N_HEAVY[nuc]) < 1e-12, nuc


def test_plddt_filter_off_counts_tyr():
    for nuc in NUCLEOTIDES:
        m = structure_metrics(_scenario(nuc), nuc, min_plddt=0.0)
        assert m["nuc_contact_res"] == 3 and m["base_contact_res"] == 1, nuc  # TYR now counts


def test_ion_metrics():
    for nuc in NUCLEOTIDES:
        m = structure_metrics(_scenario(nuc), nuc)
        assert m["ion"] == "MG"
        assert m["ion_coord_protein_res"] == 1 and m["ion_coord_protein_atoms"] == 1  # ASP OD1
        assert m["ion_coord_nuc_O"] == 1                                              # O2B
        assert m["ion_coord_number"] == 2
        assert m["ion_contact_res"] == 1  # ASP, within 4.5 A of Mg


def test_guanine_specific_atoms_count_as_base():
    """N2 exists only in guanine; a residue next to it is a base contact for GTP."""
    pos = {n: x for n, _, x in nucleotide_template("GTP")}
    text = _scenario("GTP", extra=[("SER", "OG", pos["N2"] + [0, 0, 3.0])])
    m = structure_metrics(text, "GTP")
    assert m["base_contact_res"] == 1 and m["nuc_atoms_unrecognized"] == 0


def test_no_ion_no_nucleotide_and_wrong_nucleotide():
    for nuc in NUCLEOTIDES:
        text = "".join(line + "\n" for line in _scenario(nuc).splitlines() if "MG" not in line)
        m = structure_metrics(text, nuc)
        assert m["ion"] == "" and "ion_contact_res" not in m
        prot_only = "".join(line + "\n" for line in _scenario(nuc).splitlines() if line.startswith("ATOM"))
        m = structure_metrics(prot_only, nuc)
        assert m["nuc_found"] is False and "nuc_contact_res" not in m
    # asking for GTP in an ATP complex finds no nucleotide instead of mislabelling ATP atoms
    assert structure_metrics(_scenario("ATP"), "GTP")["nuc_found"] is False
    try:
        structure_metrics(_scenario("ATP"), "CTP")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported nucleotide accepted")


def test_atom_sets_match_the_chemical_component_library():
    """The hard-coded phosphate/base/ribose names must partition the CCD heavy atoms exactly."""
    import biotite.structure.info as info
    for nuc in NUCLEOTIDES:
        a = info.residue(nuc)
        ccd = set(map(str, a[a.element != "H"].atom_name))
        assert PHOSPHATE | RIBOSE | BASE[nuc] == ccd, (nuc, ccd ^ (PHOSPHATE | RIBOSE | BASE[nuc]))
        assert not (PHOSPHATE & RIBOSE) and not (PHOSPHATE & BASE[nuc]) and not (RIBOSE & BASE[nuc])
    assert (BASE["ATP"] ^ BASE["GTP"]) == {"N6", "O6", "N2"}


def test_parse_ligands_header():
    # ames writes args.ligand (a list) into progress.log with str(), i.e. a Python list repr
    assert parse_ligands("['ATP', 'MG']") == ["ATP", "MG"]
    assert parse_ligands("['GTP']") == ["GTP"]
    assert parse_ligands("GTP,MN") == ["GTP", "MN"]
    assert parse_ligands("None") == []


def test_synthetic_complex_is_sane():
    for nuc in NUCLEOTIDES:
        m = structure_metrics(build_complex("GADVPSELTRIQN" * 3, ion="MG", nucleotide=nuc), nuc)
        assert m["nuc_atoms"] == N_HEAVY[nuc] and m["nuc_atoms_unrecognized"] == 0
        assert m["nuc_contact_res"] >= 3 and 0 < m["nuc_atom_contact_frac"] <= 1
        assert m["ion"] == "MG" and m["ion_coord_nuc_O"] >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
