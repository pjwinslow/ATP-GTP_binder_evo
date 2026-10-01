"""summarize_matrix.structure_metrics on hand-built geometries. Needs biotite (ames dependency).

    python tests/test_structure_metrics.py      # or: pytest tests
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from summarize_matrix import ATP_N_HEAVY, parse_ligands, structure_metrics  # noqa: E402
from synthetic import _fmt, atp_template, build_complex  # noqa: E402


def _scenario(low_plddt_tyr=40.0):
    """ATP at the origin, one residue per situation, Mg near O2B.

      1 LYS  NZ  2.8 A from PG                     -> phosphate contact
      2 ALA  CB  25 A away                         -> no contact
      3 TYR  OH  3.0 A from N6 but pLDDT 40        -> removed by the pLDDT filter
      4 ASP  OD1 2.1 A from Mg                     -> Mg coordination AND an ATP contact:
                                                      Mg is 2.1 A from O2B, so OD1 is <= 4.2 A from ATP
    """
    atp = atp_template()
    pos = {n: x for n, _, x in atp}
    mg = pos["O2B"] + 2.1 * (pos["O2B"] - pos["PB"]) / np.linalg.norm(pos["O2B"] - pos["PB"])
    away = (pos["O2B"] - pos["PB"]) / np.linalg.norm(pos["O2B"] - pos["PB"])
    lines = [
        _fmt(1, "ATOM", "NZ", "LYS", "A", 1, pos["PG"] + [2.8, 0, 0], 85.0, "N"),
        _fmt(2, "ATOM", "CB", "ALA", "A", 2, [25.0, 25.0, 25.0], 85.0, "C"),
        _fmt(3, "ATOM", "OH", "TYR", "A", 3, pos["N6"] + [0, 3.0, 0], low_plddt_tyr, "O"),
        _fmt(4, "ATOM", "OD1", "ASP", "A", 4, mg + 2.1 * away, 85.0, "O"),
    ]
    for i, (n, e, x) in enumerate(atp, start=5):
        lines.append(_fmt(i, "HETATM", n, "ATP", "B", 1, x, 75.0, e))
    lines.append(_fmt(99, "HETATM", "MG", "MG", "C", 1, mg, 75.0, "Mg"))
    return "\n".join(lines) + "\n"


def test_contacts_and_plddt_filter():
    m = structure_metrics(_scenario())
    assert m["atp_atoms"] == ATP_N_HEAVY == 31 and m["atp_atoms_unrecognized"] == 0
    assert m["atp_contact_res"] == 2  # LYS + ASP; TYR removed by pLDDT filter, ALA too far
    assert m["phosphate_contact_res"] == 2 and m["adenine_contact_res"] == 0
    assert abs(m["atp_lcd"] - 2 / 31) < 1e-12
    assert m["contact_composition"]["K"] == 1 and m["contact_composition"]["D"] == 1
    assert sum(m["contact_composition"].values()) == 2

    # brute-force check of the fraction of ATP atoms within 4.5 A of a confident protein atom
    pos = {n: x for n, _, x in atp_template()}
    mg = pos["O2B"] + 2.1 * (pos["O2B"] - pos["PB"]) / np.linalg.norm(pos["O2B"] - pos["PB"])
    away = (pos["O2B"] - pos["PB"]) / np.linalg.norm(pos["O2B"] - pos["PB"])
    protein = [pos["PG"] + [2.8, 0, 0], mg + 2.1 * away]  # NZ, OD1 (the confident ones)
    near = sum(any(np.linalg.norm(x - p) < 4.5 for p in protein) for x in pos.values())
    assert abs(m["atp_atom_contact_frac"] - near / 31) < 1e-12


def test_plddt_filter_off_counts_tyr():
    m = structure_metrics(_scenario(), min_plddt=0.0)
    assert m["atp_contact_res"] == 3 and m["adenine_contact_res"] == 1  # TYR now counts


def test_ion_metrics():
    m = structure_metrics(_scenario())
    assert m["ion"] == "MG"
    assert m["ion_coord_protein_res"] == 1 and m["ion_coord_protein_atoms"] == 1  # ASP OD1
    assert m["ion_coord_atp_O"] == 1                                              # ATP O2B
    assert m["ion_coord_number"] == 2
    assert m["ion_contact_res"] == 1  # ASP, within 4.5 A of Mg


def test_no_ion_and_no_atp():
    text = "".join(line + "\n" for line in _scenario().splitlines() if "MG" not in line)
    m = structure_metrics(text)
    assert m["ion"] == "" and "ion_contact_res" not in m
    prot_only = "".join(line + "\n" for line in _scenario().splitlines() if line.startswith("ATOM"))
    m = structure_metrics(prot_only)
    assert m["atp_found"] is False and "atp_contact_res" not in m


def test_parse_ligands_header():
    # ames writes args.ligand (a list) into progress.log with str(), i.e. a Python list repr
    assert parse_ligands("['ATP', 'MG']") == ["ATP", "MG"]
    assert parse_ligands("['ATP']") == ["ATP"]
    assert parse_ligands("ATP,MN") == ["ATP", "MN"]
    assert parse_ligands("None") == []


def test_synthetic_complex_is_sane():
    m = structure_metrics(build_complex("GADVPSELTRIQN" * 3, ion="MG"))
    assert m["atp_atoms"] == 31 and m["atp_atoms_unrecognized"] == 0
    assert m["atp_contact_res"] >= 3 and 0 < m["atp_atom_contact_frac"] <= 1
    assert m["ion"] == "MG" and m["ion_coord_atp_O"] >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
