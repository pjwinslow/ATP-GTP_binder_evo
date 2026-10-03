"""Synthetic protein + nucleotide (+ ion) structures for tests (no GPU, no ESMFold2).

The protein is an ideal helical backbone (N, CA, C, O, CB) with the sequence's
residue names; the nucleotide (ATP or GTP) is the CCD template (biotite ships it
offline) pressed against the helix; the ion sits 2.1 A from a beta-phosphate oxygen
(or, with ``ion_on_protein``, on the far side of the helix: touching residues, not the nucleotide).
Written in the layout ames' cif2pdb produces, B-factor = pLDDT.
"""
import numpy as np

THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
         "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
         "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def _place(a, b, c, bond, angle, dihedral):
    """NeRF: atom d at `bond` from c, angle b-c-d, dihedral a-b-c-d (degrees)."""
    ang, dih = np.radians(angle), np.radians(dihedral)
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + (-bond * np.cos(ang)) * bc + bond * np.sin(ang) * np.cos(dih) * m + bond * np.sin(ang) * np.sin(dih) * n


def helix_backbone(n: int, phi=-57.0, psi=-47.0) -> list:
    """[(N, CA, C, O, CB)] per residue, ideal alpha-helix by default."""
    N = np.zeros(3)
    CA = np.array([1.458, 0, 0])
    C = CA + 1.525 * np.array([-np.cos(np.radians(180 - 111.2)), np.sin(np.radians(180 - 111.2)), 0])
    res = []
    for i in range(n):
        O = _place(N, CA, C, 1.231, 120.8, psi + 180)
        CB = _place(C, N, CA, 1.53, 110.5, -122.6)
        res.append((N, CA, C, O, CB))
        if i == n - 1:
            break
        N2 = _place(N, CA, C, 1.329, 116.2, psi)
        CA2 = _place(CA, C, N2, 1.458, 121.7, 180.0)
        C2 = _place(C, N2, CA2, 1.525, 111.2, phi)
        N, CA, C = N2, CA2, C2
    return res


def _fmt(serial, rec, name, resname, chain, resid, xyz, bfac, elem):
    nm = f" {name:<3s}" if len(name) < 4 else f"{name:<4s}"
    return (f"{rec:<6s}{serial:>5d} {nm} {resname:>3s} {chain}{resid:>4d}    "
            f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{bfac:>6.2f}          {elem:>2s}  ")


def nucleotide_template(nucleotide: str = "ATP") -> list:
    """[(name, element, xyz)] heavy atoms of a CCD nucleotide (ATP, GTP), centred on the origin."""
    import biotite.structure.info as info
    a = info.residue(nucleotide)
    a = a[a.element != "H"]
    xyz = a.coord - a.coord.mean(axis=0)
    return [(str(n), str(e).upper(), x) for n, e, x in zip(a.atom_name, a.element, xyz)]


def build_complex(seq: str, ion: str = "", nucleotide: str = "ATP", plddt: float = 80.0,
                  nuc_plddt: float = 75.0, min_gap: float = 3.2, ion_on_protein: bool = False) -> str:
    """PDB text: chain A protein, chain B nucleotide, chain C `ion` (CCD code, optional).

    ion_on_protein: put the ion 2.4 A from the CA of residue 4, on the side of the helix away from the
    nucleotide, so it contacts protein residues and no nucleotide atom (an ion-only "binder")."""
    bb = helix_backbone(len(seq))
    prot_xyz = np.array([x for r in bb for x in r])
    centre = np.array([r[1] for r in bb]).mean(axis=0)
    axis = np.array([r[1] for r in bb])
    axis = np.linalg.svd(axis - centre)[2][0]  # helix axis
    perp = np.cross(axis, [0.3, 0.5, 0.8])
    perp /= np.linalg.norm(perp)

    nuc = nucleotide_template(nucleotide)
    nuc_xyz = np.array([x for _, _, x in nuc])
    offset = 5.0
    while True:  # slide the nucleotide outward until nothing is closer than min_gap
        moved = nuc_xyz + centre + perp * offset
        if np.linalg.norm(moved[:, None] - prot_xyz[None], axis=2).min() >= min_gap:
            break
        offset += 0.25

    lines, serial = [], 0
    for i, (aa, r) in enumerate(zip(seq, bb), start=1):
        for name, elem, xyz in zip(("N", "CA", "C", "O", "CB"), "NCCOC", r):
            if name == "CB" and aa == "G":
                continue
            serial += 1
            lines.append(_fmt(serial, "ATOM", name, THREE[aa], "A", i, xyz, plddt, elem))
    for (name, elem, _), xyz in zip(nuc, moved):
        serial += 1
        lines.append(_fmt(serial, "HETATM", name, nucleotide, "B", 1, xyz, nuc_plddt, elem))
    if ion:
        names = [n for n, _, _ in nuc]
        o2b, pb = moved[names.index("O2B")], moved[names.index("PB")]
        pos = o2b + 2.1 * (o2b - pb) / np.linalg.norm(o2b - pb)
        if ion_on_protein:
            pos = bb[min(3, len(seq) - 1)][1] - 2.4 * perp
        serial += 1
        lines.append(_fmt(serial, "HETATM", ion, ion, "C", 1, pos, nuc_plddt, ion[:1] + ion[1:].lower()))
    return "\n".join(lines) + "\nEND\n"
