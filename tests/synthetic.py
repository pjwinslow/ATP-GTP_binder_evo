"""Synthetic protein + ATP (+ ion) structures for tests (no GPU, no ESMFold2).

The protein is an ideal helical backbone (N, CA, C, O, CB) with the sequence's
residue names; ATP is the CCD template (biotite ships it offline) pressed against
the helix; the ion sits 2.1 A from an ATP beta-phosphate oxygen. Written in the
layout ames' cif2pdb produces, B-factor = pLDDT.
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


def atp_template() -> list:
    """[(name, element, xyz)] heavy atoms of ATP, centred on the origin."""
    import biotite.structure.info as info
    a = info.residue("ATP")
    a = a[a.element != "H"]
    xyz = a.coord - a.coord.mean(axis=0)
    return [(str(n), str(e).upper(), x) for n, e, x in zip(a.atom_name, a.element, xyz)]


def build_complex(seq: str, ion: str = "", plddt: float = 80.0, atp_plddt: float = 75.0,
                  min_gap: float = 3.2) -> str:
    """PDB text: chain A protein, chain B ATP, chain C `ion` (CCD code, optional)."""
    bb = helix_backbone(len(seq))
    prot_xyz = np.array([x for r in bb for x in r])
    centre = np.array([r[1] for r in bb]).mean(axis=0)
    axis = np.array([r[1] for r in bb])
    axis = np.linalg.svd(axis - centre)[2][0]  # helix axis
    perp = np.cross(axis, [0.3, 0.5, 0.8])
    perp /= np.linalg.norm(perp)

    atp = atp_template()
    atp_xyz = np.array([x for _, _, x in atp])
    offset = 5.0
    while True:  # slide ATP outward until nothing is closer than min_gap
        moved = atp_xyz + centre + perp * offset
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
    for (name, elem, _), xyz in zip(atp, moved):
        serial += 1
        lines.append(_fmt(serial, "HETATM", name, "ATP", "B", 1, xyz, atp_plddt, elem))
    if ion:
        names = [n for n, _, _ in atp]
        o2b, pb = moved[names.index("O2B")], moved[names.index("PB")]
        pos = o2b + 2.1 * (o2b - pb) / np.linalg.norm(o2b - pb)
        serial += 1
        lines.append(_fmt(serial, "HETATM", ion, ion, "C", 1, pos, atp_plddt, ion[:1] + ion[1:].lower()))
    return "\n".join(lines) + "\nEND\n"
