"""Stand-in for ames' esmfold2_runner module (same call signature, no GPU).
Handles `--ligand ATP[,ion]` and `--ligand GTP[,ion]`.

ames.py does ``from esmfold2_runner import esmfold2_runner``; mock_ames.py puts this
module into sys.modules under that name so the real ames code path (ligand input,
scoring, selection, logging) runs end to end. Confidence values are a made-up
function of the sequence - the fraction of charged residues - so that selection
has something to climb; they say nothing about real folding or binding.
MOCK_ION_ON_PROTEIN=1 puts the ion against the protein, away from the nucleotide.
"""
import os

from synthetic import build_complex

CHARGED = set("DERK")


def esmfold2_runner(seq_data_list):
    if isinstance(seq_data_list, dict):
        seq_data_list = [seq_data_list]
    structures, plddts, ptms, iptms = [], [], [], []
    for data in seq_data_list:
        seq = data["seq1"]["sequence"]
        ligands = data.get("ligand", [])
        assert ligands and ligands[0] in ("ATP", "GTP"), f"expected a nucleotide first, got {ligands}"
        nucleotide = ligands[0]
        ion = ligands[1] if len(ligands) > 1 else ""
        frac = sum(aa in CHARGED for aa in seq) / len(seq)
        structures.append(build_complex(seq, ion=ion, nucleotide=nucleotide,
                                        ion_on_protein=os.environ.get("MOCK_ION_ON_PROTEIN") == "1"))
        plddts.append(0.8)  # mean pLDDT is 0-1 (structure B-factors are pLDDT*100), as in the real runner
        ptms.append(0.6)
        iptms.append(round(min(0.95, 0.1 + 1.5 * frac), 3))
    return structures, plddts, ptms, iptms
