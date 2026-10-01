#!/usr/bin/env python3
"""Summarise a matrix of ames runs (alphabet x cation x replicate).

    summarize_matrix.py outputs/atp_matrix [outputs/atp_matrix_neutral ...] [--out DIR]

Run ``visualames -l <run>/progress.log`` in every run first (submit_matrix.sh
does); it writes the ``lineage.tsv`` read here. Runs are grouped by what their
``progress.log`` header says (protein_alphabet1, ligand, beta, annealing), not by
directory name, so selected runs and neutral-drift controls can sit in any tree.

Why the ATP metrics are recomputed
----------------------------------
ames' ``lcd``/``iplddt`` pool all ligand chains. In an ``ATP,MG`` run Mg is a
separate chain: residues touching both ATP and Mg are counted twice and the Mg atom
adds to the denominator, so ames' scores are not comparable between ATP and ATP+Mg
runs. Here ATP and the ion are measured separately on the final lineage structure
(same contact rule as ames: heavy atoms < 4.5 A, both atoms pLDDT >= 60).

Outputs (in --out, default <first root>/summary_atp)
  runs.csv                one row per run
  conditions.csv          per selection x alphabet x cation: n, median/min/max
  contact_composition.csv amino-acid fractions among ATP-contacting residues
  final_metrics.png       final lineage member of every run
  trajectories.png        lineage trajectories of the ames metrics
"""

import argparse
import ast
import base64
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ames_alphabets import STANDARD_AA, letters_from_key  # noqa: E402

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# CCD atom names of ATP (31 heavy atoms)
ATP_PHOSPHATE = {"PA", "O1A", "O2A", "O3A", "PB", "O1B", "O2B", "O3B", "PG", "O1G", "O2G", "O3G"}
ATP_ADENINE = {"N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"}
ATP_RIBOSE = {"C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"}
ATP_N_HEAVY = len(ATP_PHOSPHATE | ATP_ADENINE | ATP_RIBOSE)

PLOOP_STRICT = re.compile(r"G.{4}GK[ST]")   # Walker A / P-loop consensus GxxxxGK[ST]
PLOOP_KR = re.compile(r"G.{4}G[KR][ST]")    # K->R, for alphabets without Lys


#===================================# STRUCTURES #===================================#

def decompress_structure(b64: str) -> str:
    """Inverse of ames' compress_str: base64(zstd(text))."""
    import zstandard
    return zstandard.decompress(base64.b64decode(b64)).decode("utf-8")


def parse_pdb(text: str) -> pd.DataFrame:
    """ATOM/HETATM records -> one row per heavy atom (ames' hydrogen rule)."""
    rows = []
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        name = line[12:16].strip()
        if name.startswith("H") or re.match(r"\d+H", name):
            continue
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = next((c for c in name if c.isalpha()), "C")
        try:
            bfac = float(line[60:66])
        except ValueError:
            bfac = 0.0
        rows.append((rec, name, line[17:20].strip(), line[21], line[22:27].strip(),
                     float(line[30:38]), float(line[38:46]), float(line[46:54]),
                     bfac, elem.upper()))
    return pd.DataFrame(rows, columns=["rec", "name", "resname", "chain", "resid",
                                       "x", "y", "z", "bfactor", "elem"])


def _contacts(a: pd.DataFrame, b: pd.DataFrame, cutoff: float, min_plddt: float) -> np.ndarray:
    """Boolean matrix [len(a), len(b)]: heavy atoms closer than cutoff, both pLDDT >= min."""
    if a.empty or b.empty:
        return np.zeros((len(a), len(b)), dtype=bool)
    d = np.linalg.norm(a[["x", "y", "z"]].to_numpy()[:, None] - b[["x", "y", "z"]].to_numpy()[None], axis=2)
    ok = (a.bfactor.to_numpy()[:, None] >= min_plddt) & (b.bfactor.to_numpy()[None] >= min_plddt)
    return (d < cutoff) & ok


def _residue_keys(atoms: pd.DataFrame, mask: np.ndarray) -> list:
    sel = atoms[mask]
    return sorted(set(zip(sel.chain, sel.resid, sel.resname)))


def structure_metrics(pdb_text: str, cutoff=4.5, min_plddt=60.0, ion_cutoff=2.8) -> dict:
    """ATP-only and ion-only metrics of one predicted protein + ATP (+ ion) complex."""
    atoms = parse_pdb(pdb_text)
    protein = atoms[atoms.rec == "ATOM"].reset_index(drop=True)
    het = atoms[atoms.rec == "HETATM"]
    atp = het[het.resname == "ATP"].reset_index(drop=True)

    out = {"atp_found": len(atp) > 0, "atp_atoms": len(atp)}
    comp = {aa: 0 for aa in STANDARD_AA}
    out["contact_composition"] = comp
    if protein.empty or atp.empty:
        return out

    c = _contacts(protein, atp, cutoff, min_plddt)
    res = _residue_keys(protein, c.any(axis=1))
    out["atp_contact_res"] = len(res)
    out["atp_lcd"] = len(res) / len(atp)  # = ames' lcd for an ATP-only run
    out["atp_atom_contact_frac"] = float(c.any(axis=0).mean())
    out["atp_plddt"] = float(atp.bfactor.mean())
    out["atp_atoms_unrecognized"] = int((~atp.name.isin(ATP_PHOSPHATE | ATP_ADENINE | ATP_RIBOSE)).sum())
    for aa in (THREE2ONE.get(r[2]) for r in res):
        if aa:
            comp[aa] += 1
    for label, names in (("phosphate", ATP_PHOSPHATE), ("adenine", ATP_ADENINE), ("ribose", ATP_RIBOSE)):
        sub = c[:, atp.name.isin(names).to_numpy()]
        out[f"{label}_contact_res"] = len(_residue_keys(protein, sub.any(axis=1))) if sub.size else 0

    # ions: single-atom HETATM groups other than ATP (Mg, Mn, Ca, ...)
    het_groups = het[het.resname != "ATP"].groupby(["chain", "resid", "resname"]).filter(lambda g: len(g) == 1)
    out["ion"] = ",".join(sorted(set(het_groups.resname))) if len(het_groups) else ""
    if len(het_groups):
        ion = het_groups.reset_index(drop=True)
        ci = _contacts(protein, ion, cutoff, min_plddt)
        out["ion_contact_res"] = len(_residue_keys(protein, ci.any(axis=1)))  # as ames counts it
        # coordination shell: protein O/N and ATP O within ion_cutoff, no pLDDT filter
        cp = _contacts(protein[protein.elem.isin(["O", "N"])], ion, ion_cutoff, 0.0)
        ca = _contacts(atp[atp.elem == "O"], ion, ion_cutoff, 0.0)
        out["ion_coord_protein_atoms"] = int(cp.any(axis=1).sum())
        out["ion_coord_protein_res"] = len(_residue_keys(protein[protein.elem.isin(["O", "N"])], cp.any(axis=1)))
        out["ion_coord_atp_O"] = int(ca.any(axis=1).sum())
        out["ion_coord_number"] = out["ion_coord_protein_atoms"] + out["ion_coord_atp_O"]
    return out


#======================================# RUNS #======================================#

def read_header(log: Path) -> dict:
    """``#--key = value`` lines at the top of progress.log (stops at the first data line)."""
    header = {}
    with open(log) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("#--"):
                key, _, value = line[3:].partition("=")
                header[key.strip()] = value.strip()
    return header


def parse_ligands(text: str) -> list:
    """Header value of ``ligand``: ames logs the Python list (``['ATP', 'MG']``) or ``None``."""
    text = text.strip()
    if text.startswith("["):
        return [str(x) for x in ast.literal_eval(text)]
    return [x.strip() for x in text.split(",") if x.strip() and x.strip() != "None"]


def run_row(run_dir: Path, args) -> dict:
    header = read_header(run_dir / "progress.log")
    alphabet = header["protein_alphabet1"]
    ligands = parse_ligands(header.get("ligand", "None"))
    ions = [x for x in ligands if x != "ATP"]
    neutral = float(header["beta"]) == 0.0 and header.get("annealing") != "True"

    lin = pd.read_csv(run_dir / "lineage.tsv", sep="\t")
    folded = lin[lin["gndx"] >= 1]  # generation 0 holds the unfolded random founders (all metrics 0)
    last, first = lin.iloc[-1], (folded if len(folded) else lin).iloc[0]
    seq = str(last["seq1"])
    letters = letters_from_key(alphabet)

    row = {
        "run": str(run_dir),
        "selection": "neutral" if neutral else "selected",
        "alphabet": "ALL20" if alphabet == "uniform" else alphabet.replace("-raw", ""),
        "cation": ions[0] if ions else "none",
        "ligand": ",".join(ligands),
        "generations": int(last["gndx"]),
        "seq": seq,
        "seq_len": len(seq),
        "alphabet_ok": set(seq) <= set(letters),
        "ploop_strict": bool(PLOOP_STRICT.search(seq)),
        "ploop_KR": bool(PLOOP_KR.search(seq)),
    }
    for col in ("score", "plddt", "ptm", "iptm", "iplddt", "cd", "lcd"):
        row[col] = float(last[col])
        row[f"{col}_first"] = float(first[col])

    try:
        row.update(structure_metrics(decompress_structure(last["structure"]),
                                     args.cutoff, args.min_plddt, args.ion_cutoff))
    except Exception as err:  # simulacrum / corrupted structure
        print(f"WARNING {run_dir}: no structure metrics ({err})", file=sys.stderr)
        row["contact_composition"] = {aa: 0 for aa in STANDARD_AA}
    row["confident_pose"] = bool(row.get("atp_atom_contact_frac", 0) >= args.min_atom_frac
                                 and row.get("atp_plddt", 0) >= args.min_ligand_plddt)
    return row


def alphabet_order(name: str) -> tuple:
    return (len(letters_from_key(name)), name)


#=====================================# FIGURES #=====================================#
# Colours: validated with the dataviz skill's validate_palette.js (--ordinal, light).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, BLUE_DARK, GRAY, GRAY_DARK = "#2a78d6", "#184f95", "#9d9c96", "#6b6a65"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # smallest -> largest alphabet

FINAL_ROWS = [("atp_contact_res", "ATP-contacting residues"),
              ("phosphate_contact_res", "Residues touching\nATP phosphates"),
              ("iptm", "ipTM (ames)"),
              ("score", "ames score")]
TRAJ_ROWS = [("score", "ames score"), ("iptm", "ipTM"), ("lcd", "ligand contact density")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def _from_zero(ax, min_top: float = 0.0) -> None:
    """Counts, ipTM and scores are magnitudes: start at 0 so tiny differences are not inflated.
    Headroom keeps thick marks off the frame; min_top keeps all-zero panels from zooming into noise."""
    ax.set_ylim(0, max(ax.get_ylim()[1] * 1.08, min_top, 1e-9))


def _cation_title(c: str) -> str:
    return "ATP only" if c == "none" else f"ATP + {c}"


def plot_final(runs: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cations = sorted(runs.cation.unique(), key=lambda c: (c != "none", c))
    alphabets = sorted(runs.alphabet.unique(), key=alphabet_order)
    has_neutral = (runs.selection == "neutral").any()
    rows = [r for r in FINAL_ROWS if r[0] in runs.columns]
    rng = np.random.default_rng(0)

    fig, axs = plt.subplots(len(rows), len(cations), figsize=(3.9 * len(cations) + 0.8, 2.1 * len(rows) + 0.9),
                            squeeze=False, sharex=True, facecolor=SURFACE)
    for j, cation in enumerate(cations):
        for i, (metric, label) in enumerate(rows):
            ax = axs[i, j]
            _style(ax)
            for k, alph in enumerate(alphabets):
                for sel, dx, colour, dark in (("selected", -0.19 if has_neutral else 0, BLUE, BLUE_DARK),
                                              ("neutral", 0.19, GRAY, GRAY_DARK)):
                    v = runs[(runs.cation == cation) & (runs.alphabet == alph) & (runs.selection == sel)][metric].dropna()
                    if v.empty:
                        continue
                    ax.scatter(k + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=26, color=colour,
                               edgecolor=SURFACE, linewidth=0.8, zorder=3)
                    ax.hlines(v.median(), k + dx - 0.15, k + dx + 0.15, color=dark, linewidth=2, zorder=4)
            _from_zero(ax, min_top=1.0 if metric.endswith("_res") else 0.0)
            if j == 0:
                ax.set_ylabel(label, fontsize=8.5, color=INK)
            if i == 0:
                ax.set_title(_cation_title(cation), fontsize=10, color=INK, loc="left")
        axs[-1, j].set_xticks(range(len(alphabets)))
        axs[-1, j].set_xticklabels([f"{a}\n({len(letters_from_key(a))} aa)" for a in alphabets], fontsize=7.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=BLUE, markeredgecolor=SURFACE, label="evolved under selection")]
    if has_neutral:
        handles.append(Line2D([], [], marker="o", linestyle="", color=GRAY, markeredgecolor=SURFACE,
                              label="neutral drift (no selection)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False,
               fontsize=8, labelcolor=INK2)
    fig.text(0.01, 0.005, "Final lineage member of each run; points = runs, bar = median.",
             fontsize=7, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_trajectories(run_dirs: pd.DataFrame, path: Path) -> None:
    """Lineage trajectories (selected runs only): faint = runs, bold = median per alphabet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = run_dirs[run_dirs.selection == "selected"]
    if sel.empty:
        return
    cations = sorted(sel.cation.unique(), key=lambda c: (c != "none", c))
    alphabets = sorted(sel.alphabet.unique(), key=alphabet_order)
    colour = {a: ORDINAL[min(i, len(ORDINAL) - 1)] if len(alphabets) <= len(ORDINAL)
              else ORDINAL[round(i * (len(ORDINAL) - 1) / (len(alphabets) - 1))]
              for i, a in enumerate(alphabets)}

    fig, axs = plt.subplots(len(TRAJ_ROWS), len(cations), figsize=(4.4 * len(cations) + 0.8, 2.1 * len(TRAJ_ROWS) + 0.9),
                            squeeze=False, sharex=True, facecolor=SURFACE)
    for j, cation in enumerate(cations):
        for i, (metric, label) in enumerate(TRAJ_ROWS):
            ax = axs[i, j]
            _style(ax)
            for alph in alphabets:
                series = []
                for run in sel[(sel.cation == cation) & (sel.alphabet == alph)].run:
                    lin = pd.read_csv(Path(run) / "lineage.tsv", sep="\t", usecols=["gndx", metric])
                    lin = lin[lin.gndx >= 1]  # generation 0 = unfolded founders, all metrics 0
                    lin = lin.drop_duplicates("gndx", keep="last").set_index("gndx")[metric]
                    series.append(lin)
                    ax.plot(lin.index, lin.values, color=colour[alph], linewidth=0.8, alpha=0.35)
                if not series:
                    continue
                grid = np.arange(1, max(int(s.index.max()) for s in series) + 1)
                # a lineage skips generations without a surviving ancestor: hold the last value
                aligned = pd.concat([s.reindex(grid).ffill().where(grid <= s.index.max()) for s in series], axis=1)
                med = aligned.median(axis=1, skipna=True)
                ax.plot(grid, med.values, color=colour[alph], linewidth=2.2,
                        label=f"{alph} ({len(letters_from_key(alph))} aa)" if (i, j) == (0, 0) else None)
            _from_zero(ax)
            if j == 0:
                ax.set_ylabel(label, fontsize=8.5, color=INK)
            if i == 0:
                ax.set_title(_cation_title(cation), fontsize=10, color=INK, loc="left")
        axs[-1, j].set_xlabel("Generation", fontsize=8.5, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colour[a], linewidth=2.2, label=f"{a} ({len(letters_from_key(a))} aa)")
               for a in alphabets]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=INK2)
    fig.text(0.01, 0.005, "ames metrics along the final sequence's lineage; faint = runs, bold = median. "
             "In ATP+ion runs lcd also counts ion contacts.", fontsize=7, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


#=====================================# SUMMARY #=====================================#

METRICS = ["score", "iptm", "iplddt", "plddt", "lcd", "atp_contact_res", "atp_atom_contact_frac",
           "phosphate_contact_res", "adenine_contact_res", "ribose_contact_res",
           "ion_contact_res", "ion_coord_number"]


def summarize(runs: pd.DataFrame) -> tuple:
    keys = ["selection", "alphabet", "cation"]
    cond = []
    for key, g in runs.groupby(keys):
        row = dict(zip(keys, key))
        row["n_runs"] = len(g)
        for m in METRICS:
            if m in g and g[m].notna().any():
                row[f"{m}_median"], row[f"{m}_min"], row[f"{m}_max"] = g[m].median(), g[m].min(), g[m].max()
        for flag in ("confident_pose", "ploop_strict", "ploop_KR", "alphabet_ok"):
            row[f"frac_{flag}"] = g[flag].mean()
        cond.append(row)
    cond = pd.DataFrame(cond)
    cond["_o"] = cond.alphabet.map(lambda a: alphabet_order(a)[0])
    cond = cond.sort_values(["selection", "cation", "_o"], ascending=[False, True, True]).drop(columns="_o")

    comp = []
    for key, g in runs.groupby(keys):
        total = pd.DataFrame(list(g.contact_composition)).sum()
        n = total.sum()
        comp.append({**dict(zip(keys, key)), "n_contact_res": int(n),
                     **{aa: (total[aa] / n if n else np.nan) for aa in STANDARD_AA}})
    return cond, pd.DataFrame(comp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", type=Path, help="directories searched recursively for lineage.tsv")
    p.add_argument("--out", type=Path, help="output directory [<first root>/summary_atp]")
    p.add_argument("--cutoff", type=float, default=4.5, help="contact distance, A (ames: lig_contact_cutoff)")
    p.add_argument("--min-plddt", type=float, default=60.0, help="atom pLDDT for contacts (ames: lig_contact_min_plddt)")
    p.add_argument("--ion-cutoff", type=float, default=2.8, help="ion coordination distance, A")
    p.add_argument("--min-atom-frac", type=float, default=0.6,
                   help="confident_pose: fraction of ATP atoms in contact with protein (arbitrary default)")
    p.add_argument("--min-ligand-plddt", type=float, default=70.0,
                   help="confident_pose: mean ATP pLDDT (arbitrary default)")
    args = p.parse_args()

    run_dirs = sorted({lin.parent for root in args.roots for lin in root.rglob("lineage.tsv")})
    if not run_dirs:
        sys.exit("no lineage.tsv found; run visualames -l <run>/progress.log first")
    rows = []
    for rd in run_dirs:
        if not (rd / "progress.log").exists():
            print(f"WARNING {rd}: no progress.log next to lineage.tsv, skipped", file=sys.stderr)
            continue
        rows.append(run_row(rd, args))
    runs = pd.DataFrame(rows)

    out = args.out or args.roots[0] / "summary_atp"
    out.mkdir(parents=True, exist_ok=True)
    cond, comp = summarize(runs)
    runs.drop(columns="contact_composition").to_csv(out / "runs.csv", index=False)
    cond.to_csv(out / "conditions.csv", index=False)
    comp.to_csv(out / "contact_composition.csv", index=False)
    plot_final(runs, out / "final_metrics.png")
    plot_trajectories(runs, out / "trajectories.png")

    bad = runs[~runs.alphabet_ok]
    if len(bad):
        print(f"WARNING: {len(bad)} run(s) have final residues outside their alphabet:\n{bad.run.to_string(index=False)}")
    if "atp_atoms_unrecognized" in runs and runs.atp_atoms_unrecognized.fillna(0).gt(0).any():
        print("WARNING: ATP atom names differ from the CCD names assumed for phosphate/adenine/ribose splits; "
              "check atp_atoms_unrecognized in runs.csv")
    show = ["selection", "alphabet", "cation", "n_runs", "score_median", "iptm_median",
            "atp_contact_res_median", "phosphate_contact_res_median", "frac_confident_pose"]
    print(cond[[c for c in show if c in cond]].round(3).to_string(index=False))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
