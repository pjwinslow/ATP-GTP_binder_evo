#!/usr/bin/env python3
"""End-to-end test with the mock fold engine (no GPU): launcher -> ames -> visualames -> summarizer.

    python tests/run_mock_e2e.py [--keep DIR]

Cross-checks the summarizer's independently implemented ATP contact count against the
``lcd`` that ames' C++ code logged for the same structure.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ATP_ATOMS = 31

RUNS = [  # (root, alphabet, condition dir, ligand, extra args)
    ("sel", "GADVP", "ATP", "ATP", []),
    ("sel", "GADVP", "ATP_MG", "ATP,MG", []),
    ("sel", "GADVPSELT", "ATP_MG", "ATP,MG", []),
    ("neutral", "GADVP", "ATP_MG", "ATP,MG", ["-b0", "0"]),
]


def run(cmd, env):
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{res.stdout[-2000:]}\n{res.stderr[-3000:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, help="write outputs here instead of a temp dir")
    args = ap.parse_args()
    out = args.keep or Path(tempfile.mkdtemp(prefix="ames_atp_e2e_"))
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MOCK_SEED": "7"}
    bindir = Path(sys.executable).parent

    for root, alphabet, cond, ligand, extra in RUNS:
        rd = out / root / alphabet / cond / "run01"
        sel = extra or ["-b0", "3"]
        run([sys.executable, HERE / "mock_ames.py", "--alphabet", alphabet,
             "--iseq1", "protein:randoms:40:evolv", "--seq1_rate", "1", "--ligand", ligand,
             "-pm1", "npm", "--seq1_max_len", "160",
             "--helix_len_penalty", "1000", "--strand_len_penalty", "1000",  # mock backbone is one long helix
             "-ps", "16", "-ng", "30", *sel, "--engine", "esmfold2", "-o", rd], env)
        run([bindir / "visualames", "-l", rd / "progress.log"], env)
        print("ran", rd.relative_to(out))

    summary = out / "summary"
    run([sys.executable, HERE.parent / "summarize_matrix.py", out / "sel", out / "neutral", "--out", summary], env)
    runs = pd.read_csv(summary / "runs.csv")
    print(runs[["selection", "alphabet", "cation", "score_first", "score", "lcd", "atp_lcd", "atp_contact_res",
                "ion_contact_res", "ion_coord_number"]].round(3).to_string(index=False))

    assert len(runs) == len(RUNS)
    assert runs.alphabet_ok.all(), "a final sequence left its alphabet"
    assert (runs.atp_atoms == ATP_ATOMS).all() and (runs.atp_atoms_unrecognized == 0).all()
    for _, r in runs.iterrows():
        if r.cation == "none":  # ames lcd = contacting pairs / ligand atoms; here ATP is the only ligand
            assert abs(r.lcd - r.atp_lcd) < 6e-4, ("lcd mismatch", r.run, r.lcd, r.atp_lcd)
        else:                   # ligand = ATP (31 atoms) + 1 ion; residues touching both are counted twice
            pairs = r.atp_contact_res + r.ion_contact_res
            assert round(r.lcd * (ATP_ATOMS + 1)) == pairs, ("lcd mismatch", r.run, r.lcd, pairs)
    selected = runs[runs.selection == "selected"]
    # selection acts on ames' composite score, not on ipTM alone
    assert (selected.score > selected.score_first).all(), "selection did not raise the score along the lineage"
    assert set(runs.selection) == {"selected", "neutral"}
    for f in ("runs.csv", "conditions.csv", "contact_composition.csv", "final_metrics.png", "trajectories.png"):
        assert (summary / f).stat().st_size > 0, f
    print(f"\nOK  (outputs in {out})")


if __name__ == "__main__":
    main()
