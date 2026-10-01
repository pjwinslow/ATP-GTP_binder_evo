#!/usr/bin/env python3
"""End-to-end test with the mock fold engine (no GPU): launcher -> ames -> visualames -> summarizer.

    python tests/run_mock_e2e.py [--keep DIR]

Runs ATP and GTP conditions, with and without Mg, and cross-checks the summarizer's
independently implemented nucleotide contact count against the ``lcd`` that ames' C++
code logged for the same structure.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from summarize_matrix import N_HEAVY  # noqa: E402

RUNS = [  # (root, alphabet, condition dir, ligand, selection args)
    ("sel", "GADVP", "ATP", "ATP", ["-b0", "3"]),
    ("sel", "GADVP", "ATP_MG", "ATP,MG", ["-b0", "3"]),
    ("sel", "GADVPSELT", "ATP_MG", "ATP,MG", ["-b0", "3"]),
    ("sel", "GADVP", "GTP", "GTP", ["-b0", "3"]),
    ("sel", "GADVP", "GTP_MG", "GTP,MG", ["-b0", "3"]),
    ("sel", "GADVPSELT", "GTP_MG", "GTP,MG", ["-b0", "3"]),
    ("neutral", "GADVP", "ATP_MG", "ATP,MG", ["-b0", "0"]),
    ("neutral", "GADVP", "GTP_MG", "GTP,MG", ["-b0", "0"]),
]


def run(cmd, env):
    res = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{res.stdout[-2000:]}\n{res.stderr[-3000:]}")


def one_run(out: Path, spec, env) -> str:
    root, alphabet, cond, ligand, sel = spec
    rd = out / root / alphabet / cond / "run01"
    run([sys.executable, HERE / "mock_ames.py", "--alphabet", alphabet,
         "--iseq1", "protein:randoms:40:evolv", "--seq1_rate", "1", "--ligand", ligand,
         "-pm1", "npm", "--seq1_max_len", "160",
         "--helix_len_penalty", "1000", "--strand_len_penalty", "1000",  # mock backbone is one long helix
         "-ps", "16", "-ng", "30", *sel, "--engine", "esmfold2", "-o", rd], env)
    run([Path(sys.executable).parent / "visualames", "-l", rd / "progress.log"], env)
    return str(rd.relative_to(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, help="write outputs here instead of a temp dir")
    args = ap.parse_args()
    out = args.keep or Path(tempfile.mkdtemp(prefix="ames_nuc_e2e_"))
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MOCK_SEED": "7"}

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        for name in pool.map(lambda spec: one_run(out, spec, env), RUNS):
            print("ran", name)

    summary = out / "summary"
    run([sys.executable, HERE.parent / "summarize_matrix.py", out / "sel", out / "neutral", "--out", summary], env)
    runs = pd.read_csv(summary / "runs.csv")
    print(runs[["selection", "nucleotide", "alphabet", "cation", "score_first", "score", "lcd", "nuc_lcd",
                "nuc_contact_res", "base_contact_res", "ion_contact_res", "ion_coord_number"]]
          .round(3).to_string(index=False))

    assert len(runs) == len(RUNS)
    assert set(runs.nucleotide) == {"ATP", "GTP"} and set(runs.cation) == {"none", "MG"}
    assert runs.alphabet_ok.all(), "a final sequence left its alphabet"
    assert (runs.nuc_atoms == runs.nucleotide.map(N_HEAVY)).all(), "wrong nucleotide atom count"
    assert (runs.nuc_atoms_unrecognized == 0).all()
    for _, r in runs.iterrows():
        n = N_HEAVY[r.nucleotide]
        if r.cation == "none":  # ames lcd = contacting pairs / ligand atoms; the nucleotide is the only ligand
            assert abs(r.lcd - r.nuc_lcd) < 6e-4, ("lcd mismatch", r.run, r.lcd, r.nuc_lcd)
        else:                   # ligand = nucleotide + 1 ion; residues touching both are counted twice
            pairs = r.nuc_contact_res + r.ion_contact_res
            assert round(r.lcd * (n + 1)) == pairs, ("lcd mismatch", r.run, r.lcd, pairs)
    # Selection acts on ames' composite score (not on ipTM alone). ames is not deterministic even with
    # seeds (result threads append rows in arbitrary order), so test the group, not each run: the
    # selected runs must beat neutral drift and gain score along their lineages on average.
    assert set(runs.selection) == {"selected", "neutral"}
    selected, neutral = runs[runs.selection == "selected"], runs[runs.selection == "neutral"]
    assert selected.score.median() > neutral.score.median(), "selected runs do not outscore neutral drift"
    assert (selected.score - selected.score_first).median() > 0, "selection did not raise the score along lineages"
    for f in ("runs.csv", "conditions.csv", "contact_composition.csv", "final_metrics.png", "trajectories.png"):
        assert (summary / f).stat().st_size > 0, f
    print(f"\nOK  (outputs in {out})")


if __name__ == "__main__":
    main()
