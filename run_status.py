#!/usr/bin/env python3
"""Progress of the runs listed in the manifest that submit_matrix.sh wrote.

    run_status.py OUTROOT/manifest.tsv

One line per run (array index, directory, state) and the indices that are not finished,
ready to paste into ``submit_matrix.sh --submit <indices>`` (resubmitted runs resume from
their checkpoint). Standard library only, so it runs with any python3.

States: done (finished, DONE marker) · partial G/N (checkpoint at generation G of N-1) ·
started (log but no checkpoint yet) · not started.
"""
import csv
import sys
from pathlib import Path


def checkpoint_progress(ckp: Path):
    """(generation, total generations) from a progress.ckp, or None."""
    gen = total = None
    try:
        with open(ckp) as fh:
            for line in fh:
                if line.startswith("#--num_generations"):
                    total = int(line.split("=", 1)[1])
                elif line.startswith("#") or line.startswith("gndx\t"):
                    continue
                else:
                    gen = int(line.split("\t", 1)[0])
                    break
    except (FileNotFoundError, ValueError):
        return None
    return None if gen is None else (gen, total)


def run_state(outdir: Path) -> str:
    if (outdir / "DONE").exists():
        return "done"
    progress = checkpoint_progress(outdir / "progress.ckp")
    if progress:
        gen, total = progress
        return f"partial {gen}/{total - 1}" if total else f"partial {gen}"
    if (outdir / "progress.log").exists():
        return "started"
    return "not started"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    manifest = Path(sys.argv[1])
    if not manifest.exists():
        print(f"no manifest at {manifest}: nothing has been submitted with these settings", file=sys.stderr)
        return 1
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    root = manifest.parent
    unfinished, counts = [], {}
    print(f"{'idx':>4}  {'run':<44} state")
    for row in rows:
        outdir = Path(row["outdir"])
        state = run_state(outdir)
        kind = "not started" if state.startswith("not") else state.split()[0]
        counts[kind] = counts.get(kind, 0) + 1
        if state != "done":
            unfinished.append(row["idx"])
        try:
            shown = outdir.relative_to(root)
        except ValueError:
            shown = outdir
        print(f"{row['idx']:>4}  {str(shown):<44} {state}")
    print("\n" + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    if unfinished:
        print(f"unfinished indices: {','.join(unfinished)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
