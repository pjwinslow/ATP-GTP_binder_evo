"""Checkpoint resume for ames runs.

ames saves the selected population to ``progress.ckp`` every ``checkpoint_interval``
generations but never reads it back: ``ames`` always starts from generation 0 and
moves an existing output directory aside. On a preemptible GPU partition a requeued
job would therefore start over. ``prepare()`` and ``patch_simulator()`` make
``run_ames_alphabet.py --resume`` continue from the checkpoint instead; ames itself is
not modified.

What a resume does
  * keeps the existing output directory (ames would move it aside);
  * writes checkpoints atomically, so a kill during a save cannot corrupt them;
  * loads ``progress.ckp`` as the starting population and continues at the next
    generation, appending to ``progress.log`` after cutting it back to the checkpointed
    generation (rows of a later, unfinished generation and a half-written last line go);
  * replays the annealing schedule so the selection strength (beta) is what it would
    have been without the interruption;
  * refuses to continue if the checkpoint was made with a different alphabet, ligand
    or population size.
Without a checkpoint it behaves exactly like stock ames.

Not restored: ames' memo of already-folded sequences (rebuilt from the checkpoint
population only, so a few sequences may be folded twice) and the random-number state.

The patch rewrites the start of ames' ``fold_evolution_simulator`` from its source and
checks that the expected lines are present, so a different ames version fails with an
error instead of silently running without resume. Written against ames commit
dd39c57 (``pip install git+https://github.com/sahakyanhk/ames@dd39c57``).
"""
import ast
import inspect
import os
from pathlib import Path

import pandas as pd

TESTED_AMES = "dd39c57"

# Lines of ames.fold_evolution_simulator that the patch relies on (each must occur once).
_PREAMBLE_START = "    loghead = generate_loghead(args)\n"
_PREAMBLE_END = "    incomplete_generations = 0\n"
_LOOP = "    for gen_i in range(1, args.num_generations):\n"

# Header values that must match between a checkpoint and the run that resumes it.
_MUST_MATCH = ("protein_alphabet1", "pop_size")


def prepare() -> None:
    """Call before ``import ames.ames``: keep an existing output dir, save checkpoints atomically."""
    import amestools  # ames' flat module, the same object ames.py imports from

    def keep_output_dir(directory_path, *args, **kwargs):
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    def save_checkpoint_atomic(ckp_gen, args):
        final = os.path.join(args.outpath, args.ckp)
        tmp = final + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(amestools.generate_loghead(args))
        ckp_gen.to_csv(tmp, mode="a", index=False, header=True, sep="\t")
        os.replace(tmp, final)

    amestools.backup_output = keep_output_dir
    amestools.save_checkpoint = save_checkpoint_atomic


def truncate_log(logpath: str, last_gen: int) -> int:
    """Cut ``progress.log`` after the last complete row with gndx <= last_gen; returns the new size."""
    keep = pos = 0
    with open(logpath, "rb") as fh:
        for raw in fh:
            text = raw.decode("utf-8", "replace")
            pos += len(raw)
            if text.startswith("#") or text.startswith("gndx\t") or not text.strip():
                keep = pos  # parameter header and column header
                continue
            try:
                gndx = int(text.split("\t", 1)[0])
            except ValueError:
                break
            if gndx > last_gen or not raw.endswith(b"\n"):
                break
            keep = pos
    with open(logpath, "r+b") as fh:
        fh.truncate(keep)
    return keep


def read_checkpoint(path: str) -> pd.DataFrame:
    """The selected population saved by ames, with ``sequence_data`` turned back into dicts."""
    population = pd.read_csv(path, sep="\t", comment="#", index_col=None)
    population["sequence_data"] = population["sequence_data"].apply(ast.literal_eval)
    return population


def _check_compatible(header: dict, args) -> None:
    for key in _MUST_MATCH:
        if header.get(key) != getattr(args, key):
            raise RuntimeError(f"cannot resume: checkpoint has {key}={header.get(key)!r}, "
                               f"this run has {getattr(args, key)!r}. Use a new output directory.")
    if str(header.get("ligand_chains")) != str(getattr(args, "ligand_chains", None)):
        raise RuntimeError(f"cannot resume: checkpoint scored ligand chains {header.get('ligand_chains')!r}, "
                           f"this run scores {str(getattr(args, 'ligand_chains', None))!r} "
                           f"(--score-nucleotide-only differs). Use a new output directory.")
    if str(header.get("ligand")) != str(args.ligand):
        raise RuntimeError(f"cannot resume: checkpoint has ligand={header.get('ligand')!r}, "
                           f"this run has {str(args.ligand)!r}. Use a new output directory.")


def _make_resume_or_start(m):
    """Build the replacement for the start of the simulator, bound to ames' module ``m``."""

    def resume_or_start(logpath, evolver1, evolver2):
        args = m.args
        ckp = os.path.join(args.outpath, args.ckp)
        if not os.path.exists(ckp):  # nothing to resume: identical to stock ames
            loghead = m.generate_loghead(args)
            print(loghead)
            with open(logpath, "w") as fh:
                fh.write(loghead)
            init_gen = m.create_init_gen(evolver1, evolver2, args)
            init_gen.to_csv(logpath, mode="a", index=False, header=True, sep="\t")
            return init_gen, 0

        from amestools import read_header
        _check_compatible(read_header(ckp), args)
        init_gen = read_checkpoint(ckp)
        start_gen = int(init_gen["gndx"].iloc[0])
        size = truncate_log(logpath, start_gen)
        for i in range(1, start_gen + 1):  # replay annealing exactly as the loop applied it
            if args.annealing and args.annealing_start <= i <= args.annealing_end:
                m.update_beta(args)
        left = max(args.num_generations - 1 - start_gen, 0)
        print(f"#RESUMING from the checkpoint at generation {start_gen}: {left} generation(s) left, "
              f"beta={args.beta}, progress.log cut back to {size} bytes")
        return init_gen, start_gen

    return resume_or_start


def patch_simulator(ames_module) -> None:
    """Call after ``import ames.ames`` and before ``main()``: make the simulator resumable."""
    source = inspect.getsource(ames_module.fold_evolution_simulator)
    for anchor in (_PREAMBLE_START, _PREAMBLE_END, _LOOP):
        if source.count(anchor) != 1:
            raise RuntimeError(
                f"--resume: ames' fold_evolution_simulator does not look like the version this patch "
                f"was written for (commit {TESTED_AMES}); missing line: {anchor.strip()!r}. "
                f"Install that commit: pip install git+https://github.com/sahakyanhk/ames@{TESTED_AMES}")
    head, rest = source.split(_PREAMBLE_START)
    _, tail = rest.split(_PREAMBLE_END)
    patched = (head
               + "    init_gen, start_gen = _resume_or_start(logpath, evolver1, evolver2)\n"
               + "    sequence_lookup = build_sequence_lookup(init_gen)\n"
               + "    print_genlog(init_gen, args)\n\n"
               + _PREAMBLE_END + tail)
    patched = patched.replace(_LOOP, "    for gen_i in range(start_gen + 1, args.num_generations):\n")
    namespace = ames_module.__dict__
    namespace["_resume_or_start"] = _make_resume_or_start(ames_module)
    exec(compile(patched, "<ames fold_evolution_simulator, resumable>", "exec"), namespace)
