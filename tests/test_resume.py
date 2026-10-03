"""Resume tests: log truncation, and kill-then-resume of a real ames run with the mock fold engine.

    python tests/test_resume.py        # ~40 s; needs ames installed (python >= 3.12 env)
"""
import hashlib
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from ames_resume import patch_simulator, truncate_log  # noqa: E402

PS, NG = 16, 30
B0, BT, ANN_S, ANN_E = 1.0, 5.0, 4, 24
STEP = round((BT - B0) / (ANN_E - ANN_S), 3)


def _expected_beta(gen: int) -> float:
    """Selection strength of rows created in generation `gen`, as ames' loop updates it."""
    beta = B0
    for i in range(1, gen + 1):
        if ANN_S <= i <= ANN_E:
            beta = round(beta + STEP, 3)
    return beta


def _ames_cmd(outdir: Path, ligand="GTP,MG", extra=(), ng=NG):
    return [sys.executable, str(HERE / "mock_ames.py"), "--alphabet", "GADVP", "--resume", *extra,
            "--iseq1", "protein:randoms:30:evolv", "--ligand", ligand,
            "--helix_len_penalty", "1000", "--strand_len_penalty", "1000",
            "-ps", str(PS), "-ng", str(ng), "-ckpi", "1",
            "-b0", str(B0), "-ann", "-bt", str(BT), "-ann_s", str(ANN_S), "-ann_e", str(ANN_E),
            "--engine", "esmfold2", "-o", str(outdir)]


def _checkpoint_gen(path: Path):
    """Generation stored in progress.ckp (first data row), or None if there is no checkpoint yet."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or line.startswith("gndx\t"):
                    continue
                return int(line.split("\t", 1)[0])
    except (FileNotFoundError, ValueError):
        pass
    return None


def test_truncate_log():
    header = "#--beta = 1\n#\ngndx\tid\tx\n"
    rows = "".join(f"{g}\tg{g}s{n}\tv\n" for g in range(5) for n in range(3))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "progress.log"
        # generations 0-4 complete, then a half-written row of generation 5
        path.write_text(header + rows + "5\tg5s0\tv\n5\tg5s1\tincompl")
        truncate_log(str(path), 2)
        kept = path.read_text()
        assert kept == header + "".join(f"{g}\tg{g}s{n}\tv\n" for g in range(3) for n in range(3))
        truncate_log(str(path), 2)  # idempotent
        assert path.read_text() == kept
        # a checkpoint newer than the log keeps everything that is there
        truncate_log(str(path), 99)
        assert path.read_text() == kept


def test_patch_refuses_unknown_ames_source():
    def fold_evolution_simulator():  # a stand-in that lacks the lines the patch rewrites
        return None
    try:
        patch_simulator(SimpleNamespace(fold_evolution_simulator=fold_evolution_simulator, __dict__={}))
    except RuntimeError as err:
        assert "dd39c57" in str(err)
    else:
        raise AssertionError("patched an unrecognised simulator")


def test_kill_and_resume():
    with tempfile.TemporaryDirectory(prefix="ames_resume_") as tmp:
        out = Path(tmp) / "run01"
        log, ckp = out / "progress.log", out / "progress.ckp"

        # 1. start, then SIGKILL (as a preempted job would be) once the checkpoint is inside the annealing window
        proc = subprocess.Popen(_ames_cmd(out), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while (_checkpoint_gen(ckp) or 0) < 8:
            assert proc.poll() is None and time.time() < deadline, "run ended before the checkpoint reached gen 8"
            time.sleep(0.05)
        proc.kill()
        proc.wait()
        killed_at = _checkpoint_gen(ckp)
        assert killed_at < NG - 1, "run finished before it could be killed; raise NG"
        (out / "progress.ckp.tmp").write_text("garbage from a save that was interrupted")  # must be ignored

        # 2. resume to the end
        res = subprocess.run(_ames_cmd(out), capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-2000:]
        assert f"#RESUMING from the checkpoint at generation {killed_at}" in res.stdout
        assert _checkpoint_gen(ckp) == NG - 1

        # 3. the log holds every generation exactly once, with a single header
        lines = log.read_text().splitlines()
        assert sum(l.startswith("gndx\t") for l in lines) == 1
        assert all(l.startswith("#") for l in lines[: next(i for i, l in enumerate(lines) if l.startswith("gndx\t"))])
        df = pd.read_csv(log, sep="\t", comment="#")
        assert Counter(df.gndx) == {g: PS for g in range(NG)}, "generation missing or duplicated after resume"

        # 4. the annealing schedule continued across the break: every row carries the beta of its birth generation
        born = df[df.id.str.startswith("g")]
        gen_born = born.id.str.extract(r"^g(\d+)s")[0].astype(int)
        assert (born.beta - gen_born.map(_expected_beta)).abs().max() < 1e-9
        assert gen_born.max() > killed_at, "no rows were created after the resume"
        # ames' annealing window is inclusive, so beta ends one step past the nominal target
        assert abs(born.beta.max() - _expected_beta(NG - 1)) < 1e-9

        # 5. the standard analysis still works on the stitched log
        subprocess.run([str(Path(sys.executable).parent / "visualames"), "-l", str(log)],
                       check=True, capture_output=True)
        lineage = pd.read_csv(out / "lineage.tsv", sep="\t", usecols=["gndx"])
        assert lineage.gndx.iloc[-1] == NG - 1

        # 6. resuming a finished run changes nothing and leaves no *_backup_* directory
        before = hashlib.sha256(log.read_bytes()).hexdigest()
        res = subprocess.run(_ames_cmd(out), capture_output=True, text=True)
        assert res.returncode == 0 and "0 generation(s) left" in res.stdout, res.stdout[-500:]
        assert hashlib.sha256(log.read_bytes()).hexdigest() == before
        assert not [p for p in Path(tmp).iterdir() if "backup" in p.name]

        # 7. a checkpoint from a different condition is refused instead of mixed in
        res = subprocess.run(_ames_cmd(out, ligand="ATP"), capture_output=True, text=True)
        assert res.returncode != 0 and "cannot resume" in res.stderr, res.stderr[-800:]
        assert hashlib.sha256(log.read_bytes()).hexdigest() == before


def test_resume_keeps_the_scoring_mode():
    """--score-nucleotide-only changes what selection acts on, so a run cannot continue in the other mode."""
    with tempfile.TemporaryDirectory(prefix="ames_scoremode_") as tmp:
        out = Path(tmp) / "run01"
        nuc_only = ("--score-nucleotide-only",)
        res = subprocess.run(_ames_cmd(out, extra=nuc_only, ng=6), capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-2000:]
        assert "#scoring: ligand contacts and interface pLDDT on chain B only (ames default: B,C)" in res.stderr
        assert re.search(r"#--ligand_chains\s+= B\n", (out / "progress.log").read_text())
        before = hashlib.sha256((out / "progress.log").read_bytes()).hexdigest()

        # the same mode continues (nothing left to do) ...
        res = subprocess.run(_ames_cmd(out, extra=nuc_only, ng=6), capture_output=True, text=True)
        assert res.returncode == 0 and "0 generation(s) left" in res.stdout, res.stderr[-800:]
        # ... the other one is refused, in both directions, and the log is not touched
        res = subprocess.run(_ames_cmd(out, ng=6), capture_output=True, text=True)
        assert res.returncode != 0 and "cannot resume" in res.stderr and "--score-nucleotide-only" in res.stderr, res.stderr[-800:]
        assert hashlib.sha256((out / "progress.log").read_bytes()).hexdigest() == before

        pooled = Path(tmp) / "run02"
        res = subprocess.run(_ames_cmd(pooled, ng=6), capture_output=True, text=True)
        assert res.returncode == 0 and "#scoring" not in res.stderr, res.stderr[-800:]
        assert re.search(r"#--ligand_chains\s+= B,C\n", (pooled / "progress.log").read_text())
        res = subprocess.run(_ames_cmd(pooled, extra=nuc_only, ng=6), capture_output=True, text=True)
        assert res.returncode != 0 and "cannot resume" in res.stderr, res.stderr[-800:]


def test_score_nucleotide_only_needs_a_ligand():
    with tempfile.TemporaryDirectory(prefix="ames_noligand_") as tmp:
        cmd = [sys.executable, str(HERE / "mock_ames.py"), "--alphabet", "GADVP", "--score-nucleotide-only",
               "--iseq1", "protein:randoms:30:evolv", "-ps", "4", "-ng", "3", "--engine", "esmfold2",
               "-o", str(Path(tmp) / "run01")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0 and "--score-nucleotide-only needs --ligand" in res.stderr, res.stderr[-800:]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
