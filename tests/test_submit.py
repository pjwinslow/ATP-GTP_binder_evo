"""submit_matrix.sh with a fake sbatch: directives, manifest, guards, and the job body itself.

    python tests/test_submit.py        # ~30 s; needs ames installed (python >= 3.12 env)

The job body is run for two array tasks with the real launcher/ames/visualames but the mock
fold engine (and without the GPU check), then --status and the skip-if-DONE path are checked.
"""
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUBMIT = ROOT / "submit_matrix.sh"

FAKE_SBATCH = """#!/bin/bash
# fake sbatch: store the job script it receives on stdin
dir="$(dirname "$0")/captured"; mkdir -p "$dir"
n=$(ls "$dir" | wc -l)
cat > "$dir/job$n.sh"
echo "Submitted batch job $n"
"""


def _setup(tmp: Path):
    bindir = tmp / "bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(FAKE_SBATCH)
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IXUSR)
    # `python` and `visualames` inside the job body must be this interpreter's
    path = f"{bindir}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    return bindir, path


def _run(args, env_extra, path, cwd):
    # blank the conda variables so the result does not depend on the shell running the tests
    env = {**os.environ, "CONDA_PREFIX": "", "CONDA_EXE": "", "CONDA_ENV": "", "ESMCFOLD_CCD_PATH": "", "CCD_PATH": "",
           "PATH": path, "WORKDIR": str(cwd), **env_extra}
    return subprocess.run(["bash", str(SUBMIT), *args], env=env, capture_output=True, text=True, cwd=cwd)


def _job_scripts(bindir: Path):
    return sorted((bindir / "captured").glob("job*.sh"), key=lambda p: int(p.stem[3:]))


def test_submit_and_run_job_body():
    with tempfile.TemporaryDirectory(prefix="ames_submit_") as t:
        tmp = Path(t)
        bindir, path = _setup(tmp)
        work = tmp / "work"
        work.mkdir()
        base = {"OUTROOT": str(tmp / "out"), "ALPHABETS": "GADVP", "CATIONS": "none MG", "REPS": "1",
                "SMOKE": "1"}  # 2 nucleotides x 1 alphabet x 2 cation conditions = 4 runs
        env = {**base, "ENV_ACTIVATE": "true"}

        # plan only: prints, submits nothing
        res = _run([], env, path, work)
        assert res.returncode == 0 and "4 runs planned" in res.stdout and "Nothing submitted" in res.stdout, res.stdout
        assert not (bindir / "captured").exists()
        assert "GTP,MG" in res.stdout and "ATP" in res.stdout

        # bad input is rejected
        res = _run([], {**env, "NUCLEOTIDES": "CTP"}, path, work)
        assert res.returncode != 0 and "ATP and GTP only" in res.stderr
        assert _run(["--bogus"], env, path, work).returncode != 0

        # submit: one array job with the settings of the user's working MSI script
        res = _run(["--submit"], env, path, work)
        assert res.returncode == 0 and "Submitting 4 jobs (array 0-3%25)" in res.stdout, res.stdout + res.stderr
        job = _job_scripts(bindir)[0].read_text()
        for expected in ("#SBATCH --partition=preempt-gpu", "#SBATCH --gres=gpu:1", "#SBATCH --mem=40G",
                         "#SBATCH --time=01:00:00",  # SMOKE; the real default is 24:00:00
                         "#SBATCH --array=0-3%25", "#SBATCH --requeue", "#SBATCH --open-mode=append",
                         "#SBATCH --job-name=nuc_evo_smoke", "--resume"):
            assert expected in job, expected
        assert "--mail-" not in job and "--account" not in job and "--cpus-per-task" not in job
        assert "export PYTHONUNBUFFERED=1" in job and "export PYTHONNOUSERSITE=1" in job
        # jobs run offline (the hub rate-limits shared IPs), after check_env has confirmed the cache is complete
        assert "export HF_HUB_OFFLINE=1" in job and "ESMCFOLD_CCD_PATH" not in job
        assert job.index("true") < job.index("set -euo pipefail"), "strict mode must come after the env activation"

        manifest = (Path(base["OUTROOT"] + "_smoke") / "manifest.tsv").read_text().splitlines()
        assert manifest[0].split("\t")[:3] == ["idx", "nucleotide", "alphabet"] and len(manifest) == 5
        assert [row.split("\t")[5] for row in manifest[1:]] == ["ATP", "ATP,MG", "GTP", "GTP,MG"]

        # defaults: the conda activation from the user's script, optional mail/account/cpus
        res = _run(["--submit"], {k: v for k, v in env.items() if k != "ENV_ACTIVATE"} | {
            "OUTROOT": str(tmp / "out2"), "MAIL_USER": "me@example.org", "ACCOUNT": "grp", "CPUS": "4",
            "TIME": "12:00:00"}, path, work)
        assert res.returncode == 0, res.stderr
        job2 = _job_scripts(bindir)[1].read_text()
        assert ("source /common/software/install/migrated/anaconda/python3-2020.07-mamba/etc/profile.d/conda.sh "
                "&& conda activate esmfold2") in job2
        for expected in ("#SBATCH --mail-user=me@example.org", "#SBATCH --mail-type=END,FAIL",
                         "#SBATCH --account=grp", "#SBATCH --cpus-per-task=4"):
            assert expected in job2, expected

        # HF_OFFLINE=0 allows downloads; CCD_PATH hands the jobs a local ccd.pkl (which must exist)
        ccd = tmp / "ccd.pkl"
        ccd.write_text("")
        res = _run(["--submit"], {**env, "HF_OFFLINE": "0", "CCD_PATH": str(ccd)}, path, work)
        assert res.returncode == 0, res.stderr
        online = _job_scripts(bindir)[-1].read_text()
        assert "HF_HUB_OFFLINE" not in online and f"export ESMCFOLD_CCD_PATH={ccd}\n" in online

        # ESMCFOLD_CCD_PATH in the submitting shell is picked up, and the plan shows which file is used
        res = _run([], {**env, "ESMCFOLD_CCD_PATH": str(ccd)}, path, work)
        assert f"ccd.pkl:     {ccd}" in res.stdout, res.stdout
        assert "from the Hugging Face cache" in _run([], env, path, work).stdout

        # a CCD_PATH that is not a file is refused before anything is submitted
        n_before = len(_job_scripts(bindir))
        res = _run(["--submit"], {**env, "CCD_PATH": str(tmp / "missing.pkl")}, path, work)
        assert res.returncode != 0 and "is not a file" in res.stderr and len(_job_scripts(bindir)) == n_before

        # a path with a space must still be a single valid assignment in the job script
        spaced_dir = tmp / "my files"
        spaced_dir.mkdir()
        spaced_ccd = spaced_dir / "ccd.pkl"
        spaced_ccd.write_text("")
        res = _run(["--submit"], {**env, "CCD_PATH": str(spaced_ccd)}, path, work)
        assert res.returncode == 0, res.stderr
        spaced = _job_scripts(bindir)[-1].read_text()
        line = next(l for l in spaced.splitlines() if l.startswith("export ESMCFOLD_CCD_PATH="))
        assert "export HF_HUB_OFFLINE=1" in spaced
        assert subprocess.run(["bash", "-c", f'{line}; printf %s "$ESMCFOLD_CCD_PATH"'],
                              capture_output=True, text=True).stdout == str(spaced_ccd)
        for extra in _job_scripts(bindir)[-2:]:
            extra.unlink()  # keep the numbering below

        # resubmitting specific indices; changed settings are refused instead of shifting the indices
        res = _run(["--submit", "1,2"], env, path, work)
        assert "Resubmitting specific jobs: 1,2" in res.stdout
        assert "#SBATCH --array=1,2" in _job_scripts(bindir)[2].read_text()
        res = _run(["--submit"], {**env, "ALPHABETS": "GADVP ALL20"}, path, work)  # (SMOKE pins REPS=1)
        assert res.returncode != 0 and "differs from the matrix" in res.stderr

        # --status before anything ran
        res = _run(["--status"], env, path, work)
        assert res.returncode == 0 and "4 not started" in res.stdout and "unfinished indices: 0,1,2,3" in res.stdout, res.stdout

        # --status can be pointed at the output directory (or manifest) of runs submitted elsewhere
        out_dir = Path(base["OUTROOT"] + "_smoke")
        elsewhere = {**env, "OUTROOT": str(tmp / "somewhere_else")}
        assert "no manifest" in _run(["--status"], elsewhere, path, work).stderr
        for target in (str(out_dir), str(out_dir) + "/", str(out_dir / "manifest.tsv")):
            res = _run(["--status", target], elsewhere, path, work)
            assert res.returncode == 0 and "4 not started" in res.stdout, (target, res.stdout, res.stderr)
        res = _run(["--status", str(tmp / "nope")], elsewhere, path, work)
        assert res.returncode != 0 and "Pass the directory the runs were submitted to" in res.stderr

        # run the job body for tasks 0 (ATP) and 3 (GTP+MG) with the mock fold engine
        body = _job_scripts(bindir)[0].read_text()
        body = "\n".join(line for line in body.splitlines() if "check_env.py" not in line)  # no GPU here
        body = body.replace("/run_ames_alphabet.py", "/tests/mock_ames.py")
        script = tmp / "job_under_test.sh"
        script.write_text(body)
        for task in (0, 3):
            res = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=work,
                                 env={**os.environ, "PATH": path, "SLURM_ARRAY_TASK_ID": str(task),
                                      "SLURM_JOB_ID": "42"})
            assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-2500:]
            assert f"array task {task}" in res.stdout and "finished successfully" in res.stdout
        out = Path(base["OUTROOT"] + "_smoke")
        assert (out / "GADVP/ATP/run01/DONE").exists() and (out / "GADVP/GTP_MG/run01/DONE").exists()
        assert (out / "GADVP/GTP_MG/run01/lineage.tsv").exists()
        header = (out / "GADVP/GTP_MG/run01/progress.log").read_text().split("gndx\t")[0]
        assert "#--ligand                   = ['GTP', 'MG']" in header and "protein_alphabet1        = GADVP" in header

        # a finished run is skipped, so resubmitting it is harmless
        res = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=work,
                             env={**os.environ, "PATH": path, "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "43"})
        assert res.returncode == 0 and "already complete" in res.stdout

        res = _run(["--status"], env, path, work)
        assert "2 done" in res.stdout and "unfinished indices: 1,2" in res.stdout, res.stdout


def test_environment_selection_and_failed_activation():
    with tempfile.TemporaryDirectory(prefix="ames_env_") as t:
        tmp = Path(t)
        bindir, path = _setup(tmp)
        work = tmp / "work"
        work.mkdir()
        base = {"OUTROOT": str(tmp / "out"), "ALPHABETS": "GADVP", "CATIONS": "none", "REPS": "1",
                "NUCLEOTIDES": "ATP"}
        # a conda installation (with a conda.sh) that owns an environment, and a different one the shell loaded
        mini, other = tmp / "miniforge3", tmp / "msi_anaconda"
        for root in (mini, other):
            (root / "etc/profile.d").mkdir(parents=True)
            (root / "etc/profile.d/conda.sh").write_text("")
        env_dir = mini / "envs/esmfold2"
        env_dir.mkdir(parents=True)
        # the user's real situation: CONDA_PREFIX is in miniforge, CONDA_EXE belongs to another conda
        active = {"CONDA_PREFIX": str(env_dir), "CONDA_EXE": str(other / "bin/conda")}

        # the environment active in the submitting shell is used, through the conda installation that OWNS it
        out = _run([], {**base, **active}, path, work).stdout
        assert f"environment: source {mini}/etc/profile.d/conda.sh && conda activate {env_dir}" in out, out
        assert "active in this shell" in out and str(other) not in out and "/common/software" not in out

        # an environment created elsewhere (-p): fall back to the conda that CONDA_EXE belongs to
        custom = tmp / "scratch_env"
        custom.mkdir()
        out = _run([], {**base, "CONDA_PREFIX": str(custom), "CONDA_EXE": str(other / "bin/conda")}, path, work).stdout
        assert f"source {other}/etc/profile.d/conda.sh && conda activate {custom}" in out, out

        # nothing to derive a conda root from (no /envs/ in the path, no CONDA_EXE): the MSI default,
        # not a lookup of /etc/profile.d/conda.sh, which exists on machines with a system-wide conda
        out = _run([], {**base, "CONDA_PREFIX": str(custom), "CONDA_EXE": ""}, path, work).stdout
        assert "/common/software" in out and f"conda activate {custom}" not in out, out

        # no conda.sh can be found for the active environment: the MSI default, not a broken command
        out = _run([], {**base, "CONDA_PREFIX": "/nowhere/envs/x", "CONDA_EXE": "/nowhere/bin/conda"}, path, work).stdout
        assert "/common/software" in out and "conda activate esmfold2" in out and "/nowhere" not in out

        # CONDA_ENV names an environment in the MSI conda instead
        out = _run([], {**base, **active, "CONDA_ENV": "other"}, path, work).stdout
        assert "/common/software" in out and "conda activate other" in out

        # ENV_ACTIVATE wins over everything
        out = _run([], {**base, **active, "CONDA_ENV": "other", "ENV_ACTIVATE": "module load x"}, path, work).stdout
        assert "environment: module load x" in out

        # nothing active and nothing given: the MSI conda.sh and esmfold2
        out = _run([], base, path, work).stdout
        assert "/common/software" in out and "conda activate esmfold2" in out

        # a failed activation stops the job with an explanation instead of running on whatever python is on PATH
        res = _run(["--submit"], {**base, "ENV_ACTIVATE": "false", "SMOKE": "1"}, path, work)
        assert res.returncode == 0, res.stderr
        job = _job_scripts(bindir)[0]
        res = subprocess.run(["bash", str(job)], capture_output=True, text=True, cwd=work,
                             env={**os.environ, "PATH": path, "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "1"})
        assert res.returncode == 1 and "could not activate the environment" in res.stderr, res.stderr


def test_min_length_reaches_ames():
    with tempfile.TemporaryDirectory(prefix="ames_minlen_") as t:
        tmp = Path(t)
        bindir, path = _setup(tmp)
        work = tmp / "work"
        work.mkdir()
        env = {"OUTROOT": str(tmp / "out"), "ALPHABETS": "GADVP", "CATIONS": "none", "NUCLEOTIDES": "ATP",
               "SMOKE": "1", "ENV_ACTIVATE": "true"}

        plan = _run([], env, path, work).stdout
        assert "chain length: start 65, limits none..160 (soft), mutations: npm" in plan, plan
        assert "WARNING: MUT=npm" in plan and "WARNING: no MINLEN" in plan
        plan = _run([], {**env, "MINLEN": "50", "MUT": "pmo"}, path, work).stdout
        assert "chain length: start 65, limits 50..160 (soft), mutations: pmo" in plan, plan
        assert "WARNING" not in plan
        # the same warnings reach stderr on a real submit, without blocking it
        res = _run(["--submit"], env, path, work)
        assert res.returncode == 0 and "WARNING: MUT=npm" in res.stderr and "WARNING: no MINLEN" in res.stderr

        assert _run(["--submit"], env, path, work).returncode == 0
        assert 'MINLEN=""' in _job_scripts(bindir)[0].read_text()

        res = _run(["--submit"], {**env, "MINLEN": "30"}, path, work)
        assert res.returncode == 0, res.stderr  # MINLEN does not change the matrix, so no manifest conflict
        job = _job_scripts(bindir)[-1].read_text()
        assert 'MINLEN="30"' in job and '${MINLEN:+--seq1_min_len "$MINLEN"}' in job

        # run that job body with the mock fold engine and look at the parameters ames recorded
        body = "\n".join(line for line in job.splitlines() if "check_env.py" not in line)
        script = tmp / "job.sh"
        script.write_text(body.replace("/run_ames_alphabet.py", "/tests/mock_ames.py"))
        res = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=work,
                             env={**os.environ, "PATH": path, "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "7"})
        assert res.returncode == 0, res.stdout[-1200:] + res.stderr[-2000:]
        header = (tmp / "out_smoke/GADVP/ATP/run01/progress.log").read_text().split("gndx\t")[0]
        assert re.search(r"#--seq1_min_len\s+= 30", header), header[:2000]
        assert re.search(r"#--seq1_max_len\s+= 160", header)


def test_score_ligands_setting():
    with tempfile.TemporaryDirectory(prefix="ames_scoreligands_") as t:
        tmp = Path(t)
        bindir, path = _setup(tmp)
        work = tmp / "work"
        work.mkdir()
        env = {"OUTROOT": str(tmp / "out"), "ALPHABETS": "GADVP", "NUCLEOTIDES": "ATP", "SMOKE": "1",
               "MINLEN": "50", "MUT": "pmo", "ENV_ACTIVATE": "true"}
        note = "NOTE: SCORE_LIGANDS=all: ames pools the ion"

        # default: ames' own pooled scoring, with a note when an ion is in the matrix
        plan = _run([], {**env, "CATIONS": "none MG"}, path, work).stdout
        assert note in plan, plan
        assert note not in _run([], {**env, "CATIONS": "none"}, path, work).stdout  # nothing to pool
        assert note not in _run([], {**env, "CATIONS": "none MG", "SCORE_LIGANDS": "nucleotide"}, path, work).stdout
        assert note in _run([], {**env, "CATIONS": "none MG MN"}, path, work).stdout

        assert _run(["--submit"], {**env, "CATIONS": "none MG"}, path, work).returncode == 0
        assert 'SCORE_FLAG=""' in _job_scripts(bindir)[-1].read_text()
        res = _run(["--submit"], {**env, "CATIONS": "none MG", "SCORE_LIGANDS": "nucleotide"}, path, work)
        assert res.returncode == 0, res.stderr  # not part of the manifest, so no conflict with the first submit
        job = _job_scripts(bindir)[-1].read_text()
        assert 'SCORE_FLAG="--score-nucleotide-only"' in job and "${SCORE_FLAG} --engine esmfold2" in job

        # run that job body: the launcher sees the flag and ames records that only chain B was scored
        body = "\n".join(line for line in job.splitlines() if "check_env.py" not in line)
        script = tmp / "job.sh"
        script.write_text(body.replace("/run_ames_alphabet.py", "/tests/mock_ames.py"))
        res = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=work,
                             env={**os.environ, "PATH": path, "SLURM_ARRAY_TASK_ID": "1", "SLURM_JOB_ID": "9"})
        assert res.returncode == 0, res.stdout[-1200:] + res.stderr[-2000:]
        assert "#scoring: ligand contacts and interface pLDDT on chain B only" in res.stderr, res.stderr[-800:]
        log = next((tmp / "out_smoke/GADVP").glob("ATP_MG/run01/progress.log")).read_text()
        assert re.search(r"#--ligand_chains\s+= B\n", log), log[:1500]

        res = _run(["--submit"], {**env, "SCORE_LIGANDS": "both"}, path, work)
        assert res.returncode != 0 and "SCORE_LIGANDS must be all or nucleotide" in res.stderr


def test_check_env_rejects_esm_that_cannot_load_the_weights():
    sys.path.insert(0, str(ROOT))
    from check_env import esm_version_problem
    for old in ("3.4.0", "3.2.1.post1", "3.0.8", "2.0.0"):
        message = esm_version_problem(old)
        assert message and "pip install -U --no-deps" in message and "size mismatch" in message, old
    for fine in ("3.4.1", "3.4.1.post1", "3.4.2", "3.5.0", "4.0.0"):
        assert esm_version_problem(fine) is None, fine


def test_check_env_reports_problems_with_fixes():
    res = subprocess.run([sys.executable, str(ROOT / "check_env.py")], capture_output=True, text=True)
    assert res.stdout.startswith("python 3.")
    fails = [line for line in res.stdout.splitlines() if line.startswith("FAIL")]
    # every failure comes with a fix line; no failures means a clean bill of health
    assert (res.returncode == 0 and not fails) or (res.returncode == 1 and "Problems:" in res.stdout and fails)
    assert ("environment OK" in res.stdout) == (res.returncode == 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
