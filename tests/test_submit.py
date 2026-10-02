"""submit_matrix.sh with a fake sbatch: directives, manifest, guards, and the job body itself.

    python tests/test_submit.py        # ~30 s; needs ames installed (python >= 3.12 env)

The job body is run for two array tasks with the real launcher/ames/visualames but the mock
fold engine (and without the GPU check), then --status and the skip-if-DONE path are checked.
"""
import os
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
    env = {**os.environ, "CONDA_PREFIX": "", "CONDA_EXE": "", "CONDA_ENV": "",
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

        # resubmitting specific indices; changed settings are refused instead of shifting the indices
        res = _run(["--submit", "1,2"], env, path, work)
        assert "Resubmitting specific jobs: 1,2" in res.stdout
        assert "#SBATCH --array=1,2" in _job_scripts(bindir)[2].read_text()
        res = _run(["--submit"], {**env, "ALPHABETS": "GADVP ALL20"}, path, work)  # (SMOKE pins REPS=1)
        assert res.returncode != 0 and "differs from the matrix" in res.stderr

        # --status before anything ran
        res = _run(["--status"], env, path, work)
        assert res.returncode == 0 and "4 not started" in res.stdout and "unfinished indices: 0,1,2,3" in res.stdout, res.stdout

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
        active = {"CONDA_PREFIX": "/users/me/miniforge3/envs/esmfold2", "CONDA_EXE": "/users/me/miniforge3/bin/conda"}

        # the environment active in the submitting shell is used, through the conda installation that owns it
        out = _run([], {**base, **active}, path, work).stdout
        assert ("environment: source /users/me/miniforge3/etc/profile.d/conda.sh && "
                "conda activate /users/me/miniforge3/envs/esmfold2") in out and "active in this shell" in out
        assert "/common/software" not in out

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
