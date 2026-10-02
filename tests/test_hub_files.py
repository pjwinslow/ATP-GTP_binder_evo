"""fetch_hub_files / check_env hub checks, with a fake Hugging Face cache (no network).

    python tests/test_hub_files.py        # needs huggingface_hub (an esm dependency)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import fetch_hub_files as fhf  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLAUSIBLE_CCD = b"\x80\x04" + b"x" * 1_100_000  # passes the size and pickle-header sanity check of ccd_problem


def _fake_repo(hf_home: Path, repo: str, files: dict) -> None:
    """Lay out a cached snapshot the way huggingface_hub does: refs/main -> snapshots/<commit>/..."""
    base = hf_home / "hub" / f"models--{repo.replace('/', '--')}"
    (base / "refs").mkdir(parents=True, exist_ok=True)
    (base / "refs" / "main").write_text(COMMIT)
    snap = base / "snapshots" / COMMIT
    snap.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (snap / name).write_bytes(content if isinstance(content, bytes) else content.encode())


def _status(hf_home: Path, **env):
    """status() in a fresh interpreter whose Hugging Face cache is `hf_home` (the cache path is read at import)."""
    code = ("import json, sys; sys.path.insert(0, %r); import fetch_hub_files as f; "
            "print(json.dumps([(n.key, bool(n.path), n.repo) for n in f.status()]))" % str(ROOT))
    clean = {k: v for k, v in os.environ.items() if k not in ("HF_HOME", "HF_HUB_CACHE", "ESMCFOLD_CCD_PATH",
                                                              "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE")}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**clean, "HF_HOME": str(hf_home), **env})
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_empty_cache_reports_everything_missing():
    with tempfile.TemporaryDirectory() as t:
        assert _status(Path(t)) == [["model", False, "biohub/ESMFold2"], ["ccd", False, "biohub/ESMFold2"]]


def test_complete_cache_with_separate_esmc_backbone():
    with tempfile.TemporaryDirectory() as t:
        home = Path(t)
        _fake_repo(home, "biohub/ESMFold2", {"config.json": json.dumps({"type": "x"}), "model.safetensors": "w",
                                             "ccd.pkl": "c"})
        # the ESMFold2 config does not bundle the backbone, so biohub/ESMC-6B is needed too, and is missing
        assert _status(home) == [["model", True, "biohub/ESMFold2"], ["esmc", False, "biohub/ESMC-6B"],
                                 ["ccd", True, "biohub/ESMFold2"]]
        _fake_repo(home, "biohub/ESMC-6B", {"config.json": "{}", "model.safetensors": "w"})
        assert all(found for _, found, _ in _status(home))


def test_bundled_backbone_needs_no_second_repo_and_env_path_overrides_cache():
    with tempfile.TemporaryDirectory() as t:
        home = Path(t)
        _fake_repo(home, "biohub/ESMFold2", {"config.json": json.dumps({"esmc_config": {"hidden_size": 1}}),
                                             "model.safetensors": "w"})
        assert _status(home) == [["model", True, "biohub/ESMFold2"], ["ccd", False, "biohub/ESMFold2"]]
        ccd = home / "my_ccd.pkl"
        ccd.write_text("c")
        assert _status(home, ESMCFOLD_CCD_PATH=str(ccd))[-1][:2] == ["ccd_env", True]
        assert _status(home, ESMCFOLD_CCD_PATH=str(home / "nope.pkl"))[-1][:2] == ["ccd_env", False]


def _http_error(status):
    err = Exception(f"{status} Client Error")
    err.response = SimpleNamespace(status_code=status)
    return err


def test_fetch_waits_out_rate_limits():
    needs = [fhf.Need("ccd", "ccd.pkl", None, "biohub/ESMFold2"), fhf.Need("model", "weights", "/already/here")]
    calls, waits = [], []

    def flaky(need):  # hf_hub_download hides the 429 as the cause of LocalEntryNotFoundError
        calls.append(need.key)
        if len(calls) <= 2:
            wrapper = FileNotFoundError("cannot find the requested files in the local cache")
            wrapper.__cause__ = _http_error(429)
            raise wrapper

    failed = fhf.fetch(needs, download=flaky, retries=5, sleep=waits.append, log=lambda *_: None)
    assert failed == [] and calls == ["ccd", "ccd", "ccd"] and waits == [30, 60]  # the cached item is skipped


def test_fetch_gives_up_on_other_errors_and_after_the_retries():
    def not_found(need):
        raise _http_error(404)
    waits = []
    assert fhf.fetch([fhf.Need("ccd", "ccd.pkl", None, "r")], download=not_found, sleep=waits.append,
                     log=lambda *_: None) == ["ccd.pkl"] and waits == []

    def always_limited(need):
        raise _http_error(429)
    waits = []
    assert fhf.fetch([fhf.Need("ccd", "ccd.pkl", None, "r")], download=always_limited, retries=3,
                     sleep=waits.append, log=lambda *_: None) == ["ccd.pkl"] and waits == [30, 60, 120]

    # an ESMCFOLD_CCD_PATH that points nowhere cannot be downloaded into
    assert fhf.fetch([fhf.Need("ccd_env", "ccd.pkl at ESMCFOLD_CCD_PATH=/x", None)], download=not_found,
                     log=lambda *_: None) == ["ccd.pkl at ESMCFOLD_CCD_PATH=/x"]


def test_check_env_flags_missing_hub_files_with_the_fix():
    with tempfile.TemporaryDirectory() as t:
        home = Path(t)
        env = {k: v for k, v in os.environ.items() if not k.startswith("HF_") and k != "ESMCFOLD_CCD_PATH"}
        res = subprocess.run([sys.executable, str(ROOT / "check_env.py")], capture_output=True, text=True,
                             env={**env, "HF_HOME": str(home)})
        assert "FAIL  ccd.pkl (chemical component dictionary) is not available locally" in res.stdout, res.stdout
        assert "python fetch_hub_files.py" in res.stdout and res.returncode == 1

        _fake_repo(home, "biohub/ESMFold2", {"config.json": json.dumps({"esmc_config": {"x": 1}}),
                                             "model.safetensors": "w", "ccd.pkl": PLAUSIBLE_CCD})
        res = subprocess.run([sys.executable, str(ROOT / "check_env.py")], capture_output=True, text=True,
                             env={**env, "HF_HOME": str(home)})
        assert "ok    biohub/ESMFold2 weights:" in res.stdout and "ok    ccd.pkl (chemical component dictionary):" in res.stdout
        assert "fetch_hub_files.py" not in res.stdout  # nothing left to fetch


def _write_ccd(path: Path, components: dict, pad: int = 1_200_000) -> Path:
    """A pickle shaped like ccd.pkl (a dict of component -> molecule), padded past the size sanity check."""
    import pickle
    sys.path.insert(0, str(HERE))
    path.write_bytes(pickle.dumps({**components, "_padding": b"x" * pad}))
    return path


def test_ccd_problem_catches_bad_downloads():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        good = _write_ccd(t / "ok.pkl", {})
        assert fhf.ccd_problem(str(good)) is None
        tiny = t / "tiny.pkl"
        tiny.write_bytes(b"\x80\x04.")
        assert "bytes, far too small" in fhf.ccd_problem(str(tiny))
        html = t / "page.pkl"
        html.write_text("<html>" + "x" * 2_000_000)
        assert "HTML error page" in fhf.ccd_problem(str(html))
        assert "cannot be read" in fhf.ccd_problem(str(t / "nope.pkl"))


def test_deep_check_ccd():
    sys.path.insert(0, str(HERE))
    from fake_ccd_mol import Mol
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ok = _write_ccd(t / "a.pkl", {"ATP": Mol(), "GTP": Mol(), "MG": Mol()}, pad=10)
        assert fhf.deep_check_ccd(str(ok)) == (4, [], [])
        lacking = _write_ccd(t / "b.pkl", {"ATP": Mol(), "MG": Mol(0)}, pad=10)
        assert fhf.deep_check_ccd(str(lacking)) == (3, ["GTP"], ["MG"])
        import pickle
        (t / "c.pkl").write_bytes(pickle.dumps(["not", "a", "dict"]))
        try:
            fhf.deep_check_ccd(str(t / "c.pkl"))
        except ValueError as err:
            assert "expected a dict" in str(err)
        else:
            raise AssertionError("a list was accepted")


def test_check_env_judges_a_manually_downloaded_ccd():
    from fake_ccd_mol import Mol
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        env = {k: v for k, v in os.environ.items() if not k.startswith("HF_") and k != "ESMCFOLD_CCD_PATH"}
        env = {**env, "HF_HOME": str(t / "hf"), "PYTHONPATH": f"{HERE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}

        def check(path, *flags):
            return subprocess.run([sys.executable, str(ROOT / "check_env.py"), *flags], capture_output=True, text=True,
                                  env={**env, "ESMCFOLD_CCD_PATH": str(path)}).stdout

        good = _write_ccd(t / "good.pkl", {"ATP": Mol(), "GTP": Mol(), "MG": Mol()}, pad=1_200_000)
        out = check(good)
        assert f"ccd.pkl at ESMCFOLD_CCD_PATH={good}: {good}" in out and "FAIL  ccd.pkl at" not in out
        out = check(good, "--deep")
        assert "ok    ccd.pkl loads: 4 components, including ATP, GTP, MG with conformers" in out, out

        no_gtp = _write_ccd(t / "no_gtp.pkl", {"ATP": Mol(), "MG": Mol()})
        out = check(no_gtp, "--deep")
        assert "FAIL  ccd.pkl loads (3 components) but lacks GTP" in out, out

        page = t / "page.pkl"
        page.write_text("<!DOCTYPE html>" + "x" * 2_000_000)
        out = check(page)
        assert "FAIL  ccd.pkl at" in out and "HTML error page" in out and "Download it again" in out, out

        # a pickle cut off mid-file starts correctly and is only caught by loading it
        truncated = t / "truncated.pkl"
        truncated.write_bytes(good.read_bytes()[:1_000_100])
        out = check(truncated, "--deep")
        assert "FAIL  ccd.pkl could not be loaded" in out, out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
