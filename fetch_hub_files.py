#!/usr/bin/env python3
"""Make everything ESMFold2 needs from the Hugging Face hub available locally.

    python fetch_hub_files.py            # download what is missing (run on a login node)
    python fetch_hub_files.py --check    # only report

ESMFold2 loads its weights with ``snapshot_download`` and, the first time it builds a ligand,
downloads ``ccd.pkl`` (the chemical component dictionary: ATP, GTP, Mg, ...) with
``hf_hub_download``. Protein-only use never needs that file, so it is usually not cached. When
the hub rate-limits the IP (HTTP 429, common on a shared campus address) the download fails and
every job dies in its first fold. Fetch the files once, with a token if the IP is limited, and the
jobs (which run with HF_HUB_OFFLINE=1) then never touch the network.

If you cannot reach the hub from MSI at all, download
https://huggingface.co/biohub/ESMFold2/resolve/main/ccd.pkl on your own computer, copy it to MSI,
and give its path to ``submit_matrix.sh`` as CCD_PATH (it becomes ESMCFOLD_CCD_PATH in the jobs).
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COMPONENTS = ("ATP", "GTP", "MG")  # ligands and ion of the experiment matrix
MODEL_REPO = "biohub/ESMFold2"
CCD_FILE = "ccd.pkl"
PATTERNS = ["*.json", "*.safetensors"]  # what esm's resolve_model_dir asks for
DEFAULT_ESMC_REPO = "biohub/ESMC-6B"    # esm's default when the backbone is not bundled in the checkpoint


@dataclass
class Need:
    key: str                  # model | esmc | ccd | ccd_env
    label: str
    path: str | None          # where it is available locally, None if missing
    repo: str | None = None   # hub repo to fetch it from
    problem: str | None = None  # set when the file is there but cannot be right


def _cached_snapshot(repo: str):
    """Path of the cached snapshot of ``repo`` (same call as esm makes), or None. Never touches the network."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id=repo, allow_patterns=PATTERNS, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached
        return None


def _separate_esmc_repo(model_dir: str):
    """Repo of the ESMC backbone if esm must fetch it separately (not bundled in the config), else None."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if config.get("esmc_config"):
        return None
    return config.get("esmc_id") or DEFAULT_ESMC_REPO


def ccd_problem(path: str):
    """Why this file cannot be the CCD pickle, or None. A browser that was rate-limited or redirected can save
    an HTML page under the right name, and an interrupted download is a truncated file."""
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as fh:
            first = fh.read(1)
    except OSError as err:
        return f"cannot be read ({err})"
    if size < 1_000_000:
        return f"is only {size} bytes, far too small for the component dictionary (an error page saved by the browser?)"
    if first != b"\x80":
        return "does not start like a pickle file (an HTML error page saved by the browser?)"
    return None


def deep_check_ccd(path: str, required=REQUIRED_COMPONENTS):
    """Load the pickle, as esm does (needs rdkit), and report (components, required ones missing, required
    ones without a conformer). Slow: the dictionary is large."""
    import pickle
    with open(path, "rb") as fh:
        ccd = pickle.load(fh)
    if not isinstance(ccd, dict):
        raise ValueError(f"expected a dict of components, got {type(ccd).__name__}")
    missing = [c for c in required if c not in ccd]
    no_conformer = [c for c in required if c in ccd and not (hasattr(ccd[c], "GetNumConformers")
                                                              and ccd[c].GetNumConformers() > 0)]
    return len(ccd), missing, no_conformer


def status() -> list:
    """What a job needs from the hub, and where each item is available locally."""
    needs = []
    model = _cached_snapshot(MODEL_REPO)
    needs.append(Need("model", f"{MODEL_REPO} weights", model, MODEL_REPO))
    if model:
        esmc_repo = _separate_esmc_repo(model)
        if esmc_repo:
            needs.append(Need("esmc", f"{esmc_repo} (ESMC backbone)", _cached_snapshot(esmc_repo), esmc_repo))
    ccd_env = os.environ.get("ESMCFOLD_CCD_PATH")
    if ccd_env:
        found = Path(ccd_env).is_file()
        needs.append(Need("ccd_env", f"ccd.pkl at ESMCFOLD_CCD_PATH={ccd_env}", ccd_env if found else None,
                          problem=ccd_problem(ccd_env) if found else None))
    else:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(MODEL_REPO, CCD_FILE)
        cached = cached if isinstance(cached, str) else None
        needs.append(Need("ccd", "ccd.pkl (chemical component dictionary)", cached, MODEL_REPO,
                          problem=ccd_problem(cached) if cached else None))
    return needs


def download(need: Need) -> None:
    if need.key == "ccd":
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=need.repo, filename=CCD_FILE)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=need.repo, allow_patterns=PATTERNS)


def _is_rate_limit(err: BaseException) -> bool:
    """429 anywhere in the exception chain (hf_hub_download wraps it in LocalEntryNotFoundError)."""
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if getattr(getattr(err, "response", None), "status_code", None) == 429 or "429" in str(err)[:300]:
            return True
        err = err.__cause__ or err.__context__
    return False


def fetch(needs, download=download, retries=5, sleep=time.sleep, log=print) -> list:
    """Download every missing item, waiting out rate limits; returns the labels that could not be fetched."""
    failed = []
    for need in needs:
        if need.path:
            continue
        if need.key == "ccd_env":
            log(f"  {need.label} does not exist: point ESMCFOLD_CCD_PATH at an existing file or unset it")
            failed.append(need.label)
            continue
        for attempt in range(retries + 1):
            try:
                log(f"downloading {need.label} ...")
                download(need)
                break
            except Exception as err:  # noqa: BLE001
                if _is_rate_limit(err) and attempt < retries:
                    wait = min(600, 30 * 2 ** attempt)
                    log(f"  rate-limited by the Hugging Face hub (429); retrying in {wait} s ({attempt + 1}/{retries})")
                    sleep(wait)
                    continue
                log(f"  failed: {type(err).__name__}: {str(err)[:200]}")
                failed.append(need.label)
                break
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only report what is cached")
    parser.add_argument("--retries", type=int, default=5, help="retries per file when rate-limited [5]")
    args = parser.parse_args()

    if not args.check and os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        print("HF_HUB_OFFLINE is set, so nothing can be downloaded: unset it for this command.", file=sys.stderr)
        return 2

    failed = []
    for _ in range(2):  # the second pass picks up the ESMC repo, which is only known once the weights are cached
        needs = status()
        if args.check or all(n.path for n in needs):
            break
        failed = fetch(needs, retries=args.retries)
    needs = status()
    for need in needs:
        tag = "MISSING" if not need.path else "BAD" if need.problem else "ok  "
        print(f"{tag}  {need.label}" + (f": {need.path}" if need.path else "") + (f" {need.problem}" if need.problem else ""))
    if all(n.path and not n.problem for n in needs):
        print("\nAll files are available locally. Next: python check_env.py")
        return 0
    if any(n.problem for n in needs):
        return 1
    if not args.check:
        print("\nCould not fetch: " + "; ".join(failed or [n.label for n in needs if not n.path]))
        print("If the hub is rate-limiting this IP (429): create a free account, make a read token at\n"
              "https://huggingface.co/settings/tokens, then  export HF_TOKEN=hf_...  and run this again.\n"
              "Or download ccd.pkl on your own computer, copy it here, and submit with CCD_PATH=/path/to/ccd.pkl")
    return 1


if __name__ == "__main__":
    sys.exit(main())
