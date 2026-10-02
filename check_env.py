#!/usr/bin/env python3
"""Pre-flight check of the Python environment used by these scripts.

    python check_env.py [--gpu]

Prints what is installed and exits non-zero, with the fix, if something is missing.
``--gpu`` also requires a visible CUDA device (use it inside a GPU job).
"""
import argparse
import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

AMES_PIN = "git+https://github.com/sahakyanhk/ames@dd39c57"
ESM_MIN = (3, 4, 1)  # ames declares esm>=3.4.1; 3.4.0 cannot load the current biohub/ESMFold2 weights
ESM_FIX = 'pip install -U --no-deps "esm>=3.4.1"  (same dependencies as 3.4.0; leaves torch alone)'


def esm_version_problem(version: str):
    """Message if this esm release is too old for the published ESMFold2 weights, else None."""
    numbers = tuple(int(n) for n in re.findall(r"\d+", version)[:3])
    if numbers < ESM_MIN:
        return (f"esm {version} is older than {'.'.join(map(str, ESM_MIN))}: it builds the ESMFold2 input encoder "
                f"at half the width of the published weights and fails with 'size mismatch for "
                f"inputs_embedder.atom_attention_encoder.atom_to_token_linear.weight'. Fix: {ESM_FIX}")
    return None


CHECKS = [  # (module, how to fix)
    ("numpy", f"pip install {AMES_PIN}"),
    ("pandas", f"pip install {AMES_PIN}"),
    ("biotite", f"pip install {AMES_PIN}"),
    ("zstandard", f"pip install {AMES_PIN}"),
    ("matplotlib", f"pip install {AMES_PIN}"),
    ("Bio", "pip install biopython  (ames imports it but does not list it)"),
    ("ames", f"pip install {AMES_PIN}  (needs a C++17 compiler, e.g. module load gcc)"),
    ("evolution", "ames did not put its source directory on sys.path; reinstall ames"),
    ("pdb_contacts", "ames' C++ extension is not built; reinstall ames with a C++17 compiler"),
    ("torch", "install PyTorch in this environment"),
    ("esm.models.esmfold2", "pip install 'esm>=3.4.1'"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", action="store_true", help="fail unless a CUDA device is visible")
    parser.add_argument("--deep", action="store_true",
                        help="also load ccd.pkl and confirm ATP, GTP and MG are in it (slow, needs rdkit)")
    args = parser.parse_args()

    problems = []
    print(f"python {sys.version.split()[0]}  {sys.executable}")
    if sys.version_info < (3, 12):
        problems.append("Python >= 3.12 is required: ames' rnatools.py uses an f-string syntax that "
                        "older versions reject with a SyntaxError")

    for module, fix in CHECKS:
        try:
            mod = importlib.import_module(module)
            print(f"ok    {module} {getattr(mod, '__version__', '')}".rstrip())
        except Exception as err:  # noqa: BLE001 - report any import failure, not just ImportError
            print(f"FAIL  {module}: {type(err).__name__}: {err}")
            problems.append(f"{module}: {fix}")
        if module == "ames" and "ames" not in sys.modules:
            break  # evolution / pdb_contacts need the package's directory on sys.path

    try:
        esm_version = importlib.metadata.version("esm")
        problem = esm_version_problem(esm_version)
        print(f"{'FAIL' if problem else 'ok  '}  esm distribution {esm_version}")
        if problem:
            problems.append(problem)
    except importlib.metadata.PackageNotFoundError:
        pass  # the esm.models.esmfold2 import above already reported it

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fetch_hub_files
        for need in fetch_hub_files.status():
            bad = not need.path or need.problem
            print(f"{'FAIL' if bad else 'ok  '}  {need.label}" + (f": {need.path}" if need.path else " is not available locally")
                  + (f" {need.problem}" if need.problem else ""))
            if not need.path:
                problems.append(f"{need.label} is not available locally; jobs run offline and would die in their first fold. "
                                f"Fix, on a login node: python fetch_hub_files.py  (see its --help if the hub rate-limits you)")
            elif need.problem:
                problems.append(f"{need.label} {need.problem}. Download it again, or run python fetch_hub_files.py")
            elif args.deep and need.key in ("ccd", "ccd_env"):
                try:
                    n, missing, no_conformer = fetch_hub_files.deep_check_ccd(need.path)
                    wanted = ", ".join(fetch_hub_files.REQUIRED_COMPONENTS)
                    if missing or no_conformer:
                        print(f"FAIL  ccd.pkl loads ({n} components) but lacks {', '.join(missing) or 'nothing'}"
                              f"{'; no conformer for ' + ', '.join(no_conformer) if no_conformer else ''}")
                        problems.append("ccd.pkl does not contain usable entries for " + wanted + ": wrong or damaged file")
                    else:
                        print(f"ok    ccd.pkl loads: {n} components, including {wanted} with conformers")
                except Exception as err:  # noqa: BLE001
                    print(f"FAIL  ccd.pkl could not be loaded: {type(err).__name__}: {err}")
                    problems.append("ccd.pkl cannot be unpickled (truncated download, or rdkit missing): download it again")
    except Exception as err:  # noqa: BLE001 - huggingface_hub problems are reported by the esm import above
        print(f"info  could not check the Hugging Face cache: {type(err).__name__}: {err}")

    try:
        import torch
        have_gpu = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if have_gpu else "none"
        print(f"{'ok  ' if have_gpu else 'info'}  CUDA device: {name}")
        if args.gpu and not have_gpu:
            problems.append("no CUDA device visible to PyTorch (not on a GPU node, or a CPU-only torch build)")
    except Exception:  # noqa: BLE001 - torch problems are already reported above
        pass

    if problems:
        print("\nProblems:\n  - " + "\n  - ".join(problems))
        return 1
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
