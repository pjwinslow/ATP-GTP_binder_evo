#!/usr/bin/env python3
"""Pre-flight check of the Python environment used by these scripts.

    python check_env.py [--gpu]

Prints what is installed and exits non-zero, with the fix, if something is missing.
``--gpu`` also requires a visible CUDA device (use it inside a GPU job).
"""
import argparse
import importlib
import sys

AMES_PIN = "git+https://github.com/sahakyanhk/ames@dd39c57"
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
