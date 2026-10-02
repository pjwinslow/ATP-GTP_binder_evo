#!/usr/bin/env python3
"""Run ames with a reduced amino-acid alphabet.

    run_ames_alphabet.py --alphabet GADVP [--no-rate-normalize] [--resume] <ames arguments>

Registers the alphabet in ames' ``Evolver`` (see ames_alphabets.py) and then
runs ames' own entry point with ``-pa1 <alphabet>`` added. ames is not modified.
Everything after the wrapper's own options is passed to ames unchanged. With
``--resume`` an interrupted run continues from its checkpoint (see ames_resume.py);
without it ames starts over and moves an existing output directory aside. E.g.

    run_ames_alphabet.py --alphabet GADVPSELT \\
        --iseq1 protein:randoms:65:evolv --ligand ATP,MG \\
        -ps 100 -ng 1000 -o outputs/test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ames_alphabets import ALPHABETS, register_alphabet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    parser.add_argument("--alphabet", required=True,
                        help=f"one of {', '.join(ALPHABETS)} or an explicit letter set")
    parser.add_argument("--no-rate-normalize", action="store_true",
                        help="ames' native weights (1 per letter) instead of 20/N per letter")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <outpath>/progress.ckp if it exists (otherwise start fresh)")
    own, ames_args = parser.parse_known_args()

    if any(a in ("-pa1", "--protein_alphabet1") for a in ames_args):
        parser.error("do not pass -pa1/--protein_alphabet1, use --alphabet")
    if own.resume and "--nobackup" in ames_args:
        parser.error("--resume and --nobackup conflict: --nobackup deletes the output directory")

    try:
        import ames  # noqa: F401  puts ames' source dir on sys.path
        import evolution  # the same module object ames.py imports by bare name
    except ImportError as err:
        sys.exit(f"ERROR: cannot import ames ({err}). "
                 "Install it first: pip install git+https://github.com/sahakyanhk/ames")

    key = register_alphabet(evolution.Evolver, own.alphabet,
                            normalize=not own.no_rate_normalize)
    print(f"#alphabet {key}: {''.join(evolution.Evolver.protein[key])}", file=sys.stderr)

    if own.resume:
        import ames_resume
        ames_resume.prepare()  # before ames.ames is imported: it moves the output dir aside at import

    sys.argv = [sys.argv[0], "-pa1", key, *ames_args]
    from ames.ames import main as ames_main  # setup runs at import, reads sys.argv
    if own.resume:
        ames_resume.patch_simulator(sys.modules["ames.ames"])
    ames_main()


if __name__ == "__main__":
    main()
