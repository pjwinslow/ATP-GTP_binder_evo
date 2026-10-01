#!/usr/bin/env python3
"""run_ames_alphabet.py with the mock fold engine; use with ``--engine esmfold2``."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import os  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402

if "MOCK_SEED" in os.environ:  # ames does not seed; pandas sampling uses numpy's global RNG
    random.seed(int(os.environ["MOCK_SEED"]))
    np.random.seed(int(os.environ["MOCK_SEED"]))

import mock_esmfold2_runner  # noqa: E402

sys.modules["esmfold2_runner"] = mock_esmfold2_runner  # what ames.py imports for --engine esmfold2

import run_ames_alphabet  # noqa: E402

run_ames_alphabet.main()
