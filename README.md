# ATP and GTP binding from reduced amino-acid alphabets with ames

Run [ames](https://github.com/sahakyanhk/ames) (atomistic molecular evolution
simulator) with ESMFold2 as the structure predictor to ask: starting from random
sequences built from a **reduced alphabet**, can selection for a predicted ATP or
GTP complex produce nucleotide-binding proteins, and does it matter whether a
divalent cation is present?

| | |
|---|---|
| Nucleotides | `ATP` · `GTP` (`NUCLEOTIDES`, default both) |
| Alphabets | `GADVP` (5) ⊂ `GADVPSELT` (9) ⊂ `GADVPSELTRIQN` (13) ⊂ `ALL20` (20) |
| Ligand conditions | nucleotide alone · nucleotide + `MG` (other ions: `CATIONS="none MG MN CA"`) |
| Replicates | independent runs per condition (`REPS`, default 3) |
| Control | `CONTROL=neutral`: same mutations and scoring, **no selection** (`-b0 0`) |

The default matrix is 2 nucleotides × 4 alphabets × 2 cation conditions × 3
replicates = **48 runs**, plus the neutral control if you run it.

ames is used unmodified. Each run is a population (default 100) of chains that
start random (`protein:randoms:65:evolv`), mutate every generation (substitutions,
indels, duplications, …), are folded with ESMFold2 together with the ligand(s), and
are selected on a score from pTM, pLDDT, ipTM, interface pLDDT, protein contact
density and ligand contact density. Each run evolves under **one** nucleotide.

## Files

| File | Purpose |
|---|---|
| `ames_alphabets.py` | the four alphabets; registers them in ames' `Evolver` |
| `run_ames_alphabet.py` | `ames` with `--alphabet NAME` added; all other arguments go to ames |
| `submit_matrix.sh` | Slurm: nucleotide × alphabet × cation × replicate, one GPU job each (dry run unless `--submit`) |
| `summarize_matrix.py` | aggregate runs, re-measure nucleotide/ion contacts, make tables and figures |
| `tests/` | alphabet tests, structure-metric tests, end-to-end test with a mock fold engine (no GPU) |

## Setup on MSI

ames needs **Python ≥ 3.12** (`rnatools.py` uses an f-string syntax that is a
`SyntaxError` on 3.11 and older, although ames' README says ≥ 3.10) and a C++17
compiler (`module load gcc`) to build its contact-calculation extension.

Install into a **copy** of your working ESMFold2 environment, because ames pins
`numpy>=2.0`, which can break a torch build made against numpy 1.x:

```bash
pip install git+https://github.com/sahakyanhk/ames
pip install biopython     # imported by ames at start-up but missing from its dependency list
```

ames loads the ESMFold2 weights (`biohub/ESMFold2`) from the Hugging Face hub when
the run starts. If compute nodes have no internet, warm the cache on a login node
and `export HF_HOME=...` (inherited by the jobs) with `HF_HUB_OFFLINE=1`.

## Running

1. **Smoke test** (8 sequences × 4 generations, one job): confirms the environment,
   ESMFold2, ligand input and `visualames` work, and shows the throughput. With the
   default `NUCLEOTIDES` this submits one job per nucleotide, which also confirms that
   ESMFold2 accepts both ligands:

   ```bash
   export ENV_ACTIVATE='conda activate ames-esm'    # whatever activates your env
   SMOKE=1 ALPHABETS=GADVP CATIONS=MG ./submit_matrix.sh --submit
   ```
   ames prints `#N generations per day` every 10 generations in the job's `.out`
   file. **Size `NG`/`PS`/`TIME` from that number before launching the matrix.** A
   run folds `PS × NG` sequences one after another (default 100 × 1000 = 100,000).
   ames has **no resume**: a job that hits its walltime keeps its partial
   `progress.log` but cannot be continued.

2. **Plan, then submit the matrix.** Without `--submit` the script only prints
   the runs. `PARTITION`, `GRES`, `TIME` default to `a100-4`, `gpu:a100:1`,
   `72:00:00`, which are guesses: check `sinfo` / the MSI docs and override.

   ```bash
   ./submit_matrix.sh                                    # 48 jobs
   NUCLEOTIDES=GTP ./submit_matrix.sh                    # 24 jobs, GTP only
   PARTITION=... GRES=... TIME=... ./submit_matrix.sh --submit
   CONTROL=neutral NG=200 REPS=3 ./submit_matrix.sh --submit   # the no-selection null
   ```
   Outputs: `outputs/nuc_matrix/<alphabet>/<ATP|ATP_MG|GTP|GTP_MG>/runNN/` (neutral
   control: `outputs/nuc_matrix_neutral/…`). Each `progress.log` embeds a compressed
   structure per row and gets large; keep outputs on scratch. Each job ends with
   `visualames`, which writes `lineage.tsv`, `structures/` and plots next to the log.

3. **Summarise** (login node is fine):

   ```bash
   python summarize_matrix.py outputs/nuc_matrix outputs/nuc_matrix_neutral --out outputs/summary
   ```

   | Output | Content |
   |---|---|
   | `runs.csv` | one row per run: ames metrics at the start and end of the lineage, nucleotide-only and ion metrics, final sequence, motif flags |
   | `conditions.csv` | per selection × nucleotide × alphabet × cation: n, median/min/max, fraction with a confident pose |
   | `contact_composition.csv` | amino-acid fractions among nucleotide-contacting residues |
   | `final_metrics.png` | final lineage member of each run, selected vs neutral; one column per nucleotide × cation |
   | `trajectories.png` | ames metrics along each lineage |

   Per-run columns: `nuc_contact_res` (residues within 4.5 Å of the nucleotide),
   split into `phosphate_`, `base_` and `ribose_contact_res`; `nuc_atom_contact_frac`;
   `nuc_plddt`; `ion_contact_res`, `ion_coord_number`, `ion_coord_protein_res`.

## Design choices to know about

**Equal mutation spectrum across alphabets.** In ames every letter of the alphabet
gets weight 1 while indel/duplication/permutation operators keep fixed weights, so
a 5-letter alphabet would get ~3× more indels and duplications per mutation than 20
letters. `ames_alphabets.py` gives each letter weight `20/N`, which keeps the
operator mix identical for every alphabet (`ALL20` is exactly ames' stock
`uniform`). `--no-rate-normalize` gives ames' native weights.

**The nucleotide is measured separately from the ion.** ames' `lcd` and `iplddt` pool
all ligand chains. In a `NUC,MG` run, a residue that touches both the nucleotide and
Mg is counted twice and Mg adds one atom to the denominator (ATP has 31 heavy atoms and
GTP 32, so `lcd` = contacting pairs / 32 or / 33 with Mg). ames' numbers are therefore
**not comparable between the nucleotide-only and +Mg conditions, nor between ATP and
GTP**. A residue that coordinates a Mg bound to a phosphate oxygen is also, by geometry,
a nucleotide contact (two ~2.1 Å Mg–O bonds put it within ~4.2 Å).
`summarize_matrix.py` re-measures from the final structure, with ames' own rule (heavy
atoms < 4.5 Å, both with pLDDT ≥ 60), the nucleotide-only contacts and the ion
contacts, plus the ion coordination number (O/N within 2.8 Å). For nucleotide-only
runs `nuc_lcd` equals ames' `lcd`, and for nucleotide+ion runs
`lcd × (heavy atoms + 1) = nuc_contact_res + ion_contact_res`; `tests/run_mock_e2e.py`
asserts both for ATP and GTP against ames' C++ code.

**ATP and GTP differ only in the base.** The phosphate chain and ribose have identical
atom names; adenine has `N6`, guanine has `O6` and `N2`. `base_contact_res` counts
residues touching the base, which is where any discrimination between the two has to
come from. The atom-name sets are checked against the chemical-component library in
`tests/test_structure_metrics.py`.

**What the ATP/GTP comparison can and cannot show.** Each run evolves against one
nucleotide, so you can compare how readily each evolves a confident pose and which
residues it recruits. It does **not** measure specificity: nothing scores an
ATP-evolved sequence against GTP. To test specificity, fold the final sequences from
each condition with the other nucleotide (not implemented here).

**`confident_pose`** is a convenience flag: ≥ 60 % of nucleotide atoms in contact with
the protein and mean nucleotide pLDDT ≥ 70. Both thresholds are arbitrary defaults
(`--min-atom-frac`, `--min-ligand-plddt`); calibrate them on the neutral control.

**Neutral control.** Without it, "scores went up" cannot be told from drift and
ESMFold2's priors. Compare selected runs with `CONTROL=neutral` runs of the same
nucleotide, alphabet and cation at the same lineage length.

**Alphabets and nucleotide chemistry.** None of the reduced alphabets contains Lys, so
a canonical Walker A / P-loop `GxxxxGK[ST]` can only appear with `ALL20`
(`ploop_strict`); `ploop_KR` also accepts Arg. `GADVPSELT` has no basic residue at all,
and Arg first appears in `GADVPSELTRIQN`, so any phosphate recognition there has to
come from backbone NH, S/T hydroxyls, or acidic side chains bridged by Mg. That is a
natural thing to look at in `contact_composition.csv` and the `ion_coord_*` columns.

## Caveats

- Everything here is predicted structure. ESMFold2 confidence for a co-folded ligand
  is a proxy for binding, not a measurement; validate the interesting sequences with
  an independent predictor, point-mutant sensitivity and, ideally, experiment.
- Selection uses ames' score weights for protein + ligand runs
  (`PROTEIN_LIGAND_EVOLUTION`: 0.1 pTM, 0.1 pLDDT, 0.2 ipTM, 0.2 ipLDDT, 0.2 protein
  contact density, 0.2 ligand contact density), chosen by the ames author, not tuned for
  nucleotides. They are hard-coded in `ScoringFunction` (`amestools.py`); the
  `data/scoring_weights.json` file in ames is never read, so changing them means patching
  ames (a small override in `run_ames_alphabet.py` would do it).
- ames' bundled `bash_helpers/run_ames.sbatch` passes `--seq1_len_constr`, which `ames`
  rejects (`unrecognized arguments`); `submit_matrix.sh` uses `--seq1_max_len`.
- ames' `-ed/--evoldict` option is parsed but never used, which is why alphabets are
  registered by `run_ames_alphabet.py` instead.
- Not run against real ESMFold2 here (no GPU or weights), for ATP or GTP. I read the
  `esm` 3.4.1 source to confirm what ames relies on: `result.plddt` is 0–1, structure
  B-factors are pLDDT × 100, ligands come out as one residue named by the CCD code with
  CCD atom names and the chain IDs ames requested. The phosphate/base/ribose split in
  `summarize_matrix.py` relies on those names, and it warns if a nucleotide atom name is
  not recognised (`nuc_atoms_unrecognized` in `runs.csv`). Whether ESMFold2 places GTP as
  well as ATP is something only the smoke test can tell you.

## Tests

```bash
python tests/test_alphabets.py            # needs ames importable
python tests/test_structure_metrics.py    # needs biotite (an ames dependency)
python tests/run_mock_e2e.py              # ~1 min on 4 cores: launcher → ames → visualames → summarizer, mock fold engine
```
The mock engine (`tests/mock_esmfold2_runner.py`) builds synthetic protein + nucleotide
(+ Mg) structures and returns made-up confidences, so it exercises the plumbing and the
scoring path for both nucleotides, not ESMFold2 or any biology.
