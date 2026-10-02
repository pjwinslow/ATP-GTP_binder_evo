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

ames is not modified (the alphabets and checkpoint resume are patched in at run time).
Each run is a population (default 100) of chains that
start random (`protein:randoms:65:evolv`), mutate every generation (substitutions,
indels, duplications, …), are folded with ESMFold2 together with the ligand(s), and
are selected on a score from pTM, pLDDT, ipTM, interface pLDDT, protein contact
density and ligand contact density. Each run evolves under **one** nucleotide.

## Files

| File | Purpose |
|---|---|
| `ames_alphabets.py` | the four alphabets; registers them in ames' `Evolver` |
| `run_ames_alphabet.py` | `ames` with `--alphabet NAME` added; all other arguments go to ames |
| `ames_resume.py` | checkpoint resume for ames (`--resume`), which ames lacks; see [Preemption and resume](#preemption-and-resume) |
| `submit_matrix.sh` | one Slurm job array for the whole matrix, MSI settings from a working script; `--status`, resubmit by index (plan only unless `--submit`) |
| `check_env.py` · `fetch_hub_files.py` · `run_status.py` | environment pre-flight (run by every job) · download the Hugging Face files the jobs need, once · progress of each run |
| `summarize_matrix.py` | aggregate runs, re-measure nucleotide/ion contacts, make tables and figures |
| `tests/` | alphabet, structure-metric, resume and submitter tests, end-to-end test with a mock fold engine (no GPU) |

## Setup on MSI

ames needs **Python ≥ 3.12** (`rnatools.py` uses an f-string syntax that is a
`SyntaxError` on 3.11 and older, although ames' README says ≥ 3.10) and a C++17
compiler (`module load gcc`) to build its contact-calculation extension.

The jobs activate **the conda environment that is active in the shell you submit from**
(through the conda installation that owns it), so activate your ESMFold2 environment, check
it with `python check_env.py`, and submit from that same shell. `bash submit_matrix.sh`
prints the exact activation command it will use (`environment: …`). To use something else:
`CONDA_ENV=name` (looked up in the MSI anaconda; `CONDA_SH=/path/to/conda.sh` for another
conda) or `ENV_ACTIVATE='any command'`. If the activation fails the job stops with an error,
instead of running on whatever `python` happens to be on the path. The environment needs
ames and biopython on top of ESMFold2. If you add
them to a **copy** of the environment you protect it from ames' `numpy>=2.0` pin, which
can break a torch build made against numpy 1.x:

```bash
conda activate esmfold2
pip install git+https://github.com/sahakyanhk/ames@dd39c57   # the commit these scripts were tested with
pip install biopython     # imported by ames at start-up but missing from its dependency list
pip install -U --no-deps "esm>=3.4.1"   # see below
python check_env.py       # on a login node, everything but the GPU line should say ok
```
**`esm` must be 3.4.1 or newer** (ames declares this for its ESMFold2 engine, but only in an
optional extra, so installing ames as above does not enforce it). With `esm` 3.4.0 every
run dies while loading the model with `size mismatch for
inputs_embedder.atom_attention_encoder.atom_to_token_linear.weight`: 3.4.0 builds that
layer at half the width of the weights now published as `biohub/ESMFold2`, which 3.4.1 reads
correctly. `esm` 3.4.1 declares the same dependencies as 3.4.0, so `--no-deps` leaves your
torch untouched. `check_env.py` reports the version and the fix.
Pinning `dd39c57` matters: the resume patch rewrites part of ames' main loop and refuses
to run on a version it does not recognise.

### Hugging Face files (once, on a login node)

ESMFold2 needs files from the Hugging Face hub: the weights, and `ccd.pkl` (the chemical
component dictionary that defines ATP, GTP and Mg), which it downloads the first time it
builds a ligand. Protein-only use never needs `ccd.pkl`, so it is usually not cached, and the
hub rate-limits shared IP addresses (HTTP 429, "We had to rate limit your IP"). Without the file
every job died in its first fold with `Failed to download CCD pickle file` and a 429.

```bash
python fetch_hub_files.py      # downloads what is missing and waits out rate limits
python check_env.py            # "ccd.pkl ... : /path" should now say ok
```
- If it keeps failing with 429: make a free Hugging Face account and a read token
  (https://huggingface.co/settings/tokens), `export HF_TOKEN=hf_...`, and run it again.
- If MSI cannot reach the hub, or you keep hitting the limit: download
  `https://huggingface.co/biohub/ESMFold2/resolve/main/ccd.pkl` on your own computer and copy it to
  MSI, then `export ESMCFOLD_CCD_PATH=/path/to/ccd.pkl` before `check_env.py` and `submit_matrix.sh`
  (the submitter passes it to the jobs; `CCD_PATH=` works too, and a path that is not a file is
  refused). `python check_env.py --deep` loads the file and confirms ATP, GTP and MG are in it with
  conformers, which catches a truncated download or an HTML error page saved under the right name
  (the light check that always runs catches the second by size and file header).
- The jobs run with `HF_HUB_OFFLINE=1` (`HF_OFFLINE=0` to allow downloads), so up to 25 tasks
  starting at once never contact the hub. `check_env.py`, which every job runs first, fails in
  seconds, naming the missing file, if anything is not cached.

## Running

Run from a checkout of this repo on MSI (`WORKDIR` defaults to the script's directory).
The Slurm settings are those of your working MSI script: `preempt-gpu`, `gpu:1`, 40G,
24 h, `--requeue`, at most 25 array tasks at once, one array for the whole matrix with
logs in `logs/`. Override any of them with environment variables (see the header of
`submit_matrix.sh`); `MAIL_USER=you@umn.edu` turns on the one end/fail mail per array.

1. **Smoke test** (8 sequences × 12 generations, 1 h): confirms the environment,
   ESMFold2, ligand input and `visualames` work, and measures the speed. With the
   default `NUCLEOTIDES` it runs one job per nucleotide, which also confirms that
   ESMFold2 accepts both ligands:

   ```bash
   SMOKE=1 ALPHABETS=GADVP CATIONS=MG bash submit_matrix.sh --submit
   ```
   ames prints a timing line (`#X.Xs per generation`) at generation 10, so it appears in
   each task's `logs/*.out` only because the smoke test runs 12 generations. That time is
   for 8 sequences; a real generation folds `PS` sequences one after another, so expect
   about X × PS/8 seconds per generation and X × (PS/8) × NG seconds per run (100 × 1000
   = 100,000 folds by default). Fold time also grows with chain length, so treat it as a
   floor. **Size `NG`, `PS` and the number of 24-hour slots from that before launching the
   matrix.**

2. **Plan, submit, monitor.** Without `--submit` the script only prints the plan.

   ```bash
   bash submit_matrix.sh                       # plan: 48 runs, one array index each
   bash submit_matrix.sh --submit              # submit them all
   bash submit_matrix.sh --status              # per-run progress + the unfinished indices
   bash submit_matrix.sh --submit 12,45,99     # resubmit just those indices (they resume)
   NUCLEOTIDES=GTP bash submit_matrix.sh       # 24 runs, GTP only
   CONTROL=neutral NG=200 bash submit_matrix.sh --submit   # the no-selection null
   ```
   `--status`, resubmits and the plan must use the same settings as the original submit:
   array indices refer to `outputs/nuc_matrix/manifest.tsv`, and a submit whose settings
   would change that file is refused.

   Outputs: `outputs/nuc_matrix/<alphabet>/<ATP|ATP_MG|GTP|GTP_MG>/runNN/` (neutral
   control: `outputs/nuc_matrix_neutral/…`; smoke test: `…_smoke`). Each `progress.log`
   embeds a compressed structure per row and gets large; keep outputs on scratch. A task
   ends with `visualames` (writes `lineage.tsv`, `structures/` and plots next to the log)
   and a `DONE` marker file.

### Preemption and resume

ames has no resume of its own: it always starts at generation 0 and moves an existing
output directory aside. On `preempt-gpu` with `--requeue` that would restart every
preempted run from scratch, so the launcher is run with `--resume` (`ames_resume.py`):

- ames checkpoints the population every generation (`CKPI`, default 1). A requeued task
  loads `progress.ckp`, cuts `progress.log` back to that generation, restores the
  annealing schedule and continues; a run with no checkpoint starts normally.
- Slurm requeues preempted jobs but **not** jobs that hit the 24-hour limit. A run that
  needs more than one slot ends as `TIMEOUT`: use `--status` and resubmit the unfinished
  indices; they continue from their checkpoint.
- A finished run (`DONE` present) is skipped, so resubmitting everything is safe.
- A checkpoint made with a different alphabet, ligand or population size is refused.
- Not restored: ames' memo of already-folded sequences (a few sequences may be folded
  twice) and the random-number state, so a resumed run is statistically equivalent to,
  not bit-identical with, an uninterrupted one.
- `tests/test_resume.py` kills a real run with SIGKILL, resumes it, and checks that every
  generation appears exactly once, the selection strength continues on schedule across the
  break, and `visualames` still works.

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

**Chain length can collapse: set a floor.** ames' default lower length limit is −9, which is no
limit, and its default mutation set (`npm`) contains operators that can shrink a chain a lot in one
step: `%` deletes a random chunk (up to all but 2 residues) and `r` replaces the whole chain with 3–5
random residues. In the first smoke runs on MSI (8 sequences, 12 generations) both lineages started at
65 residues and lost almost all of them in a single `%` event in the first two generations: 65 to 7
residues in the ATP+Mg run (`V7%58`, final sequence `PPAADDP`) and 65 to 10 in the GTP+Mg run (`P5%55`).
The tiny chains then scored well and were kept. The likely reason, not established, is that the
complex scores (pTM, ipTM, pLDDT) are computed over the protein *and* the ligand atoms, so a tiny chain
next to ATP can score well and nothing opposes it. Two settings address it: `MINLEN=N` (ames'
`--seq1_min_len`, a soft limit: the score is multiplied by 0.5 at N residues, 0.95 six above and
0.05 six below) and `MUT=pmo` (substitutions and single-residue indels only, so a chain changes length
by one residue at a time). Check `final_metrics.png` (final chain length) and `trajectories.png`
(chain length along each lineage) for collapse before reading anything else into the scores.

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
- The resume patch is pinned to ames commit `dd39c57`; on another version it stops with an
  error naming the line it could not find rather than running without resume.
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
python tests/test_resume.py               # ~25 s: kill a run, resume it, check the stitched log
python tests/test_submit.py               # ~25 s: fake sbatch; directives, manifest, job body, --status
python tests/test_hub_files.py            # fake Hugging Face cache; needs huggingface_hub (an esm dependency)
python tests/run_mock_e2e.py              # ~1 min on 4 cores: launcher → ames → visualames → summarizer, mock fold engine
```
The mock engine (`tests/mock_esmfold2_runner.py`) builds synthetic protein + nucleotide
(+ Mg) structures and returns made-up confidences, so it exercises the plumbing and the
scoring path for both nucleotides, not ESMFold2 or any biology. Slurm itself is not
available here: the generated job script is checked against a fake `sbatch`, and its body
is run directly, so the first real submission (the smoke test) is the check of the
Slurm/conda/GPU side.
