#!/bin/bash
# submit_matrix.sh
#
# Submits the nucleotide-binding experiment matrix as ONE Slurm job array, one run per
# array element (one GPU each):
#
#     nucleotide (ATP, GTP) x alphabet (GADVP, GADVPSELT, GADVPSELTRIQN, ALL20)
#   x cation (none -> ligand NUC;  MG -> ligand NUC,MG;  MN, CA, ... also work)
#   x replicate
#
#   bash submit_matrix.sh                       # print the plan only
#   bash submit_matrix.sh --submit              # submit all runs
#   bash submit_matrix.sh --status              # progress of every run + unfinished indices
#   bash submit_matrix.sh --status outputs/pilot   # ... for runs submitted to another OUTROOT
#   bash submit_matrix.sh --submit 12,45,99     # resubmit only these array indices
#
# Preempted or requeued jobs, and runs that hit the walltime and are resubmitted, RESUME
# from their last checkpoint (ames has no resume of its own: see ames_resume.py), and a
# finished run is skipped, so resubmitting is always safe.
#
#   SMOKE=1 bash submit_matrix.sh --submit      # tiny test: 8 sequences x 12 generations, 1 h
#   CONTROL=neutral bash submit_matrix.sh --submit   # no-selection null (same settings otherwise)
#
# --status, resubmitting and the plan must see the same settings as the original submit
# (same SMOKE/CONTROL/NUCLEOTIDES/... values), because array indices refer to the manifest
# written to $OUTROOT/manifest.tsv; a changed setting is refused rather than silently
# shifting the indices (FORCE_MANIFEST=1 overrides).
#
# Settings (environment variables, defaults in brackets)
#   Slurm     PARTITION [preempt-gpu]  TIME [24:00:00]  MEM [40G]  GPUS [gpu:1]
#             MAX_PARALLEL [25] simultaneous array tasks   CPUS [unset]   ACCOUNT [unset]
#             MAIL_USER [unset: no mail]  MAIL_TYPE [END,FAIL] (one mail per array, not per task)
#   Environment  By default the jobs activate the conda environment that is active in the
#             shell you submit from, through the conda installation that owns it, so submit
#             from the shell where `python check_env.py` passes. Overrides, in this order:
#             ENV_ACTIVATE='any command that activates the env'
#             CONDA_ENV=name [with CONDA_SH=/path/to/conda.sh, default: the MSI anaconda's]
#             With no env active and none given: CONDA_SH + environment "esmfold2".
#   Scoring   SCORE_LIGANDS [all] ames pools every ligand chain in its ligand terms (lcd, ipLDDT), so in an
#             NUC,ION run contacts with the ion alone are rewarded; "nucleotide" scores the nucleotide only
#   Hub       HF_OFFLINE [1] jobs run with HF_HUB_OFFLINE=1: every Hugging Face file must already be
#             cached (python fetch_hub_files.py on a login node; 0 = allow downloads, which the
#             hub rate-limits on shared IPs)   CCD_PATH [unset] path of a ccd.pkl, becomes ESMCFOLD_CCD_PATH
#   Layout    WORKDIR [directory of this script]  OUTROOT [$WORKDIR/outputs/nuc_matrix]
#   Matrix    NUCLEOTIDES [ATP GTP]  ALPHABETS [GADVP GADVPSELT GADVPSELTRIQN ALL20]
#             CATIONS [none MG]  REPS [3]
#   ames      PS [100] population  NG [1000] generations  LEN0 [65] start length
#             MAXLEN [160]  MINLEN [unset] soft lower length limit (ames' --seq1_min_len: the score is
#             multiplied by 0.5 at MINLEN residues, 0.95 six residues above, 0.05 six below)
#             MUT [npm] (npm = substitutions, indels, duplications, deletions of chunks, permutations;
#             pmo = substitutions and single-residue indels only)  CKPI [1] checkpoint every N generations
#             BETA0 [0.8] BETAT [8.0] ANN_S [0.15*NG] ANN_E [NG-1]  selection-strength annealing
set -euo pipefail

SUBMIT=0; STATUS=0; ARRAY_ARG=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT=1 ;;
        --status) STATUS=1 ;;
        -*) echo "unknown option: $arg" >&2; exit 1 ;;
        *) ARRAY_ARG="$arg" ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── CONFIG ────────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$HERE}"
OUTROOT="${OUTROOT:-$WORKDIR/outputs/nuc_matrix}"
PARTITION="${PARTITION:-preempt-gpu}"
TIME="${TIME:-24:00:00}"
MEM="${MEM:-40G}"
GPUS="${GPUS:-gpu:1}"
MAX_PARALLEL="${MAX_PARALLEL:-25}"      # max simultaneous array jobs (don't flood the queue)
CPUS="${CPUS:-}"
ACCOUNT="${ACCOUNT:-}"
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"
CONDA_SH="${CONDA_SH:-/common/software/install/migrated/anaconda/python3-2020.07-mamba/etc/profile.d/conda.sh}"
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
    ENV_SOURCE="ENV_ACTIVATE"
elif [[ -n "${CONDA_ENV:-}" ]]; then
    ENV_ACTIVATE="source ${CONDA_SH} && conda activate ${CONDA_ENV}"
    ENV_SOURCE="CONDA_ENV=${CONDA_ENV}"
elif [[ -n "${CONDA_PREFIX:-}" ]] && {
        # The environment active in this shell, activated through the conda installation that owns it:
        # <root>/envs/<name> belongs to <root>. CONDA_EXE is only the fallback (for environments created
        # with -p elsewhere) because a shell can load a different conda than the one that made the
        # environment, e.g. the MSI anaconda while the environment lives in the user's own miniforge.
        conda_root=""
        [[ "${CONDA_PREFIX}" == */envs/* ]] && conda_root="${CONDA_PREFIX%/envs/*}"
        if [[ -z "${conda_root}" || ! -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
            conda_root=""
            [[ -n "${CONDA_EXE:-}" ]] && conda_root="$(dirname "$(dirname "${CONDA_EXE}")")"
        fi
        [[ -n "${conda_root}" && -f "${conda_root}/etc/profile.d/conda.sh" ]]
    }; then
    ENV_ACTIVATE="source ${conda_root}/etc/profile.d/conda.sh && conda activate ${CONDA_PREFIX}"
    ENV_SOURCE="the conda environment active in this shell"
else
    ENV_ACTIVATE="source ${CONDA_SH} && conda activate esmfold2"
    ENV_SOURCE="default: no environment active (or its conda.sh was not found), no CONDA_ENV given"
fi

NUCLEOTIDES="${NUCLEOTIDES:-ATP GTP}"
ALPHABETS="${ALPHABETS:-GADVP GADVPSELT GADVPSELTRIQN ALL20}"
CATIONS="${CATIONS:-none MG}"
REPS="${REPS:-3}"
PS="${PS:-100}"
NG="${NG:-1000}"
LEN0="${LEN0:-65}"
MAXLEN="${MAXLEN:-160}"
MINLEN="${MINLEN:-}"
MUT="${MUT:-npm}"
CKPI="${CKPI:-1}"
HF_OFFLINE="${HF_OFFLINE:-1}"
SCORE_LIGANDS="${SCORE_LIGANDS:-all}"
CCD_PATH="${CCD_PATH:-${ESMCFOLD_CCD_PATH:-}}"   # ESMCFOLD_CCD_PATH in your shell works too
BETA0="${BETA0:-0.8}"
BETAT="${BETAT:-8.0}"
CONTROL="${CONTROL:-none}"
SMOKE="${SMOKE:-0}"
JOB_NAME="${JOB_NAME:-nuc_evo}"
# ──────────────────────────────────────────────────────────────────────────────

[[ "$OUTROOT" = /* ]] || OUTROOT="$WORKDIR/$OUTROOT"
for nuc in $NUCLEOTIDES; do
    [[ "$nuc" == "ATP" || "$nuc" == "GTP" ]] || { echo "NUCLEOTIDES: ATP and GTP only, got '$nuc'" >&2; exit 1; }
done
if [[ "$SMOKE" == "1" ]]; then
    PS=8; NG=12; TIME=01:00:00; REPS=1   # 12 > 10: ames prints its timing line at generation 10
    OUTROOT="${OUTROOT}_smoke"; JOB_NAME="${JOB_NAME}_smoke"
fi
ANN_S="${ANN_S:-$((NG * 15 / 100))}"
ANN_E="${ANN_E:-$((NG - 1))}"
case "$CONTROL" in
    none)    SEL_ARGS="-b0 $BETA0 -ann -bt $BETAT -ann_s $ANN_S -ann_e $ANN_E" ;;
    neutral) SEL_ARGS="-b0 0"; OUTROOT="${OUTROOT}_neutral"; JOB_NAME="${JOB_NAME}_neutral" ;;
    *) echo "CONTROL must be none or neutral" >&2; exit 1 ;;
esac
MANIFEST="$OUTROOT/manifest.tsv"
# Built here, not with ${VAR:+...} inside the heredoc: bash drops double quotes there, which would break
# a path containing a space. printf %q escapes it properly.
if [[ -n "$CCD_PATH" && ! -f "$CCD_PATH" ]]; then
    echo "ERROR: CCD_PATH / ESMCFOLD_CCD_PATH is set to '$CCD_PATH', which is not a file" >&2
    exit 1
fi
case "$SCORE_LIGANDS" in
    all)        SCORE_FLAG="" ;;
    nucleotide) SCORE_FLAG="--score-nucleotide-only" ;;
    *) echo "SCORE_LIGANDS must be all or nucleotide" >&2; exit 1 ;;
esac
HF_OFFLINE_LINE=""
[[ "$HF_OFFLINE" == "1" ]] && HF_OFFLINE_LINE="export HF_HUB_OFFLINE=1   # files come from the local cache: no network, no rate limit"
CCD_LINE=""
[[ -n "$CCD_PATH" ]] && CCD_LINE="export ESMCFOLD_CCD_PATH=$(printf '%q' "$CCD_PATH")"

manifest_text() {   # row N (0-based) is array index N
    printf 'idx\tnucleotide\talphabet\tcation\trep\tligand\toutdir\n'
    local idx=0 nuc alphabet cation i rep ligand cond
    for nuc in $NUCLEOTIDES; do
      for alphabet in $ALPHABETS; do
        for cation in $CATIONS; do
          if [[ "$cation" == "none" ]]; then ligand="$nuc"; cond="$nuc"; else ligand="$nuc,$cation"; cond="${nuc}_$cation"; fi
          for i in $(seq 1 "$REPS"); do
            rep=$(printf '%02d' "$i")
            printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$nuc" "$alphabet" "$cation" "$rep" "$ligand" \
                   "$OUTROOT/$alphabet/$cond/run$rep"
            idx=$((idx + 1))
          done
        done
      done
    done
}

length_warnings() {
    [[ "$MUT" == "npm" ]] && echo "WARNING: MUT=npm includes '%' (delete a chunk) and 'r' (replace the chain by 3-5 random residues). In the MSI smoke test a single '%' took 65-residue chains to 7 and 10 residues in generations 1-2 (see README: Chain length). MUT=pmo avoids this."
    local ion has_ion=0
    for ion in $CATIONS; do [[ "$ion" != "none" ]] && has_ion=1; done
    if [[ "$SCORE_LIGANDS" == "all" && "$has_ion" == 1 ]]; then
        echo "NOTE: SCORE_LIGANDS=all: ames pools the ion into the ligand terms, so a protein that binds only the ion is rewarded (in the pilot, GADVP's whole ligand score was Mg contacts, none with ATP). SCORE_LIGANDS=nucleotide scores the nucleotide only."
    fi
    [[ -z "$MINLEN" ]] && echo "WARNING: no MINLEN, so nothing opposes short chains. MINLEN=50 is a reasonable floor for a 65-residue start."
    return 0
}

if [[ $STATUS == 1 ]]; then
    # --status <dir> or <manifest.tsv>: look at runs submitted to another OUTROOT (pilot, smoke, neutral, ...)
    target="$MANIFEST"
    if [[ -n "$ARRAY_ARG" ]]; then
        if [[ -d "$ARRAY_ARG" ]]; then target="${ARRAY_ARG%/}/manifest.tsv"; else target="$ARRAY_ARG"; fi
    fi
    exec python3 "$HERE/run_status.py" "$target"
fi

N_RUNS=$(( $(manifest_text | wc -l) - 1 ))
if [[ $SUBMIT == 0 ]]; then
    manifest_text | awk -F'\t' 'NR == 1 {printf "%4s  %-8s %-14s %-6s %s\n", "idx", "ligand", "alphabet", "rep", "run directory"; next}
        {printf "%4d  %-8s %-14s %-6s %s\n", $1, $6, $3, $5, $7}'
    echo
    echo "environment: ${ENV_ACTIVATE}   (${ENV_SOURCE})"
    echo "ccd.pkl:     ${CCD_PATH:-from the Hugging Face cache (python fetch_hub_files.py)}"
    echo "chain length: start ${LEN0}, limits ${MINLEN:-none}..${MAXLEN} (soft), mutations: ${MUT}"
    length_warnings
    echo "$N_RUNS runs planned: PS=$PS NG=$NG, $PARTITION, $GPUS, $MEM, $TIME, up to $MAX_PARALLEL at once."
    echo "Nothing submitted. Add --submit to submit; --status shows progress afterwards."
    exit 0
fi

# ── SUBMIT ────────────────────────────────────────────────────────────────────
length_warnings >&2
cd "$WORKDIR"
mkdir -p "$OUTROOT" logs
if [[ -f "$MANIFEST" ]] && ! manifest_text | cmp -s - "$MANIFEST"; then
    if [[ "${FORCE_MANIFEST:-0}" != "1" ]]; then
        echo "ERROR: $MANIFEST exists and differs from the matrix these settings describe;" >&2
        echo "array indices would point at different runs. Use the original settings, a new OUTROOT," >&2
        echo "or FORCE_MANIFEST=1 if you know what you are doing." >&2
        exit 1
    fi
fi
manifest_text > "$MANIFEST"

if [[ -n "$ARRAY_ARG" ]]; then
    ARRAY_SPEC="$ARRAY_ARG"
    echo "Resubmitting specific jobs: ${ARRAY_SPEC}"
else
    ARRAY_SPEC="0-$((N_RUNS - 1))%${MAX_PARALLEL}"
    echo "Submitting ${N_RUNS} jobs (array ${ARRAY_SPEC})"
fi

{
cat <<HEAD
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=${ARRAY_SPEC}
#SBATCH --output=${WORKDIR}/logs/${JOB_NAME}_%A_%a.out
#SBATCH --error=${WORKDIR}/logs/${JOB_NAME}_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
${CPUS:+#SBATCH --cpus-per-task=${CPUS}}
${ACCOUNT:+#SBATCH --account=${ACCOUNT}}
${MAIL_USER:+#SBATCH --mail-type=${MAIL_TYPE}}
${MAIL_USER:+#SBATCH --mail-user=${MAIL_USER}}

# ── Environment (strict mode only after conda: its activate scripts are not 'set -u' clean)
${ENV_ACTIVATE} || { echo "ERROR: could not activate the environment; check CONDA_ENV / CONDA_SH / ENV_ACTIVATE in submit_matrix.sh" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1   # redirected stdout is block-buffered; without this logs/*.out lags far behind the job
${HF_OFFLINE_LINE}
${CCD_LINE}

# ── Settings fixed at submit time
SCRIPTS="${HERE}"
WORKDIR="${WORKDIR}"
MANIFEST="${MANIFEST}"
PS="${PS}"; NG="${NG}"; LEN0="${LEN0}"; MAXLEN="${MAXLEN}"; MINLEN="${MINLEN}"; MUT="${MUT}"; CKPI="${CKPI}"; SCORE_FLAG="${SCORE_FLAG}"
SEL_ARGS="${SEL_ARGS}"
HEAD
cat <<'BODY'

set -euo pipefail
cd "$WORKDIR"

IFS=$'\t' read -r IDX NUC ALPHABET CATION REP LIGAND OUTDIR < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
[[ -n "${OUTDIR:-}" ]] || { echo "no manifest row for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

echo "=========================================="
echo "Job $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID  (restarts so far: ${SLURM_RESTART_COUNT:-0})"
echo "Run:  $NUC  $ALPHABET  cation=$CATION  rep=$REP  ligand=$LIGAND"
echo "Out:  $OUTDIR"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

if [[ -f "$OUTDIR/DONE" ]]; then
    echo "Run already complete, nothing to do"
    exit 0
fi

python "$SCRIPTS/check_env.py" --gpu
mkdir -p "$(dirname "$OUTDIR")"

# --resume: continue from $OUTDIR/progress.ckp after a preemption or a walltime hit
python "$SCRIPTS/run_ames_alphabet.py" --alphabet "$ALPHABET" --resume \
    --iseq1 "protein:randoms:${LEN0}:evolv" --seq1_rate 1 \
    --ligand "$LIGAND" \
    -pm1 "$MUT" --seq1_max_len "$MAXLEN" ${MINLEN:+--seq1_min_len "$MINLEN"} \
    -ps "$PS" -ng "$NG" -ckpi "$CKPI" $SEL_ARGS \
    ${SCORE_FLAG} --engine esmfold2 \
    -o "$OUTDIR"

visualames -l "$OUTDIR/progress.log"
touch "$OUTDIR/DONE"
echo "Task $SLURM_ARRAY_TASK_ID finished successfully"
BODY
} | sbatch

echo "Submitted. Monitor with:  squeue -u \$USER -n ${JOB_NAME}"
echo "Check progress with:      bash $(basename "$0") --status"
