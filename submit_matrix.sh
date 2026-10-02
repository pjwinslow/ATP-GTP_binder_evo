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
#   bash submit_matrix.sh --submit 12,45,99     # resubmit only these array indices
#
# Preempted or requeued jobs, and runs that hit the walltime and are resubmitted, RESUME
# from their last checkpoint (ames has no resume of its own: see ames_resume.py), and a
# finished run is skipped, so resubmitting is always safe.
#
#   SMOKE=1 bash submit_matrix.sh --submit      # tiny test: 8 sequences x 4 generations, 30 min
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
#   Environment  CONDA_SH [/common/software/.../python3-2020.07-mamba/etc/profile.d/conda.sh]
#             CONDA_ENV [esmfold2]   or ENV_ACTIVATE='any command that activates the env'
#   Layout    WORKDIR [directory of this script]  OUTROOT [$WORKDIR/outputs/nuc_matrix]
#   Matrix    NUCLEOTIDES [ATP GTP]  ALPHABETS [GADVP GADVPSELT GADVPSELTRIQN ALL20]
#             CATIONS [none MG]  REPS [3]
#   ames      PS [100] population  NG [1000] generations  LEN0 [65] start length
#             MAXLEN [160]  MUT [npm] (npm | pmo)  CKPI [1] checkpoint every N generations
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
CONDA_ENV="${CONDA_ENV:-esmfold2}"
ENV_ACTIVATE="${ENV_ACTIVATE:-source ${CONDA_SH} && conda activate ${CONDA_ENV}}"

NUCLEOTIDES="${NUCLEOTIDES:-ATP GTP}"
ALPHABETS="${ALPHABETS:-GADVP GADVPSELT GADVPSELTRIQN ALL20}"
CATIONS="${CATIONS:-none MG}"
REPS="${REPS:-3}"
PS="${PS:-100}"
NG="${NG:-1000}"
LEN0="${LEN0:-65}"
MAXLEN="${MAXLEN:-160}"
MUT="${MUT:-npm}"
CKPI="${CKPI:-1}"
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
    PS=8; NG=4; TIME=00:30:00; REPS=1
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

if [[ $STATUS == 1 ]]; then
    exec python3 "$HERE/run_status.py" "$MANIFEST"
fi

N_RUNS=$(( $(manifest_text | wc -l) - 1 ))
if [[ $SUBMIT == 0 ]]; then
    manifest_text | awk -F'\t' 'NR == 1 {printf "%4s  %-8s %-14s %-6s %s\n", "idx", "ligand", "alphabet", "rep", "run directory"; next}
        {printf "%4d  %-8s %-14s %-6s %s\n", $1, $6, $3, $5, $7}'
    echo
    echo "$N_RUNS runs planned: PS=$PS NG=$NG, $PARTITION, $GPUS, $MEM, $TIME, up to $MAX_PARALLEL at once."
    echo "Nothing submitted. Add --submit to submit; --status shows progress afterwards."
    exit 0
fi

# ── SUBMIT ────────────────────────────────────────────────────────────────────
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
${ENV_ACTIVATE}
export PYTHONNOUSERSITE=1

# ── Settings fixed at submit time
SCRIPTS="${HERE}"
WORKDIR="${WORKDIR}"
MANIFEST="${MANIFEST}"
PS="${PS}"; NG="${NG}"; LEN0="${LEN0}"; MAXLEN="${MAXLEN}"; MUT="${MUT}"; CKPI="${CKPI}"
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
    -pm1 "$MUT" --seq1_max_len "$MAXLEN" \
    -ps "$PS" -ng "$NG" -ckpi "$CKPI" $SEL_ARGS \
    --engine esmfold2 \
    -o "$OUTDIR"

visualames -l "$OUTDIR/progress.log"
touch "$OUTDIR/DONE"
echo "Task $SLURM_ARRAY_TASK_ID finished successfully"
BODY
} | sbatch

echo "Submitted. Monitor with:  squeue -u \$USER -n ${JOB_NAME}"
echo "Check progress with:      bash $(basename "$0") --status"
