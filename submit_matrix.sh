#!/bin/bash
# Submit the ATP-binding experiment matrix to Slurm, one GPU job per run:
#
#     alphabet (GADVP, GADVPSELT, GADVPSELTRIQN, ALL20)
#   x cation   (none -> ligand ATP;  MG -> ligand ATP,MG;  MN, CA, ... also work)
#   x replicate
#
# Prints the plan and exits unless --submit is given.
#
#   ENV_ACTIVATE='conda activate ames' ./submit_matrix.sh            # show plan
#   ENV_ACTIVATE='conda activate ames' ./submit_matrix.sh --submit
#   SMOKE=1 ENV_ACTIVATE=... ./submit_matrix.sh --submit             # tiny test run
#   CONTROL=neutral ... ./submit_matrix.sh --submit                  # no-selection null
#
# Settings (environment variables, defaults in brackets)
#   ENV_ACTIVATE  command that activates the env with ames + esm       [required]
#   OUTROOT       output root                                          [outputs/atp_matrix]
#   ALPHABETS     space-separated                                      [GADVP GADVPSELT GADVPSELTRIQN ALL20]
#   CATIONS       "none" = ATP only, otherwise CCD code of the ion     [none MG]
#   REPS          replicates per condition                             [3]
#   PS NG         population size, generations                         [100 1000]
#   LEN0 MAXLEN   starting / maximum chain length                      [65 160]
#   MUT           ames protein mutation set: npm | pmo                 [npm]
#   BETA0 BETAT   selection strength at start / after annealing        [0.8 8.0]
#   ANN_S ANN_E   annealing window (generations)                       [0.15*NG, NG-1]
#   CONTROL       none | neutral (beta=0, no annealing: mutation +     [none]
#                 drift only, the null distribution for ATP scores)
#   SMOKE         1 = PS=8 NG=4, 30 min walltime, outputs to $OUTROOT_smoke
#   Slurm: PARTITION GRES TIME MEM CPUS ACCOUNT MODULES (check sinfo / MSI docs)
set -euo pipefail

SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

: "${ENV_ACTIVATE:?set ENV_ACTIVATE, e.g. ENV_ACTIVATE='conda activate ames'}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTROOT="${OUTROOT:-outputs/atp_matrix}"
ALPHABETS="${ALPHABETS:-GADVP GADVPSELT GADVPSELTRIQN ALL20}"
CATIONS="${CATIONS:-none MG}"
REPS="${REPS:-3}"
PS="${PS:-100}"
NG="${NG:-1000}"
LEN0="${LEN0:-65}"
MAXLEN="${MAXLEN:-160}"
MUT="${MUT:-npm}"
BETA0="${BETA0:-0.8}"
BETAT="${BETAT:-8.0}"
CONTROL="${CONTROL:-none}"
SMOKE="${SMOKE:-0}"

PARTITION="${PARTITION:-a100-4}"
GRES="${GRES:-gpu:a100:1}"
TIME="${TIME:-72:00:00}"
MEM="${MEM:-50g}"
CPUS="${CPUS:-8}"
ACCOUNT="${ACCOUNT:-}"
MODULES="${MODULES:-}"

if [[ "$SMOKE" == "1" ]]; then
    PS=8; NG=4; TIME=00:30:00; REPS=1
    OUTROOT="${OUTROOT}_smoke"
fi
ANN_S="${ANN_S:-$((NG * 15 / 100))}"
ANN_E="${ANN_E:-$((NG - 1))}"

case "$CONTROL" in
    none)    SEL_ARGS="-b0 $BETA0 -ann -bt $BETAT -ann_s $ANN_S -ann_e $ANN_E" ;;
    neutral) SEL_ARGS="-b0 0"; OUTROOT="${OUTROOT}_neutral" ;;
    *) echo "CONTROL must be none or neutral" >&2; exit 1 ;;
esac

printf '%-44s %-16s %s\n' "run directory" "ligand" "alphabet"
n_jobs=0
for alphabet in $ALPHABETS; do
  for cation in $CATIONS; do
    if [[ "$cation" == "none" ]]; then ligand="ATP"; cond="ATP"; else ligand="ATP,$cation"; cond="ATP_$cation"; fi
    for i in $(seq 1 "$REPS"); do
      rep=$(printf '%02d' "$i")
      outdir="$OUTROOT/$alphabet/$cond/run$rep"
      printf '%-44s %-16s %s\n' "$outdir" "$ligand" "$alphabet"
      n_jobs=$((n_jobs + 1))
      [[ $SUBMIT == 1 ]] || continue

      mkdir -p "$(dirname "$outdir")"
      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=atp_${alphabet}_${cond}_r${rep}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${outdir}.out
${ACCOUNT:+#SBATCH --account=${ACCOUNT}}
set -euo pipefail
${MODULES:+module load ${MODULES}}
${ENV_ACTIVATE}
nvidia-smi -L || true

# no --nobackup: ames moves an existing output dir aside instead of deleting it
python "${HERE}/run_ames_alphabet.py" --alphabet ${alphabet} \\
    --iseq1 'protein:randoms:${LEN0}:evolv' --seq1_rate 1 \\
    --ligand ${ligand} \\
    -pm1 ${MUT} --seq1_max_len ${MAXLEN} \\
    -ps ${PS} -ng ${NG} ${SEL_ARGS} \\
    --engine esmfold2 \\
    -o "${outdir}"

visualames -l "${outdir}/progress.log"
EOF
    done
  done
done

if [[ $SUBMIT == 1 ]]; then
    echo "submitted $n_jobs jobs"
else
    echo "$n_jobs jobs planned (PS=$PS NG=$NG TIME=$TIME PARTITION=$PARTITION GRES=$GRES), re-run with --submit"
fi
