#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=mt_zhja
#SBATCH --output=logs/finetune_zhja_%A_%a.out
#SBATCH --error=logs/finetune_zhja_%A_%a.err

# Zh→Ja CJK extension: shared radical transfer experiment
# 4 reps × 2 models × 3 train sizes × 5 seeds = 120 conditions
# Data: FLORES+ cmn_Hans / jpn_Jpan (pre-saved as CSVs in data/)

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "========================================================================"
echo "Job: $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID  node $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================================================"

module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

mkdir -p models_zhja logs results/predictions_zhja

# Verify data and model cache
$PYTHON - <<'EOF'
import os, sys
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
from transformers import MarianTokenizer, AutoTokenizer
import pandas as pd

missing = []
for name, fn in [
    ("opus-mt-zh-ja", lambda: MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-big-zh-ja')),
    ("NLLB",          lambda: AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')),
]:
    try: fn(); print(f"  {name}: OK")
    except: missing.append(name); print(f"  {name}: MISSING")

for f in ['data/flores_zhja_train.csv', 'data/flores_zhja_test.csv']:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"  {f}: {len(df)} rows OK")
    else:
        missing.append(f); print(f"  {f}: MISSING")

if missing: sys.exit(1)
print("All assets OK.")
EOF
[ $? -ne 0 ] && exit 1

LOG="logs/zhja_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/finetune_experiment_zhja.py \
    --save_predictions --preds_dir results/predictions_zhja \
    2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
[ $EXIT_CODE -eq 0 ] && echo "✓ Task $SLURM_ARRAY_TASK_ID done." || \
    { echo "✗ Task failed (exit $EXIT_CODE)"; exit $EXIT_CODE; }
echo "Finished: $(date)  Runtime: ${SECONDS}s"
