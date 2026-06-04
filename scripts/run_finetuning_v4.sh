#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=mt_v4
#SBATCH --output=logs/finetune_v4_%A_%a.out
#SBATCH --error=logs/finetune_v4_%A_%a.err

# v4: selective decomposition (Saunders 2020)
# 6 reps × 2 models × 3 train sizes × 5 seeds = 180 conditions

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

mkdir -p models_v4 logs results/predictions_v4

# Verify cache
$PYTHON - <<'EOF'
import os; os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
from datasets import load_dataset
from transformers import MarianTokenizer, AutoTokenizer
for name, fn in [
    ("WMT19",   lambda: load_dataset('wmt19','zh-en',split='train[:1]')),
    ("opus-mt", lambda: MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')),
    ("NLLB",    lambda: AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')),
]:
    try: fn(); print(f"  {name}: OK")
    except Exception as e: print(f"  {name}: MISSING"); import sys; sys.exit(1)
EOF
[ $? -ne 0 ] && exit 1

LOG="logs/experiment_v4_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/finetune_experiment_v4.py \
    --save_predictions --predictions_dir results/predictions_v4 \
    2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
[ $EXIT_CODE -eq 0 ] && echo "✓ Task $SLURM_ARRAY_TASK_ID done." || \
    { echo "✗ Task failed (exit $EXIT_CODE)"; exit $EXIT_CODE; }
echo "Finished: $(date)  Runtime: ${SECONDS}s"
