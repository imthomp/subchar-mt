#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=mt_v5
#SBATCH --output=logs/finetune_v5_%A_%a.out
#SBATCH --error=logs/finetune_v5_%A_%a.err

# v5: decomposition-depth ablation (Han et al. arXiv 2512.15556)
# 4 reps (baseline, radicals_d1, radicals_d2, radicals_full) × 2 models × 3 sizes × 5 seeds
# = 120 conditions

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "Job: $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID  node $SLURM_NODELIST  $(date)"

module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

mkdir -p models_v5 logs

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
    except: print(f"  {name}: MISSING"); import sys; sys.exit(1)
EOF
[ $? -ne 0 ] && exit 1

LOG="logs/v5_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/finetune_experiment_v5.py 2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
[ $EXIT_CODE -eq 0 ] && echo "✓ Task $SLURM_ARRAY_TASK_ID done." || \
    { echo "✗ Failed (exit $EXIT_CODE)"; exit $EXIT_CODE; }
echo "Finished: $(date)  Runtime: ${SECONDS}s"
