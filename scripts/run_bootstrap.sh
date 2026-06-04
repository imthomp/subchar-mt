#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=bootstrap
#SBATCH --output=logs/bootstrap_%j.out
#SBATCH --error=logs/bootstrap_%j.err

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "Bootstrap significance test started: $(date)"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/significance_test.py \
    --mode bootstrap \
    --preds_dir results/predictions/ \
    --out results/bootstrap_results.csv
echo "Done: $(date)"
