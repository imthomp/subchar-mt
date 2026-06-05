#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=m12
#SBATCH --qos=normal
#SBATCH --job-name=probing
#SBATCH --output=logs/probing_%j.out
#SBATCH --error=logs/probing_%j.err

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

module purge
module load python/3.12

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Probing started: $(date)"
PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/run_probing.py
echo "Done: $(date)"
