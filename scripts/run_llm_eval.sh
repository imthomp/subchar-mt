#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=llm_eval
#SBATCH --output=logs/llm_eval_%j.out
#SBATCH --error=logs/llm_eval_%j.err

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "LLM eval started: $(date)  node: $SLURM_NODELIST"

module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p results/llm_eval logs

PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/llm_eval.py \
    --preds_dir results/predictions_v3/ \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --n_sentences 100 \
    --train_size 1000 \
    --out_name judgments_llama8b.csv \
    2>&1 | tee logs/llm_eval.log

echo "Done: $(date)"
