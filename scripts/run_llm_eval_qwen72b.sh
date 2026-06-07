#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=llm_qwen32b
#SBATCH --output=logs/llm_qwen32b_%j.out
#SBATCH --error=logs/llm_qwen32b_%j.err

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "Qwen2.5-32B LLM eval started: $(date)  node: $SLURM_NODELIST"
nvidia-smi | grep "MiB\|GPU" | head -6

module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p results/llm_eval logs

PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/llm_eval.py \
    --preds_dir results/predictions_v3/ \
    --model_id Qwen/Qwen2.5-32B-Instruct \
    --out_name judgments_qwen32b.csv \
    --n_sentences 100 \
    --train_size 1000 \
    2>&1 | tee logs/llm_eval_qwen32b.log

echo "Done: $(date)"
