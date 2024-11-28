#!/bin/sh
export VLLM_USE_MODELSCOPE=False
export TOKENIZERS_PARALLELISM=False
# python3 -u inference.py --dataset_name "math" --prompt_type "llama" --model_name "/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-math-rpo-checkpoint-1-1pair/checkpoint-57"
# python3 -u inference.py --dataset_name "gsm8k" --prompt_type "llama" --model_name "/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-math-rpo-checkpoint-1-1pair/checkpoint-57"
# python3 -u inference.py --dataset_name "math" --model_name "/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-math-rpo-checkpoint-1-1pair/checkpoint-114"
# python3 -u inference.py --dataset_name "gsm8k" --model_name "/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-math-rpo-checkpoint-1-1pair/checkpoint-114"

# python3 -u inference.py

TARGET_DIR="/mnt/2050data/xiaoxuan/sft-code/checkpoints/deepseek-math-7b-rl-math-hybrid-rpo-checkpoint-1-15pair"

for dir in "$TARGET_DIR"/*/; do
    if [ -d "$dir" ]; then
        abs_path=$(realpath "$dir")
        echo $abs_path
        python3 -u inference.py --dataset_name "math" --prompt_type "deepseek-math" --model_name "$abs_path"
    fi
done