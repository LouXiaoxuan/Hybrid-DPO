#!/bin/sh
export VLLM_USE_MODELSCOPE=False
export TOKENIZERS_PARALLELISM=False

# python3 -u sample.py

# python3 -u sample.py --dataset_name "gsm8k" --n_samples 40 --model_name "Qwen2-Math-7B" --model_path "/mnt/data/xiaoxuan/llms/Qwen2-Math-7B" --prompt_type "qwen-boxed"
python3 -u sample.py --dataset_name "math" --n_samples 20 --model_name "Qwen2-Math-7B" --model_path "/mnt/data/xiaoxuan/llms/Qwen2-Math-7B" --prompt_type "qwen-boxed"