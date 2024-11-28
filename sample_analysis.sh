#!/bin/sh
export VLLM_USE_MODELSCOPE=False
export TOKENIZERS_PARALLELISM=False
python3 -u sample_analysis.py