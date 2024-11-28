import json
import numpy as np
from typing import Optional, Set
from transformers import set_seed
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    set_seed,
)
from vllm import LLM, SamplingParams
from data_loader import load_q_cot_a
from utils_tool import extract_answer, is_math_equal


def main(args):
    set_seed(args.seed)

    llm = LLM(
        model=args.model_name,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.available_gpus // args.pipeline_parallel_size,
    )
    
    question_prompts, _, answers = load_q_cot_a(args.dataset_dir, args.dataset_name, args.prompt_type, split="test")
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    outputs = llm.generate(
        question_prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_length,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in args.model_name .lower()
                else None
            ),
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))

    pred_list = []
    correct_cnt = 0

    for output in outputs:      
        policy_sample = output.outputs[0].text
        pred = extract_answer(policy_sample)
        pred_list.append(pred)

    scores, _ = is_math_equal(pred_list, answers)
    accuracy = np.array(scores).mean()
    for idx, score in enumerate(scores):
        if score:
            correct_cnt += 1
        #     print(idx, pred_list[idx], "and",  answers[idx])
        # else:
        #     print(idx, "Error!", pred_list[idx], "and",  answers[idx])

    print("Final output:")            
    print(idx+1) 
    print(correct_cnt)
    print(accuracy)            


if __name__ == '__main__':

    @dataclass
    class ScriptArguments:
        dataset_name: Optional[str] = field(default="math", metadata={"help": "the dataset name"})

        # # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/Meta-Llama-3.1-8B", metadata={"help": "the model name"})
        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/llama3.1-8b-math-sft-checkpoint/checkpoint-236", metadata={"help": "the model name"})
        # # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/llama3.1-8b-sft-gsm8k-dpo-checkpoint-1-2pairs/checkpoint-108", metadata={"help": "the model name"})
        # prompt_type: Optional[str] = field(default="llama", metadata={"help": "the prompt type"})

        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/Qwen2-Math-7B", metadata={"help": "the model name"})
        # # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-gsm8k-dpo-checkpoint-1-1pair-v2/checkpoint-116", metadata={"help": "the model name"})
        # prompt_type: Optional[str] = field(default="qwen-boxed", metadata={"help": "the prompt type"})

        model_name: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/checkpoints/qwen2.5-7b-instruct-math-hybrid-rpo-checkpoint-1-15pair/checkpoint-504", metadata={"help": "the model name"})
        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-gsm8k-checkpoint/checkpoint-59", metadata={"help": "the model name"})
        prompt_type: Optional[str] = field(default="qwen-boxed", metadata={"help": "the prompt type"})

        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/deepseek-math-7b-rl", metadata={"help": "the model name"})
        # prompt_type: Optional[str] = field(default="deepseek-math", metadata={"help": "the prompt type"})
        
        pipeline_parallel_size: Optional[int] = field(default=1, metadata={"help": "pipeline parrallel size"})
        available_gpus: Optional[int] = field(default=4, metadata={"help": "available gpus"})
        dataset_dir: Optional[str] = field(default="/mnt/2050data/xiaoxuan/MAIDPO/data", metadata={"help": "the dataset name"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        temperature: Optional[float] = field(default=0.0, metadata={"help": "temperature"})
        top_p: Optional[float] = field(default=1.0, metadata={"help": "top_p"})


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    main(args)
