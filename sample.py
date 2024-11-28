import os
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, set_seed

from vllm import LLM, SamplingParams
from data_loader import load_q_cot_a


def main(args):
    set_seed(args.seed)
    llm = LLM(
        model=args.model_path,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.available_gpus // args.pipeline_parallel_size,
    )

    question_prompts, targets, answers = load_q_cot_a(args.dataset_dir, args.dataset_name, args.prompt_type, split="train")
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    save_path = args.save_dir +"{model_name}/{dataset_name}/".format(model_name=args.model_name, dataset_name=args.dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    for i in range(16,30):
        fout = open(save_path +"sample_{num}.json".format(num=i), "w")

        outputs = llm.generate(
            question_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_length,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_path.lower()
                    else None
                ),
            ),
        )

        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        final_list = []

        for idx, output in enumerate(outputs):
            item = {"prompt":output.prompt, "gt_cot":targets[idx], "gt_ans":answers[idx]}
            item["sample"] = output.outputs[0].text
            final_list.append(item)
        
        json.dump(final_list, fout, indent=2)
        fout.close()


if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        dataset_name: Optional[str] = field(default="math", metadata={"help": "the dataset name"})
        n_samples: Optional[int] = field(default=20, metadata={"help": "Number of samples; negative means all"})

        # model_name: Optional[str] = field(default="Llama3.1-8B-gsm8k-dpo1", metadata={"help": "the dataset name"})
        # model_path: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/llama3.1-8b-math-sft-checkpoint/checkpoint-354", metadata={"help": "the model path"})
        # prompt_type: Optional[str] = field(default="llama", metadata={"help": "the prompt type"})

        model_name: Optional[str] = field(default="Qwen2-Math-7B", metadata={"help": "the model name"})
        model_path:Optional[str]=field(default="/mnt/2050data/xiaoxuan/llms/Qwen2-Math-7B", metadata={"help":"the model name"})
        prompt_type: Optional[str] = field(default="qwen-boxed", metadata={"help": "the prompt type"})

        # model_path: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/deepseek-math-7b-rl", metadata={"help": "the model name"})
        # prompt_type: Optional[str] = field(default="deepseek-math", metadata={"help": "the prompt type"})
        
        save_dir: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/samples/", metadata={"help": "save dir for samples"})
        pipeline_parallel_size: Optional[int] = field(default=1, metadata={"help": "pipeline parrallel size"})
        available_gpus: Optional[int] = field(default=4, metadata={"help": "available gpus"})
        dataset_dir: Optional[str] = field(default="/mnt/2050data/xiaoxuan/MAIDPO/data", metadata={"help": "the dataset name"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        temperature: Optional[float] = field(default=0.8, metadata={"help": "temperature"})
        top_p: Optional[float] = field(default=1.0, metadata={"help": "top_p"})
        
    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    main(args)