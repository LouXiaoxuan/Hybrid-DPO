import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from utils_tool import extract_answer, is_math_equal

def main(args):

    sample_path = args.sample_dir +"{model_name}/{dataset_name}/".format(model_name=args.model_name, dataset_name=args.dataset_name)
    fout = open(args.output_dir, "w")

    prompt_lists = []
    gt_cot_lists = []
    gt_ans_lists = []
    sample_lists = []
    pred_list = []

    for idx in range(args.n_samples):
        fin = open(sample_path + "sample_{num}.json".format(num=idx), "r")
        raw_data = json.load(fin)
        for i, item in enumerate(raw_data):
            if idx == 0:
                prompt_lists.append(item["prompt"])
                gt_cot_lists.append(item["gt_cot"])
                gt_ans_lists.append(item["gt_ans"])
            # sample_lists.append(item["sample"][0])
            # pred = extract_answer(item["sample"][0])
            sample_lists.append(item["sample"])
            pred = extract_answer(item["sample"])
            pred_list.append(pred)     
        fin.close()
            
    ans_list = gt_ans_lists * args.n_samples
    scores, _ = is_math_equal(pred_list, ans_list)
    final_list = []

    for idx, prompt in enumerate(prompt_lists):
        temp_item = {"prompt":prompt, "gt_cot":gt_cot_lists[idx], "gt_ans":gt_ans_lists[idx], "chosen_samples":[], "rejected_samples":[]}
        final_list.append(temp_item)
        
    for idx, score in enumerate(scores):
        if score:
            final_list[idx % len(prompt_lists)]["chosen_samples"].append(sample_lists[idx])
        else:
            final_list[idx % len(prompt_lists)]["rejected_samples"].append(sample_lists[idx])

    json.dump(final_list, fout, indent=2)


if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        dataset_name: Optional[str] = field(default="gsm8k", metadata={"help": "the dataset name"})
        n_samples: Optional[int] = field(default=40, metadata={"help": "Number of samples; negative means all"})

        model_name:Optional[str]=field(default="deepseek-math-7b-rl", metadata={"help":"the model name"})
        output_dir: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/samples/deepseek-math-7b-rl-orig-gsm8k-total-sample-0.json", metadata={"help": "directory"})

        # model_name:Optional[str]=field(default="Qwen2.5-Math-7B-instruct-dpo1", metadata={"help":"the model name"})
        # output_dir: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/samples/qwen2.5-math-7b-instruct-dpo1-gsm8k-total-sample-0.json", metadata={"help": "directory"})

        # model_name:Optional[str]=field(default="deepseek-math-7b-rl", metadata={"help":"the model name"})
        # output_dir: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/samples/deepseek-math-7b-rl-gsm8k-total-sample-0.json", metadata={"help": "directory"})

        sample_dir: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/samples/", metadata={"help": "save dir for samples"})

    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    main(args)