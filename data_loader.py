import torch
import json
import os
import random
import datasets
from itertools import product
from random import sample
from torch.utils.data import Dataset
from utils_tool import load_jsonl, parse_question, parse_ground_truth, construct_prompt

def get_dpo_dataset(data_file, num_pairs):
    with open(data_file, "r") as f:
        raw_data = json.load(f)

    data = []
    flag = 0
    for item in raw_data:

        if len(item["chosen_samples"]) == 0:
            item["chosen_samples"].append(item["gt_cot"])

        if len(item["rejected_samples"]) == 0:
            continue

        all_answer_pairs = list(product(item["chosen_samples"], item["rejected_samples"]))
        random_pairs = sample(all_answer_pairs, num_pairs)
        
        for pair in random_pairs:
            temp_item = {}
            temp_item["prompt"] = item["prompt"]
            temp_item["chosen"] = pair[0]
            temp_item["rejected"] = pair[1]
            data.append(temp_item)

            if flag == 1:
                print(all_answer_pairs)
                print(random_pairs)
                print(temp_item)
                flag = 0

    return datasets.Dataset.from_list(data)

def get_another_dpo_dataset(prompt_data_file, sample_data_file, num_pairs):
    with open(prompt_data_file, "r") as f:
        prompt_data = json.load(f)

    with open(sample_data_file, "r") as f:
        sample_data = json.load(f)

    data = []
    flag = 0
    for prompt_item, sample_item in zip(prompt_data, sample_data):

        if len(sample_item["chosen_samples"]) == 0:
            sample_item["chosen_samples"].append(sample_item["gt_cot"])

        if len(sample_item["rejected_samples"]) == 0:
            continue

        all_answer_pairs = list(product(sample_item["chosen_samples"], sample_item["rejected_samples"]))
        random_pairs = sample(all_answer_pairs, num_pairs)
        
        for pair in random_pairs:
            temp_item = {}
            temp_item["prompt"] = prompt_item["prompt"]
            temp_item["chosen"] = pair[0]
            temp_item["rejected"] = pair[1]
            data.append(temp_item)

            if flag == 1:
                print(all_answer_pairs)
                print(random_pairs)
                print(temp_item)
                flag = 0

    return datasets.Dataset.from_list(data)

def get_hybrid_dpo_dataset(chosen_data_file, rejected_data_dile, num_pairs):
    with open(chosen_data_file, "r") as f:
        chosen_raw_data = json.load(f)

    with open(rejected_data_dile, "r") as f:
        rejected_raw_data = json.load(f)

    data = []
    flag = 0
    for chosen_item, rejected_item in zip(chosen_raw_data, rejected_raw_data):

        if len(chosen_item["chosen_samples"]) == 0:
            chosen_item["chosen_samples"].append(chosen_item["gt_cot"])

        if len(rejected_item["rejected_samples"]) == 0:
            continue

        all_answer_pairs = list(product(chosen_item["chosen_samples"], rejected_item["rejected_samples"]))
        if len(all_answer_pairs) < num_pairs:
            if len(rejected_item["chosen_samples"]) == 0:
                rejected_item["chosen_samples"].append(rejected_item["gt_cot"])
            updated_answer_pairs = list(product(rejected_item["chosen_samples"], rejected_item["rejected_samples"]))
            random_pairs = sample(updated_answer_pairs, num_pairs)
        else:
            random_pairs = sample(all_answer_pairs, num_pairs)
        
        for pair in random_pairs:
            temp_item = {}
            temp_item["prompt"] = rejected_item["prompt"]
            temp_item["chosen"] = pair[0]
            temp_item["rejected"] = pair[1]
            data.append(temp_item)

            if flag == 1:
                print(all_answer_pairs)
                print(random_pairs)
                print(temp_item)
                flag = 0

    return datasets.Dataset.from_list(data)

def load_q_cot_a(dataset_dir, dataset_name, prompt_type, split):

    data_file = f"{dataset_dir}/{dataset_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        print("the dataset file cannot be found!")
        exit()

    questions = [parse_question(example, dataset_name) for example in examples]
    question_prompts = [construct_prompt(question, dataset_name, prompt_type) for question in questions]
    ground_truth = [parse_ground_truth(example, dataset_name) for example in examples]
    targets, answers = zip(*ground_truth)
    targets = list(targets)
    answers = list(answers)

    return question_prompts, targets, answers

class HybridSFTDataset(Dataset):

    def __init__(self, sample_file, tokenizer, max_length, max_prompt_length):
        super(HybridSFTDataset, self).__init__()
        self.sample_file = sample_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.data = []

        with open(sample_file, "r") as f:
            raw_data = json.load(f)

        for item in raw_data:
            q = item["prompt"]
            if len(item["chosen_samples"]) == 0:
                item["chosen_samples"].append(item["gt_cot"])
            cot = sample(item["chosen_samples"], 1)[0]
            q_input_ids, generation_labels = self.tokenize_element(q, cot, "keep_end")
            full_labels = [-100] * len(q_input_ids) + generation_labels
            full_input_ids = q_input_ids + generation_labels
            self.data.append({"input_ids":torch.LongTensor(full_input_ids), "labels":torch.LongTensor(full_labels)})

    def tokenize_element(self, prompt: str, generation: str, truncation_mode: str):

        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()
        
        if generation == "":
            return prompt_token_ids, "none"

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)
        
        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]


        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = self.tokenizer.decode(generation_token_ids, skip_special_tokens=True) + ' ' + self.tokenizer.eos_token

        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)
  
        return prompt_token_ids, generation_token_ids

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

class SupervisedDataset(Dataset):

    def __init__(self, dataset_dir, dataset_name, tokenizer, max_length, max_prompt_length, prompt_type, split):
        super(SupervisedDataset, self).__init__()
        self.data_dir = dataset_dir
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.prompt_type = prompt_type
        self.data=[]

        question_prompts, targets, answers = load_q_cot_a(dataset_dir, dataset_name, prompt_type, split)
        print(targets[0])

        if split == "train":
            for q, cot, a in zip(question_prompts, targets, answers):
                q_input_ids, generation_labels = self.tokenize_element(q, cot, "keep_end")
                full_labels = [-100] * len(q_input_ids) + generation_labels
                full_input_ids = q_input_ids + generation_labels
                self.data.append({"input_ids":torch.LongTensor(full_input_ids), "labels":torch.LongTensor(full_labels)})
        elif split == "test":
            for q, a in zip(question_prompts, answers):
                q_input_ids, _ = self.tokenize_element(q, "", "keep_end")
                self.data.append({"input_ids": q_input_ids, "answers":a})

    def tokenize_element(self, prompt: str, generation: str, truncation_mode: str):

        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()
        
        if generation == "":
            return prompt_token_ids, "none"

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)
        
        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]


        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = self.tokenizer.decode(generation_token_ids, skip_special_tokens=True) + ' ' + self.tokenizer.eos_token

        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)
  
        return prompt_token_ids, generation_token_ids

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)



if __name__ == "__main__":

    data_file = "/mnt/data/xiaoxuan/sft-code/samples/llama3.1-8b-sft-gsm8k-total-sample-0.json"

    get_dpo_dataset(data_file, 2)
