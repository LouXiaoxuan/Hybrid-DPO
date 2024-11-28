from typing import Optional
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)
from trl import DPOConfig, DPOTrainer
from data_loader import get_dpo_dataset


def main(args):
    set_seed(args.seed)

    poplicy_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name)
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = get_dpo_dataset(args.data_file, args.num_pairs)    

    training_args = DPOConfig(
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reetrant),
        bf16=args.bf16,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        remove_unused_columns=False,
        run_name="DPO_Ex",
        truncation_mode="keep_end",
        report_to=args.report_to,
        rpo_alpha=1.0,
    )
    
    trainer = DPOTrainer(
        poplicy_model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        
        model_name: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/checkpoints/llama3.1-8b-instruct-qwen-prompt-math-hybrid-rpo-checkpoint-1-15pair/checkpoint-1509", metadata={"help": "the policy model name"})
        ref_model_name: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/checkpoints/llama3.1-8b-instruct-qwen-prompt-math-hybrid-rpo-checkpoint-1-15pair/checkpoint-1509", metadata={"help": "the reference model name"})
        data_file: Optional[str] = field(default="/mnt/2050data/xiaoxuan/sft-code/samples/llama3.1-8b-instruct-hybrid-rpo1-math-total-sample-0.json", metadata={"help": "the model name"})
        num_pairs: Optional[int] = field(default=15, metadata={"help": "number of DPO-data pairs for each prompt"})
        output_dir: Optional[str] = field(default="./checkpoints/llama3.1-8b-instruct-math-dpo-checkpoint-2-15pair", metadata={"help": "directory"})

        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-gsm8k-dpo-checkpoint-1-1pair/checkpoint-116", metadata={"help": "the policy model name"})
        # ref_model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/checkpoints/qwen2-math-7b-orig-gsm8k-dpo-checkpoint-1-1pair/checkpoint-116", metadata={"help": "the reference model name"})
        # data_file: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/samples/qwen2-math-7b-dpo1-gsm8k-total-sample-0.json", metadata={"help": "the model name"})
        # num_pairs: Optional[int] = field(default=1, metadata={"help": "number of DPO-data pairs for each prompt"})
        # output_dir: Optional[str] = field(default="./checkpoints/qwen2-math-7b-orig-gsm8k-dpo-checkpoint-2-1pair", metadata={"help": "directory"})
        
        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/Qwen2.5-Math-7B-Instruct", metadata={"help": "the policy model name"})
        # ref_model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/Qwen2.5-Math-7B-Instruct", metadata={"help": "the reference model name"})
        # data_file: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/samples/qwen2-math-7b-dpo1-gsm8k-total-sample-0.json", metadata={"help": "the model name"})
        # num_pairs: Optional[int] = field(default=1, metadata={"help": "number of DPO-data pairs for each prompt"})
        # output_dir: Optional[str] = field(default="./checkpoints/qwen2-math-7b-orig-gsm8k-dpo-checkpoint-2-1pair", metadata={"help": "directory"})

        # model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/deepseek-math-7b-rl", metadata={"help": "the policy model name"})
        # ref_model_name: Optional[str] = field(default="/mnt/data/xiaoxuan/llms/deepseek-math-7b-rl", metadata={"help": "the reference model name"})
        # data_file: Optional[str] = field(default="/mnt/data/xiaoxuan/sft-code/samples/deepseek-math-7b-rl-math-total-sample-0.json", metadata={"help": "the model name"})
        # num_pairs: Optional[int] = field(default=15, metadata={"help": "number of DPO-data pairs for each prompt"})
        # output_dir: Optional[str] = field(default="./checkpoints/deepseek-math-7b-rl-orig-math-rpo-checkpoint-1-15pair", metadata={"help": "directory"})

        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        batch_size: Optional[int] = field(default=2, metadata={"help": "bz"})
        # learning_rate: Optional[float] = field(default=7e-7, metadata={"help": "learning rate"})
        learning_rate: Optional[float] = field(default=3e-7, metadata={"help": "learning rate"})
        lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "learning rate decay"})
        warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warm up"})
        weight_decay: Optional[float] = field(default=0.00, metadata={"help": "weight decay"})
        # beta: Optional[float] = field(default=0.1, metadata={"help": "beta"})
        beta: Optional[float] = field(default=0.2, metadata={"help": "beta"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "gradient accumulation steps"})
        gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "None"})
        gradient_checkpointing_use_reetrant: Optional[bool] = field(default=False, metadata={"help": "None"})
        save_strategy: Optional[str] = field(default="epoch", metadata={"help": "no save during train"})
        save_steps: Optional[int] = field(default=100, metadata={"help": "save steps"})
        report_to: Optional[str] = field(default="none", metadata={"help": "wandb, none"})
        num_train_epochs: Optional[float] = field(default=3, metadata={"help": "training epoches"})


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    

    main(args)