import re
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
import wandb

from prompt import sft_formatting_prompts_func
from hyperparameter import get_training_arguments
import random
import numpy as np
import os
import torch
import argparse

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():

    set_seed(42)
    parser = argparse.ArgumentParser(description="Process datasets")
    parser.add_argument("--model_signature", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Huggingface model signature for training.")
    parser.add_argument("--output_path", default=None, help="Path to save the adapter.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Set for learning rate experiments.")

    parser.add_argument("--wandb_project", default="nlu-gsm8k", help="WandB project name")
    parser.add_argument("--wandb_token", type=str, default=None, help="WandB API Key for login")

    args = parser.parse_args()

    OUTPUT_DIR = args.output_path
    RUN_NAME = OUTPUT_DIR.split('/')[-1]

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    if args.wandb_token:
        print(f"Logging into WandB with provided token {args.wandb_token}...")
        wandb.login(key=args.wandb_token)

    DATASET_NAME = "openai/gsm8k"
    DATASET_CONFIG = "main"
    MAX_TRAIN_INSTANCES = 3000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = args.model_signature
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    ds = load_from_disk("dataset/gsm8k_3k_sft")
    train_test = ds.train_test_split(test_size=0.1, seed=42)
    train_dataset_raw = train_test["train"]
    eval_dataset_raw = train_test["test"]

    if len(train_dataset_raw) > MAX_TRAIN_INSTANCES:
        train_dataset_raw = train_dataset_raw.select(range(MAX_TRAIN_INSTANCES))

    def map_to_text(example):
        return sft_formatting_prompts_func(tokenizer, example)

    train_dataset = train_dataset_raw.map(
        map_to_text, remove_columns=train_dataset_raw.column_names
    )
    eval_dataset = eval_dataset_raw.map(
        map_to_text, remove_columns=eval_dataset_raw.column_names
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_arguments = get_training_arguments(output_model=OUTPUT_DIR, learning_rate=args.learning_rate)

    training_arguments.report_to = ["wandb"]
    training_arguments.run_name = RUN_NAME

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)

    wandb.finish()

if __name__ == "__main__":
    main()