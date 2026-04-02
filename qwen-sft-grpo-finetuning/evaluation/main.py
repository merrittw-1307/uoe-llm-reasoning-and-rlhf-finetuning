import json
from multiprocessing import Value
import os
import re
import argparse
from tqdm import tqdm
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from peft import PeftModel
import random
import numpy as np

from gsm8k import load_gsm8k_questions, process_gsm8k_questions

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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Process datasets")
    parser.add_argument("--model_signature", default="Qwen/Qwen2.5-0.5B-Instruct", help="Choose the model")
    parser.add_argument("--sft_adapter_path", required=False, help="Path to the SFT adapter.")
    parser.add_argument("--grpo_adapter_path", required=False, help="Path to the saved GRPO adapter.")
    parser.add_argument("--output_path", default=None, required=True, help="Path to save the evaluation output.")
    args = parser.parse_args()

    if args.sft_adapter_path and args.grpo_adapter_path is None:
        training = "sft"
    elif args.sft_adapter_path and args.grpo_adapter_path:
        training = "sft+grpo"
    elif args.sft_adapter_path is None and args.grpo_adapter_path is None:
        training = "zero-shot"
    else:
        raise ValueError("Invalid combination of arguments. Please provide either both --sft_adapter_path and --grpo_adapter_path for sft+grpo, only --sft_adapter_path for sft, or neither for zero-shot.")

    OUTPUT_DIR = args.output_path
    output_file_path = f"{OUTPUT_DIR}/results.json"
    log_file_path = f"{OUTPUT_DIR}/results.txt"

    ensure_dir(output_file_path)

    with open(output_file_path, 'w') as f:
        json.dump([], f)

    with open(log_file_path, 'w') as f:
        f.write(f"Start evaluating the {training} experiment.\n")
        f.write(f"Args: {args}\n")

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if training == "sft+grpo":

        with open(log_file_path, 'w') as f:
            f.write(f"Evaluating model {args.model_signature} with SFT from {args.sft_adapter_path} and GRPO from {args.grpo_adapter_path}.\n")
        
        base_id = args.model_signature
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        
        # 1. Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        # 2. Load and Merge SFT Adapter (Exactly like training)
        print("Merging SFT adapter...")
        model = PeftModel.from_pretrained(base_model, args.sft_adapter_path)
        model = model.merge_and_unload()
        
        # 3. Load GRPO Adapter on top of the merged model
        print("Loading GRPO adapter...")
        model = PeftModel.from_pretrained(model, args.grpo_adapter_path)
        
        model_id = f"{base_id}+SFT+GRPO"
        model.eval()
    
    elif training == "sft":
        base_id = args.model_signature
        adapter_path = args.sft_adapter_path
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model_id = f"{base_id}+LoRA"
        model.eval()

    elif training == "zero-shot":
        model_id = args.model_signature
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        model.eval()
    else:
        raise ValueError("Please select either yes or no")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_from_disk("dataset/gsm8k_test_100")
    questions = load_gsm8k_questions(dataset)

    results, overall_accuracy, valid_accuracy, invalid_rate = process_gsm8k_questions(
        questions,
        output_file_path,
        None,          
        model_id,      
        model,
        tokenizer,
        device
    )

    end_time = time.time()
    duration = end_time - start_time

    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Overall Accuracy (including invalid): {overall_accuracy:.2%}")
    print(f"Valid Accuracy (excluding invalid): {valid_accuracy:.2%}")
    print(f"Invalid Rate: {invalid_rate:.2%}")
    print(f"Evaluation Duration: {duration:.2f} seconds")

    with open(log_file_path, 'a') as f:
        f.write(f"Overall Accuracy (including invalid): {overall_accuracy:.2%}\n")
        f.write(f"Valid Accuracy (excluding invalid): {valid_accuracy:.2%}\n")
        f.write(f"Invalid Rate: {invalid_rate:.2%}\n")
        f.write(f"Evaluation Duration: {duration:.2f} seconds\n")

    print(f"Log file updated: {log_file_path}")

if __name__ == "__main__":
    main()