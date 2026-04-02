import json
import os
import argparse
from tqdm import tqdm
import torch
from dotenv import load_dotenv
from mmlu_redux import process_mmlu_redux_questions
import openai
import pandas as pd
from CoMAT_Instruction import INSTRUCTION

load_dotenv()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    DATASET_CHOICES = [
        "mmlu-redux-college_mathematics"
    ]
    parser = argparse.ArgumentParser(description="Process MMLU questions")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True, help="Choose the dataset")
    parser.add_argument("--method", choices=["comat"], required=True, help="Choose the method")
    parser.add_argument("--model", choices=["gpt"], required=True, help="Choose the model")
    parser.add_argument("--dataconfig", choices=["normal"], default="normal", help="Choose the data configuration")
    args = parser.parse_args()

    output_dir = f"final_results/{args.dataset}/{args.method}/{args.model}"
    output_file_path = f"{output_dir}/{args.method}_{args.model}_{args.dataconfig}.json"
    log_file_path = f"{output_dir}/{args.method}_{args.model}_{args.dataconfig}_log.txt"

    ensure_dir(output_file_path)

    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    with open(log_file_path, 'w') as f:
        f.write(f"Start evaluating the {args.dataset} dataset with {args.method} method using {args.model} model and {args.dataconfig} configuration\n")
    print(f"Created log file: {log_file_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    tokenizer = None

    if args.model == "gpt":
        model = openai
    else: 
        raise ValueError("Please use the correct model")
    
    csv_path = "mmlu-redux-college_mathematics_dataset.csv"
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")
    dataset = pd.read_csv(csv_path)

    if args.dataconfig == "normal":
        results, accuracy = process_mmlu_redux_questions(
            dataset=dataset,
            output_file_path=output_file_path,
            formulation_prompt=INSTRUCTION, 
            model_type=args.model,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
    else:
        raise ValueError("Please select a valid data configuration")

    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

    with open(log_file_path, 'a') as f:
        f.write(f"Final Accuracy: {accuracy:.2%}\n")

    print(f"Log file updated: {log_file_path}")


if __name__ == "__main__":
    main()
