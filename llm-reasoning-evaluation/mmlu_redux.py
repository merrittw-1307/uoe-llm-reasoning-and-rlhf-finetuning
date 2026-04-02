import random
import re
import json
from tqdm import tqdm
from utils import predict_gpt, model_evaluation

def process_mmlu_redux_questions(dataset, output_file_path, formulation_prompt, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    for index, example in tqdm(dataset.iterrows(), desc="Processing questions"):
        if example["error_type"] != "ok":
            continue

        question = example['question']
        options = eval(example['choices'])  
        correct_answer = int(example['answer'])

        print(f"Processing question: {question}")

        system_content = formulation_prompt

        formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([ABCD])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
            final_answer_numeric = ord(final_answer_letter) - ord('A')
        else:
            final_answer_numeric = -1  

        is_correct = (final_answer_numeric == correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "options": options,
            "model_result": model_result,
            "final_answer": final_answer_numeric,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy
