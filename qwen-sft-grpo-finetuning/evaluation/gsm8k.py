import random
import re
import json
from tqdm import tqdm
from utils import model_evaluation
import string
from math_verify import parse, verify

def load_gsm8k_questions(dataset):
    questions = []
    for item in dataset:
        question = item['question']
        answer_match = re.search(r'####\s*(\d+)', item['answer'])
        if answer_match:
            answer = answer_match.group(1)
            questions.append({
                'question': question,
                'answer': answer
            })
    return questions

def process_gsm8k_questions(
    questions,
    output_file_path,
    formulation_prompt_path,
    model_type,
    model,
    tokenizer=None,
    device=None
):
    results = []
    total_correct = 0
    total_questions = 0
    valid_correct = 0
    valid_questions = 0

    for example in tqdm(questions, desc="Processing GSM8K questions"):
        question = example['question']
        correct_answer = example['answer']

        print(f"Processing question: {question}")

        model_result = model_evaluation(
            model,
            tokenizer,
            None,
            question,
            500
        )

        print(f"Model result: {model_result}")

        # Extract answer after "the answer is"
        final_answer_match = re.search(r"The answer is[:\s]*([^\.\n]+)", model_result, re.IGNORECASE)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
        else:
            fallback = re.findall(r"-?\d+\.?\d*", model_result)
            if fallback:
                final_answer = fallback[-1]
            else:
                final_answer = "Invalid"

        # Use math_verify to check correctness
        try:
            gold = parse(correct_answer)
            answer = parse(final_answer)
            is_correct = verify(gold, answer)
        except Exception as e:
            print(f"Error in math_verify: {e}")
            is_correct = False

        if is_correct:
            total_correct += 1
        total_questions += 1
        
        # Track valid answers separately
        if final_answer != "Invalid":
            valid_questions += 1
            if is_correct:
                valid_correct += 1

        result = {
            "question": question,
            "model_result": model_result,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }
        results.append(result)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {len(results)}")

    # Calculate all three accuracies
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    valid_accuracy = valid_correct / valid_questions if valid_questions > 0 else 0
    invalid_count = total_questions - valid_questions
    invalid_rate = invalid_count / total_questions if total_questions > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Total questions: {total_questions}")
    print(f"Valid Accuracy (excluding invalid): {valid_accuracy:.2%}")
    
    return results, overall_accuracy, valid_accuracy, invalid_rate