import re

def build_rl_dataset(example):
    
    system_prompt = "Think step by step before answering the question, and provide the final answer as 'the answer is [answer]' format."

    truth = example['answer'].split("####")[1].strip()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['question']}
    ]
    
    return {
        "prompt": messages,
        "answer": truth
    }
    