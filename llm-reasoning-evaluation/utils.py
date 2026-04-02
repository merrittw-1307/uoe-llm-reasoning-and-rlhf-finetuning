import io
import sys
import openai
import torch
from torch.nn import DataParallel
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()

def predict_gpt(openai, messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content

def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device=None):
    if model_type == "gpt":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_gpt(model, messages)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Model result: {model_result}")
    return model_result
