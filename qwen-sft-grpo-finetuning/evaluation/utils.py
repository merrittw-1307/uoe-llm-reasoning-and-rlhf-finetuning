import io
import sys
import torch
import transformers
from typing import Tuple, Optional, Any, List, Union

def model_evaluation(model, tokenizer, system_content, question, max_new_tokens):
    instruction = (
        "Think step by step before answering the question, and provide the final answer "
        "as 'the answer is [answer]' format."
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_ids = inputs.to(model.device)
    attention_mask = None

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    model_result = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return model_result
