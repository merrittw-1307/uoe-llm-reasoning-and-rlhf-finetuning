## File Descriptions

### `main.py`

This is the main script that sets up the entire process, including loading datasets, model initialisation, and evaluation.

To execute the main script (after installing the dependencies properly as in requirements.txt), use the following command:

```bash
cd partI
```

```bash
python main.py --dataset  mmlu-redux-college_mathematics  --method comat --model gpt
```

### `mmlu_redux.py`

This file contains functions for processing the MMLU-Redux dataset.

- **`process_mmlu_redux_questions`**: Processes questions from the MMLU-Redux dataset and evaluates them.

This directory also contains prompt files associated with the College Mathematics dataset:

- **`MMLU-Redux-college_mathematics_prompts/comat.txt`**: Prompt file for the CoMAT method.
- **`CoMAT_Instruction.py`**: Prompt file for the CoMAT method, used for evaluation in this assignment.


### `shapley_value_evaluation.py`

This script evaluates the Shapley value, a concept from cooperative game theory. The Shapley value is used to fairly distribute total gains among participants based on their individual contributions.

### `utils.py`

This file contains functions for using gpt to make predictions with different models and evaluate their outputs.
