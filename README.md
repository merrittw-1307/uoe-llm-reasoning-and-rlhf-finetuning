# LLM Reasoning Evaluation & Small Model Fine-Tuning
 
Two-part NLP coursework exploring LLM mathematical reasoning: (1) prompt engineering analysis via CoMAT and Shapley value attribution, (2) fine-tuning a 0.5B parameter model with SFT and GRPO reinforcement learning on GSM8K.
 
University of Edinburgh — INFR11287 Advanced Topics in NLP (2025–26)
 
---
 
## Overview
 
This project investigates two complementary approaches to improving mathematical reasoning in language models:
 
**Part 1** treats LLMs as black boxes and evaluates a structured prompt engineering technique (CoMAT) on the MMLU-Redux college mathematics dataset. Shapley value analysis is used to quantify the marginal contribution of each reasoning step.
 
**Part 2** goes inside the model — fine-tuning Qwen-2.5-0.5B using Supervised Fine-Tuning (SFT) and then applying Group Relative Policy Optimisation (GRPO) with verifiable rewards on the GSM8K grade-school mathematics dataset.
 
---
 
## Part 1: CoMAT Prompt Engineering & Shapley Value Analysis
 
### What is CoMAT?
 
Chain of Mathematically Annotated Thought (CoMAT) is a structured prompt engineering technique that decomposes mathematical word problems into four explicit symbolic reasoning steps before solving:
 
| Step | Description |
|------|-------------|
| **s1** Identification & Definition | Extract and name all variables and constants |
| **s2** Structural Logic Translation | Translate relationships into formal equations |
| **s3** Explicit Factual Representation | Instantiate all constants with their values |
| **s4** Question Formalisation | State the formal objective using prior steps |
 
After these four steps, the model performs Reasoning Execution and derives a final answer.
 
**Example**: For a ticket revenue problem (50,000 capacity, $250 VIP / $100 regular, $6.5M revenue):
- s1 defines `v` (VIP tickets), `r` (regular tickets)
- s2 translates to `v + r = 50000` and `250v + 100r = 6500000`
- s3 instantiates all constants
- s4 formalises: Find `v` satisfying both constraints
- Execution solves: `v = 10,000`
 
### Evaluation Setup
 
- **Dataset**: MMLU-Redux (college_mathematics subset)
- **Model**: GPT-4o-mini
- **Configuration**: temperature=0.0, max_tokens=2000
- **Framework**: OpenAI API via custom evaluation pipeline
 
### Shapley Value Analysis
 
To measure the marginal contribution of each CoMAT step, we compute Shapley values from cooperative game theory. Given 4 steps (players), the Shapley value for step `i` is:
 
```
φᵢ = (1/|Π|) Σ_π [v(Sᵢ ∪ {i}) - v(Sᵢ)]
```
 
Where `v(S)` = fraction of questions answered correctly using only steps in subset `S`, computed over all 4! = 24 permutations.
 
**Implementation**: `shapley_value_evaluation.py` enumerates all permutations, evaluates correctness from pre-computed `evaluation_with_steps.csv`, and calculates each step's marginal contribution across every ordering.
 
Key finding: step contribution is not monotonically ordered — some intermediate steps carry higher marginal value than earlier ones, reflecting how symbolic structure enables downstream reasoning.
 
### Temperature Analysis
 
- **Temperature 0.0**: Deterministic decoding; consistent results, well-suited for exact-answer mathematical tasks
- **Temperature 0.7**: Sampling variance introduced; accuracy typically lower for constrained problems where stochasticity hurts precision without improving exploration
 
### File Structure
 
```
llm-reasoning-evaluation/
├── main.py                        # Evaluation entry point
├── mmlu_redux.py                  # MMLU-Redux dataset processing
├── CoMAT_Instruction.py           # CoMAT prompt template
├── shapley_value_evaluation.py    # Shapley value computation
├── utils.py                       # GPT prediction utilities
└── requirements.txt
```
 
### Running Part 1
 
```bash
cd llm-reasoning-evaluation
pip install -r requirements.txt
 
# Evaluate with CoMAT
python main.py --dataset mmlu-redux-college_mathematics --method comat --model gpt
```
 
---
 
## Part 2: SFT + GRPO Fine-Tuning on GSM8K
 
### Setup
 
- **Base models**: Qwen-2.5-0.5B and Qwen-2.5-0.5B-Instruct
- **Dataset**: GSM8K (3,000 training samples, 10% validation, 100 test)
- **Hardware**: Single T4 GPU (16GB VRAM) via Google Colab
- **Experiment tracking**: Weights & Biases
 
### Stage 1: Supervised Fine-Tuning (SFT)
 
SFT teaches the model a structured Chain-of-Thought output format. Each training sample uses a strict template with special tokens:
 
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{step-by-step reasoning}
The answer is {final_answer}<|im_end|>
```
 
**SFT Results**:
 
| Model | Zero-Shot Accuracy | Post-SFT Accuracy |
|-------|-------------------|-------------------|
| Qwen-2.5-0.5B (Base) | Low | Improved |
| Qwen-2.5-0.5B-Instruct | Moderate | Higher |
 
The Instruct model consistently outperforms the Base model after SFT despite identical architecture and training data. The Instruct model was pre-aligned with RLHF and instruction-following datasets — it has already learned structured output formats, making SFT adaptation more efficient. The Base model requires substantially more signal to converge on the expected format.
 
**Bug identified in original codebase**: `finetuning/main.py` contained a data leakage bug where the validation set was not actually held out — identical data appeared in both train and validation splits, producing overly optimistic validation loss curves that did not reflect true generalisation. Fixed prior to all experiments.
 
### Stage 2: GRPO (Group Relative Policy Optimisation)
 
GRPO is a reinforcement learning method that eliminates the need for a separate critic or reward model. It generates multiple candidate completions per prompt and trains the policy by comparing their relative quality.
 
**Why GRPO for maths?** Mathematical problems have verifiable answers — reward can be computed programmatically without human annotation. This makes GRPO particularly well-suited: reward signal is exact, scalable, and free.
 
**Reward functions implemented**:
 
```python
def format_reward_func(completions, **kwargs):
    """
    Reward: 0.5 if completion contains "the answer is"
    Enforces the structured output format learned during SFT
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"].lower()
        reward = 0.5 if "the answer is" in content else 0.0
        rewards.append(reward)
    return rewards
 
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward: 2.0 if numeric answer matches ground truth
    Uses regex to extract final number from completion
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        content = completion[0]["content"]
        match = re.search(
            r"the answer is\s*[\$]?([\d,]+\.?\d*)", content, re.IGNORECASE
        )
        predicted = match.group(1).replace(",", "") if match else None
        reward = 2.0 if predicted and predicted == str(gt) else 0.0
        rewards.append(reward)
    return rewards
```
 
**Reward weighting rationale**: Format reward (0.5) is intentionally lower than correctness reward (2.0). Correct answers must be worth more than superficial format compliance.
 
### GRPO Training Observations
 
**Format reward rises rapidly; correctness reward oscillates.** This is a known RL failure mode called **reward hacking** (specification gaming). The model discovers that always outputting "the answer is [something]" reliably earns 0.5 reward regardless of whether the answer is correct. The format signal is dense (easy to trigger) while the correctness signal is sparse (requires genuinely solving the problem).
 
For a 0.5B parameter model with limited capacity, learning correct mathematical reasoning while simultaneously satisfying format constraints proves difficult — the model preferentially optimises the easier objective.
 
**Mitigation approaches**:
- Delayed format reward (apply only after baseline correctness is established)
- Higher correctness reward weighting
- Increasing number of GRPO candidate generations per prompt
 
### Running Part 2
 
```bash
cd qwen-sft-grpo-finetuning
 
pip install -r requirements.txt
nvidia-smi  # Verify T4 GPU
 
# Zero-shot evaluation
python evaluation/main.py \
    --model_signature Qwen/Qwen2.5-0.5B-Instruct \
    --output_path ./outputs/zero-shot
 
# SFT training
python finetuning/main.py \
    --model_signature Qwen/Qwen2.5-0.5B-Instruct \
    --output_path ./checkpoints/sft \
    --wandb_token <YOUR_WANDB_API_KEY>
 
# GRPO (initialised from SFT checkpoint)
python grpo/main.py \
    --model_signature Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_path ./checkpoints/sft \
    --output_path ./checkpoints/sft_grpo \
    --wandb_token <YOUR_WANDB_API_KEY>
```
 
### File Structure
 
```
qwen-sft-grpo-finetuning/
├── finetuning/
│   ├── main.py              # SFT training script
│   ├── hyperparameter.py    # SFT hyperparameter config
│   └── prompt.py            # Chain-of-Thought prompt template
├── grpo/
│   ├── main.py              # GRPO training script
│   └── dataset.py           # GRPO dataset class
├── evaluation/
│   ├── main.py              # Evaluation script
│   ├── gsm8k.py             # GSM8K answer parsing
│   └── utils.py             # Generation utilities
├── dataset/
│   ├── gsm8k_3k_sft/        # SFT training data (3,000 samples)
│   ├── gsm8k_500_grpo/      # GRPO training data (500 samples)
│   └── gsm8k_test_100/      # Test set (100 samples)
└── requirements.txt
```
 
---
 
## Tech Stack
 
| Component | Technology |
|-----------|-----------|
| LLM API (Part 1) | OpenAI GPT-4o-mini |
| Base models (Part 2) | Qwen-2.5-0.5B, Qwen-2.5-0.5B-Instruct |
| Fine-tuning framework | HuggingFace Transformers + TRL |
| RL algorithm | GRPO (Group Relative Policy Optimisation) |
| Experiment tracking | Weights & Biases |
| Compute | Google Colab T4 GPU (16GB VRAM) |
| Datasets | MMLU-Redux (Part 1), GSM8K (Part 2) |
| Language | Python |
 
---
 
## Key Concepts Demonstrated
 
- **Structured prompt engineering**: CoMAT four-step symbolic decomposition vs. zero-shot baseline
- **Game-theoretic attribution**: Shapley value computation to quantify per-step contribution to model accuracy across all 24 step permutations
- **Supervised Fine-Tuning**: Format-guided SFT with special token templates for Chain-of-Thought reasoning
- **RLHF-free reinforcement learning**: GRPO with programmatic verifiable rewards — no reward model required
- **Reward hacking diagnosis**: Identifying format reward exploitation in small LMs and analysing the dense vs. sparse reward signal dynamic
- **Bug identification**: Detecting and fixing a data leakage bug in train/validation splitting prior to experiments
 
---
 
## Limitations
 
- **Model scale**: 0.5B parameters is significantly capacity-constrained for complex multi-step reasoning. Literature shows GRPO gains scale substantially with model size (7B+).
- **Reward sparsity**: Binary correctness reward provides minimal signal early in GRPO training, slowing convergence on hard problems.
- **Dataset size**: 3,000 SFT samples is small; the full GSM8K training set (7,473 samples) would likely yield substantial improvement.
- **Compute constraints**: T4 GPU limits batch size and number of GRPO candidate generations per prompt — both affect training stability and sample diversity.
