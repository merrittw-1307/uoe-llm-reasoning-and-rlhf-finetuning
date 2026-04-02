# ATNLP Assignment: Tiny Reasoning Models (SFT to GRPO)

This repository contains the codebase for training and evaluating a tiny language model (**Qwen-2.5-0.5B and Qwen-2.5-0.5B-Instruct**) on mathematical reasoning tasks (**GSM8K**). 

The pipeline consists of three stages:
1. **Zero-Shot Evaluation:** Establishing a baseline.
2. **Supervised Fine-Tuning (SFT):** Teaching the model the reasoning format.
3. **Reinforcement Learning (GRPO):** Optimizing the model using verifiable rewards without a critic model.

## 🛠️ Onboard to Google Colab

### 0. Mount File System to Your Own Google Drive

Unlike a cluster, Colab wipes all files when the runtime disconnects. We must use Google Drive to store your code, dataset, and saved models. We also need to ensure you have a Python environment set up with the necessary dependencies.

First of all, create a `.ipynb` file and we will work on the notebook for the rest of instruction. If you are familiar with other option (e.g., terminal), you are free to do that. Otherwise, from here, all code blocks are expected to run within the notebook file.

The first on-boarding step is to mount the file system to your own Google Drive. You may be asked to login to your own Google account in this step. After mounting, you should see `drive/MyDrive` appears in the file menu on the left hand side of your screen. 

```
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

Then, you need to uplaod the coursework's part2 main folder (i.e., the one containing `finetuning/`, `grpo/`, `evaluation/`, `dataset/`) to `/content/drive/MyDrive/` and rename it to `nlu_assignment`.

You should finally have a directory structure like this: 

```text
/content/drive/MyDrive/nlu_assignment/
├── dataset/
│   ├── gsm8k_3k_sft     # Dataset for SFT experiments
│   ├── gsm8k_500_grpo   # Dataset for GRPO experiments
│   └── gsm8k_test_100   # Dataset for evaluation
├── finetuning/
│   ├── main.py            # Script for SFT (Supervised Fine-Tuning)
│   ├── hyperparameter.py  # Script for setting SFT hyperparameters
│   └── prompt.py          # Script for loading the prompt template for SFT
├── grpo/
│   ├── main.py          # Script for GRPO (Group Relative Policy Optimization)
|   └── dataset.py       # Script for setting GRPO dataset class
├── evaluation/
│   ├── main.py          # Evaluation script for GSM8K
│   ├── gsm8k.py         # GSM8K specific parsing logic
│   └── utils.py         # Generation utilities
└── README.md (Optional)
```

Next, change the current working directory of the running program to the main folder for assignment part2.
```
import os
os.chdir('/content/drive/MyDrive/nlu_assignment')
```

### 1. Install Dependencies

After you correctly mount the file system, while setting the correct root directory. You should be able to install all denpendencies needed for this coursework. If you are not able to find the `requirements.txt`, check previous steps to make sure you finish them correctly.

```
!pip install -r requirements.txt
```

### 2. Verify GPU (Ensure you have a T4 GPU)

You will need to use GPU for the coursework, and please ensure you have conncted to a GPU for your notebook. If not, go to `Edit` -> `Notebook Setting -> Click T4 GPU`, and do the previous step again.

```
!nvidia-smi
```

### 3. Wandb login

We will haveily use Wandb to monitor model's training so please ensure you have an wandb account and already obtain the API key from `https://wandb.ai/authorize`.

```
import wandb
import os

# Configuration
MODEL_SIGNATURE = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["WANDB_PROJECT"] = "nlu-gsm8k-assignment"

# Login to WandB (You will be prompted for API key)
wandb.login()
```

## 🚀 Running the Pipeline

Examples of commands for running scripts in Colab's notebook.

### 1. Zero-shot evaluation

```
# Run the evaluation script using the '!' magic command in colab notebook
! python evaluation/main.py \
    --model_signature $MODEL_SIGNATURE \
    --output_path ./outputs/$MODEL_SIGNATURE-zero-shot
```

### 2. SFT

```
# Run the evaluation script using the '!' magic command in colab notebook
! python finetuning/main.py \
    --model_signature $MODEL_SIGNATURE \
    --output_path ./checkpoints/$MODEL_SIGNATURE-sft \
    --wandb_token <YOUR_WANDB_API_KEY>
```

### 3. GRPO

```
# Run the evaluation script using the '!' magic command in colab notebook
! python grpo/main.py \
    --model_signature $MODEL_SIGNATURE \
    --adapter_path ./checkpoints/$MODEL_SIGNATURE-sft \
    --output_path ./checkpoints/$MODEL_SIGNATURE-sft_grpo \
    --wandb_token <YOUR_WANDB_API_KEY>
```

## Final note

This assignment was tested with Google Colab with a single T4 GPU and the execution of the evaluation/SFT/GRPO took around 0-0.5 hour / 1-1.5 hour / 1-1.5 hour. If you see this is not the case for you under your setup, you might be doing something wrong.
