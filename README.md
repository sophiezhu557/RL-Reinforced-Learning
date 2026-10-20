# Reinforcement Learning for LLMs

This project explores Reinforcement Learning (RL) for fine-tuning large language models, focusing on reducing toxicity in generated dialogue summaries using **Group Relative Policy Optimization (GRPO)**.

## Project Overview

- **Task:** Dialogue summarization with toxicity reduction.
- **Base Model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with a pretrained PEFT adapter.
- **Dataset:** [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum).
- **RL Algorithm:** GRPO.
- **Reward Model:** [RoBERTa Hate Speech Classifier](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target).

## Project Structure

- `RL.py` — GRPO training, custom reward function, and toxicity evaluation.
- `requirements.txt` — Python dependencies.

## Workflow

1. Load and preprocess the DialogSum dataset.
2. Load the Qwen model with a pretrained PEFT adapter.
3. Define a custom reward function using the non-hate classification logits.
4. Fine-tune the model using GRPO.
5. Evaluate and compare toxicity scores before and after RL
