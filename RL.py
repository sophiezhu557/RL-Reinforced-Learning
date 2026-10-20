from datasets import DatasetDict, load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import time
from peft import PeftModel
import os
import torch
import rouge_score
import evaluate
import tqdm
import pandas as pd
import numpy as np
import json


def prepare_prompt(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": 
                    "Summarize the following conversation.\n"
                    + example["dialogue"]
                    + "\nSummary:"
                ,
            }
        ],
        # "completion": [
        #     {
        #         "role": "assistant",
        #         "content": example["summary"],
        #     }
        # ],
    }


class Toxicity:
    def __init__(self, model_name, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

        self.nothate_index = next(
            i for i, label in self.model.config.id2label.items()
            if label.lower() == "nothate"
        )

        self.hate_index = next(
            i for i, label in self.model.config.id2label.items()
            if label.lower() == "hate"
        )

    def reward(self, completions, **kwargs):
        texts = [completion[0]["content"] for completion in completions]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        nohate_reward = logits[:, self.nothate_index] 
        return nohate_reward.float().cpu().tolist()
    
    def evaluate(self, completions):
        inputs = self.tokenizer(
            completions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        toxicity_scores = probabilities[:, self.hate_index]
        return toxicity_scores.float().cpu().tolist()

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def grpo(model, 
         dataset, 
         tokenizer, 
         reward_funcs,
         output_path,
         step,
         lr,
         epoch,
         generations=2
         ):
    grpo_config = GRPOConfig(
        output_dir = output_path,
        max_steps = step,
        learning_rate = lr,
        num_train_epochs = epoch,
        num_generations = generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
    )
    trainer = GRPOTrainer(
        model = model,
        reward_funcs = reward_funcs,
        train_dataset = dataset["train"],
        processing_class = tokenizer,
        args = grpo_config
    )

    print("start_train")
    trainer.train()
    trainer.model.save_pretrained(f"{output_path}/saved_model")
    return trainer


def evaluate_toxicity(model, 
                      tokenizer, 
                      toxicity_evaluator, 
                      dataset, 
                      output_path):
    model.eval()
    device = next(model.parameters()).device

    summary_list = []
    # Get response from ppo and base model.
    for example in tqdm.tqdm(dataset["test"]):
        inputs = tokenizer.apply_chat_template(example["prompt"], 
                                                   tokenize=True, 
                                                   add_generation_prompt=True,
                                                   return_tensors="pt",
                                                   return_dict=True).to(device) #建议 return_dict=True，这样才能 model.generate(**inputs)
        with torch.inference_mode():
            output = model.generate(
                    **inputs, 
                )
        prompt_len = inputs["input_ids"].shape[1]
        # print(output)
        completion = output[0][prompt_len:]
        summary = tokenizer.decode(completion, skip_special_tokens=True)
        # print(summary)
        toxicity = toxicity_evaluator([summary])[0]

        summary_list.append({
            "Dialogue": example["prompt"][0]["content"],
            "Summary": summary,
            "Toxicity": toxicity,
        })

    summary_list = pd.DataFrame(summary_list)
    return summary_list


if __name__ == "__main__":

    dataset = load_dataset("knkarthick/dialogsum")
    dataset = DatasetDict({
        "train":dataset["train"].shuffle(seed=42).select(range(125)),
        "validation": dataset["validation"].shuffle(seed=42).select(range(25)),
        # "validation": dataset["validation"].shuffle(seed=42).select(range(25)),  
        "test": dataset["test"].filter(lambda example, index: index % 100 == 0,with_indices=True)
    })
    datasets = dataset.map(prepare_prompt)
    datasets = datasets.select_columns(["prompt"])
    print(datasets)

    # reward model
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    toxicity = Toxicity(toxicity_model_name)
        
    # load tokenizer and model
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct" #'google/flan-t5-base'
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model_tokenizer.padding_side = "left"
    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    adaptor_path = "/Users/xiaoningzhu/Documents/AI/course_2/project/peft-dialogue-summary-checkpoint-local-qwen2.5"
    model = PeftModel.from_pretrained(base_model,
                                      adaptor_path,
                                      is_trainable=True)

    print(print_number_of_trainable_model_parameters(model))

    output_path = f"./grpo_output/qwen-rl-{str(int(time.time()))}"
    os.makedirs(output_path, exist_ok=True)
    
    metrics_before = evaluate_toxicity(
        model=model,
        tokenizer=model_tokenizer,
        toxicity_evaluator=toxicity.evaluate,
        dataset=datasets,
        output_path=output_path
    )

    steps = -1
    epoch = 1
    lr = 1e-5

    trainer = grpo(
        model=model,
        dataset=datasets,
        tokenizer=model_tokenizer,
        reward_funcs=toxicity.reward,
        output_path=output_path,
        step=-1,
        epoch=epoch,
        lr=lr,
        generations=2,
    )

    metrics_after = evaluate_toxicity(
        model=trainer.model,
        tokenizer=model_tokenizer,
        toxicity_evaluator=toxicity.evaluate,
        dataset=datasets,
        output_path=output_path
    )

    results = pd.merge(metrics_before, metrics_after, on=["Dialogue"], suffixes = ("_before", "_after"))
    results["toxicity_diff"] = (
        results["Toxicity_after"]
        - results["Toxicity_before"]
    )

    # print(results)

    completion_path = (f"{output_path}/completion_results.json")

    results.to_json(
        completion_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    run_info = {
        "base_model": base_model_name,
        "adapter_path": adaptor_path,
        "reward_model": toxicity_model_name,
        "steps": step,
        "learning_rate": lr,
        "num_generations": 2,
        "train_samples": len(datasets["train"]),
        "test_samples": len(datasets["test"]),
    }

    output = {
        "parameters": run_info,
        "results": results.to_dict(orient="records"),
    }

    with open(completion_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )
