import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import DatasetDict, load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import time
from peft import PeftModel
import torch
import rouge_score
import evaluate
import tqdm
import pandas as pd
import numpy as np
import json
from rouge_score import rouge_scorer
import gc
from itertools import product


steps = -1
lr_list = [5e-6]
epoch_list = [1,2]
generations = 4
rouge_weight = 2.0


def load_model(base_model_name, adaptor_path):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )

    model = PeftModel.from_pretrained(
        base_model,
        adaptor_path,
        is_trainable=True
    )

    return model


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
    def __init__(self, model_name, rouge_weight, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.rouge_weight = rouge_weight

        self.nothate_index = next(
            i for i, label in self.model.config.id2label.items()
            if label.lower() == "nothate"
        )

        self.hate_index = next(
            i for i, label in self.model.config.id2label.items()
            if label.lower() == "hate"
        )

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        )

    def _get_logits(self, completions):
        inputs = self.tokenizer(
            completions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            return self.model(**inputs).logits
    
    def calculate_margin(self, completions):
        logits = self._get_logits(completions)

        margin = (
            logits[:, self.nothate_index]
            - logits[:, self.hate_index]
        )
        return margin.float().cpu().tolist()

    def reward(self, completions, summary, **kwargs):
        texts = [completion[0]["content"] for completion in completions]

        nohate_rewards = self.calculate_margin(texts)

        rewards = []
        for text, reference, nohate_reward in zip(
            texts, summary, nohate_rewards
        ):
            rougeL = self.rouge_scorer.score(
                reference, text
            )["rougeL"].fmeasure

            total_reward = (
                nohate_reward
                + self.rouge_weight * rougeL
            )
            rewards.append(float(total_reward))
        return rewards

    def evaluate(self, completions):
        logits = self._get_logits(completions)
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
         generations
         ):
    grpo_config = GRPOConfig(
        output_dir = output_path,
        remove_unused_columns=False,
        seed=42,
        data_seed=42,  

        # training hyperparameters
        max_steps = step,
        num_train_epochs = epoch,

        # optimization hyperparameters
        learning_rate = lr,

        # memory
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,

        # generate
        num_generations = generations,
        max_completion_length = 120

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


def evaluate_toxicity(
        model,
        tokenizer,
        toxicity_model,
        dataset,
    ):
    model.eval()
    device = next(model.parameters()).device

    score = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []

    # Get response from ppo and base model.
    for i, example in enumerate(tqdm.tqdm(dataset["test"])):
        inputs = tokenizer.apply_chat_template(example["prompt"], 
                                                   tokenize=True, 
                                                   add_generation_prompt=True,
                                                   return_tensors="pt",
                                                   return_dict=True).to(device) #建议 return_dict=True，这样才能 model.generate(**inputs)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                    **inputs, 
                    # 做generation需要加这些限制条件
                    max_new_tokens=120,
                    do_sample=False
                )
            
        completion = output[0][prompt_len:]
        summary = tokenizer.decode(completion, skip_special_tokens=True)

        toxicity = toxicity_model.evaluate([summary])[0]
        logit_margin = toxicity_model.calculate_margin([summary])[0]

        reference_summary = example["summary"]

        rouge = score.score(reference_summary,summary)
        
        reward = (
            logit_margin
            + toxicity_model.rouge_weight * rouge["rougeL"].fmeasure
        )

        results.append({
            "ID": i,
            "Dialogue": example["dialogue"],
            "Reference_summary": reference_summary,
            "Summary": summary,
            "Toxicity": toxicity,
            "Rouge1": rouge["rouge1"].fmeasure,
            "Rouge2": rouge["rouge2"].fmeasure,
            "RougeL": rouge["rougeL"].fmeasure,
            "Logit_margin": logit_margin,
            "Reward": reward
        })

    results = pd.DataFrame(results)

    return results


# Delete 
def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    dataset = load_dataset("knkarthick/dialogsum")
    dataset = DatasetDict({
        "train":dataset["train"].shuffle(seed=42).select(range(250)),
        # "validation": dataset["validation"].shuffle(seed=42).select(range(50)),
        # "validation": dataset["validation"].shuffle(seed=42).select(range(25)),  
        "test": dataset["test"].filter(lambda example, index: index % 50 == 0,with_indices=True)
    })
    datasets = dataset.map(prepare_prompt)
    datasets = datasets.select_columns(["prompt","dialogue","summary"])
    print(datasets)

    # load tokenizer and model
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
    model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model_tokenizer.padding_side = "left"

    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token

    adaptor_path = "/Users/xiaoningzhu/Documents/AI/course_2/project/runs/peft-dialogue-summary-checkpoint-local-qwen2.5"
    
    # =====================================================
    # baseline
    # =====================================================
    baselinemodel = load_model(
        base_model_name,
        adaptor_path
    )

    print(print_number_of_trainable_model_parameters(baselinemodel))

    # reward model
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    # toxicity = Toxicity(toxicity_model_name)
    # Added an extra ROUGE-L as reward. 
    # Reason: if not consider ROUGE, the model seems to have action like reward hacking, 
    # the toxicity becomes lower, but the ROUGE also drop significantly. 
    toxicity = Toxicity(toxicity_model_name, rouge_weight=rouge_weight)

    metrics_before = evaluate_toxicity(
        model=baselinemodel,
        tokenizer=model_tokenizer,
        toxicity_model=toxicity,
        dataset=datasets
    )

    metrics_before = metrics_before.rename(
        columns={
            "Summary": "Summary_before",
            "Toxicity": "Toxicity_before",
            "Rouge1": "Rouge1_before",
            "Rouge2": "Rouge2_before",
            "RougeL": "RougeL_before",
            "Logit_margin": "Logit_margin_before",
            "Reward": "Reward_before"
        }
    )

    del baselinemodel
    cleanup()


# =====================================================
# parameter list
# =====================================================

    for lr, epoch in product(lr_list, epoch_list):
        print(f"\nRunning: lr={lr}, epoch={epoch}")
        

        model = load_model(base_model_name, adaptor_path)

        output_path = f"./grpo_output/qwen-rl-{str(int(time.time()))}"
        os.makedirs(output_path, exist_ok=True)
            
        trainer = grpo(
            model=model,
            dataset=datasets,
            tokenizer=model_tokenizer,
            reward_funcs=toxicity.reward,
            output_path=output_path,
            step=steps,
            epoch=epoch,
            lr=lr,
            generations=generations,
        )

        metrics_after = evaluate_toxicity(
            model=trainer.model,
            tokenizer=model_tokenizer,
            toxicity_model=toxicity,
            dataset=datasets,
        )

        metrics_after = metrics_after.rename(
            columns={
                "Summary": "Summary_after",
                "Toxicity": "Toxicity_after",
                "Rouge1": "Rouge1_after",
                "Rouge2": "Rouge2_after",
                "RougeL": "RougeL_after",
                "Logit_margin": "Logit_margin_after",
                "Reward": "Reward_after",
            }
        )

        results = pd.merge(
            metrics_before,
            metrics_after[
                [
                    "ID",
                    "Summary_after",
                    "Toxicity_after",
                    "Rouge1_after",
                    "Rouge2_after",
                    "RougeL_after",
                    "Logit_margin_after",
                    "Reward_after",
                ]
            ],
            on="ID"
        )

        results["toxicity_diff"] = (results["Toxicity_after"] - results["Toxicity_before"])

        completion_path = (f"{output_path}/completion_results.json")

        run_info = {
            "base_model": base_model_name,
            "adapter_path": adaptor_path,
            "reward_model": toxicity_model_name,
            "rouge_weight": toxicity.rouge_weight,
            "reward_formula": "logit_margin + rouge_weight * rougeL",
            "epoch": epoch,
            "learning_rate": lr,
            "num_generations": generations,
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

        del trainer
        del model
        cleanup()