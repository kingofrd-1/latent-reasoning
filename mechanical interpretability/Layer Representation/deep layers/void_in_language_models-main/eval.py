import os
import json
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from lac import L2AdaptiveComputation
from utils import get_text
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


class EvalDataset(Dataset):

    def __init__(self, task):
        self.task = task

        if task == "MMLU":
            self.dataset = load_dataset("cais/mmlu", "all")["test"]

        elif task == "GPQA":
            self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]

        elif task == "BoolQ":
            self.dataset = load_dataset("google/boolq")["validation"]

        else:
            raise ValueError(f"Unsupported task: {task}. Please choose from ['MMLU', 'GPQA', 'BoolQ'].")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.task == "MMLU":
            question = data["question"]
            choices = [data["choices"][i] for i in range(4)]
            correct_answer_idx = data["answer"]
            correct_letter = ["A", "B", "C", "D"][correct_answer_idx]
            subject = data["subject"]

            prompt = format_prompt(question, choices)

            return {"prompt": prompt,
                    "question": question,
                    "answer": correct_letter,
                    "subject": subject}

        elif self.task == "GPQA":
            question = data["Question"]
            correct_answer = data["Correct Answer"]
            choices = [data[f"Incorrect Answer {i + 1}"] for i in range(3)]
            choices.append(correct_answer)

            prompt = format_prompt(question, choices)

            return {"prompt": prompt,
                    "question": question,
                    "answer": correct_answer}

        elif self.task == "BoolQ":
            question = data["question"]
            correct_answer = data["answer"]
            passage = data["passage"]

            prompt = format_boolq_prompt(question, passage)

            return {"prompt": prompt,
                    "question": question,
                    "answer": correct_answer}


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        batch_dict[key] = [item[key] for item in batch]
    return batch_dict


def format_prompt(question, choices):
    options = ["A", "B", "C", "D"]
    formatted_choices = "\n".join([f"{options[i]}. {choice}" for i, choice in enumerate(choices)])

    prompt = [
        {"role": "user", "content": f"""
    Question:
    {question}

    Options:
    {formatted_choices}

    Output format:
    [Option]

    """}]

    return prompt


def format_boolq_prompt(question, passage):
    prompt = [{"role": "user", "content": f"""
    Respond to the given question based on the passage with True or False.
    Passage:
    {passage}

    Options:
    {question}

    Output Format:
    [True/False]
    """}]
    return prompt


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_dataset = EvalDataset(args.task)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lac = L2AdaptiveComputation(
        model=model,
        tokenizer=tokenizer,
        skip_layers=args.skip_layers,
    )

    if args.task == "MMLU":
        dir_name = f"{args.model_id.split('/')[1]}_mmlu_eval"
        os.makedirs(dir_name, exist_ok=True)

        for i, batch in enumerate(tqdm(eval_dataloader, desc="Processing batches")):
            formatted_inputs = tokenizer.apply_chat_template(batch['prompt'],
                                                             tokenize=False,
                                                             add_generation_prompt=True)

            inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(device)

            prompt_processing_phase, response_generation_phase, prompt_l2, response_l2 = lac.generate(inputs,
                                                                                                      max_seq_length=args.seq_length,
                                                                                                      alpha=args.alpha,
                                                                                                      temperature=args.temp,
                                                                                                      top_p=args.top_p
                                                                                                      )

            model_answer = get_text(np.array(response_generation_phase).transpose(2, 1, 0, 3), tokenizer)

            output = {
                "question": batch['question'],
                "subject": batch['subject'],
                "prompt": batch['prompt'],
                "true_answer": batch['answer'],
                "model_answer": model_answer,
                "pp_phase": prompt_processing_phase,
                "rg_phase": response_generation_phase,
                "prompt_l2": prompt_l2,
                "response_l2": response_l2,
            }

            with open(f"{dir_name}/sample_{i}.json", "w") as f:
                json.dump(output, f, indent=2)
            f.close()

    elif args.task == "GPQA":
        dir_name = f"{args.model_id.split('/')[1]}_gpqa_eval"

        os.makedirs(dir_name, exist_ok=True)

        for i, batch in enumerate(tqdm(eval_dataloader, desc="Processing batches")):
            formatted_inputs = tokenizer.apply_chat_template(batch['prompt'],
                                                             tokenize=False,
                                                             add_generation_prompt=True)

            inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(device)

            prompt_processing_phase, response_generation_phase, prompt_l2, response_l2 = lac.generate(inputs,
                                                                                                      max_seq_length=args.seq_length,
                                                                                                      alpha=args.alpha,
                                                                                                      temperature=args.temp,
                                                                                                      top_p=args.top_p
                                                                                                      )

            model_answer = get_text(np.array(response_generation_phase).transpose(2, 1, 0, 3), tokenizer)

            output = {
                "question": batch['question'],
                "prompt": batch['prompt'],
                "true_answer": batch['answer'],
                "model_answer": model_answer,
                "pp_phase": prompt_processing_phase,
                "rg_phase": response_generation_phase,
                "prompt_l2": prompt_l2,
                "response_l2": response_l2,
            }

            with open(f"{dir_name}/sample_{i}.json", "w") as f:
                json.dump(output, f, indent=2)
            f.close()

    elif args.task == "BoolQ":
        dir_name = f"{args.model_id.split('/')[1]}_eval"

        os.makedirs(dir_name, exist_ok=True)

        for i, batch in enumerate(tqdm(eval_dataloader, desc="Processing batches")):
            formatted_inputs = tokenizer.apply_chat_template(batch['prompt'],
                                                             tokenize=False,
                                                             add_generation_prompt=True)

            inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(device)

            prompt_processing_phase, response_generation_phase, prompt_l2, response_l2 = lac.generate(inputs,
                                                                                                      max_seq_length=args.seq_length,
                                                                                                      alpha=args.alpha,
                                                                                                      temperature=args.temp,
                                                                                                      top_p=args.top_p
                                                                                                      )

            model_answer = get_text(np.array(response_generation_phase).transpose(2, 1, 0, 3), tokenizer)

            output = {
                "question": batch['question'],
                "prompt": batch['prompt'],
                "true_answer": batch['answer'],
                "model_answer": model_answer,
                "pp_phase": prompt_processing_phase,
                "rg_phase": response_generation_phase,
                "prompt_l2": prompt_l2,
                "response_l2": response_l2,
            }

            with open(f"{dir_name}/sample_{i}.json", "w") as f:
                json.dump(output, f, indent=2)
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate models on GPQA, MMLU, or BoolQ with L2 Adaptive Computation.")

    parser.add_argument("--model-id", type=str, required=True,
                        choices=[
                            'meta-llama/Meta-Llama-3-8B-Instruct',
                            'Qwen/Qwen2.5-7B-Instruct',
                            'mistralai/Mistral-7B-Instruct-v0.3'
                        ],
                        help="Hugging Face model ID to use for evaluation."
                        )

    parser.add_argument("--task", type=str, required=True, choices=['GPQA', 'MMLU', 'BoolQ'],
                        help="Task to evaluate LMs on (GPQA, MMLU, or BoolQ).")

    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for processing examples.")

    parser.add_argument("--seq-length", type=int, default=50,
                        help="Maximum number of new tokens to generate for each example.")

    parser.add_argument("--skip-layers", type=bool, default=True,
                        help="True for skipping layer and False for not to!")

    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Alpha parameter for L2 Adaptive Computation")

    parser.add_argument("--temp", type=float, default=0.6,
                        help="Temperature for controlling the randomness of generated text. Higher values increase randomness.")

    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (nucleus sampling).")

    args = parser.parse_args()

    evaluate(args)
