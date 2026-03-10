import os
import argparse
import random
from tqdm import tqdm
import json
import yaml
import torch
from datetime import datetime
import numpy as np
import time
from Search import Search
from vllm_run import load_vLLM_model, generate_with_vLLM_model
from utils.gpu_tools import GPUMonitor
from utils.my_node import set_model_args
from evaluate import check_answer

prompt_path = os.path.join(os.getcwd(), "prompts.yaml")
with open(prompt_path, 'r') as f:
    prompts_from_file = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

with open("./data/few_shot_example.json", "r", encoding="utf-8") as json_file:
    few_shot_example = json.load(json_file)
    json_file.close()

def save_config(args):
    os.makedirs(args.output_path, exist_ok=True)
    config_path = os.path.join(args.output_path, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)

    print(f"Configuration saved to {config_path}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_directory(root, dataset_name, model_name):
    # Get the current time for unique naming
    time_now = datetime.now().strftime("%d_%H%M%S")
    # Create a directory name that includes additional parameters
    dic_name = '{}'.format(time_now)
    # Join the directory name with the root, dataset, and model name
    path = os.path.join(root, dataset_name, model_name, dic_name)
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    # Print and return the generated path
    print(f"Created directory: {path}")
    return path

def save_tree_to_json(root, filename):
    tree_dict = root.to_dict()
    with open(filename, 'w') as file:
        json.dump(tree_dict, file, indent=4)

def insert_reserved_token(args, string):
    token = args.special_token
    result_list = []
    for _ in range(5):
        if len(string) <= 1:
            result_list.append(string + token)
        else:
            if args.middle_token:
                insert_position = random.randint(1, len(string) - 1)
            elif args.first_token:
                insert_position = 0
            elif args.last_token:
                insert_position = len(string)
            new_string = string[:insert_position] + token + string[insert_position:]
            result_list.append(new_string)
    return result_list


def main(args):
    tokenizer, model = load_vLLM_model(args.model_name, seed=args.seed, tensor_parallel_size=args.num_gpu)
    # gpu_monitor = GPUMonitor(model, interval=1.0)
    # gpu_monitor.start()

    # atexit.register(gpu_monitor.stop)
    # atexit.register(gpu_monitor.report)

    if args.json_file_path is None:
        raise ValueError("json_file_path must be specified")
    with open(args.json_file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    for index in tqdm(range(args.begin, args.end)):
        try:
            prompt_entry = data[index]
        except:
            continue
        output_file = os.path.join(args.output_path, f"{index}.json")
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping this sample.")
            continue

        print("\n")
        answer = prompt_entry.get("answer", "")
        question = prompt_entry.get("question", "")

        question_prompt = ""
        if args.shot == -1:
            question_prompt += f"{question}"
        elif args.shot == 0:
            question_prompt += prompts_from_file[args.dataset+"_zero_shot"] + '\n'
            question_prompt += f"Question:\n{question}\nThought:\nLet’s think step by step. "
        else:
            question_prompt += prompts_from_file[args.dataset+"_zero_shot"] + '\n'
            for i in range(args.shot):
                demo= few_shot_example[args.dataset][i]
                question_prompt += f"Question:\n{demo['Question']}\nThought:\n{demo['Thought']}\nAnswer:\n{demo['Answer']}\n\n"
            question_prompt += f"Question:\n{question}\nThought:\nLet’s think step by step. "

        question_prompt_list = insert_reserved_token(args, question_prompt)

        Search_agent = Search(
            model=model,
            tokenizer=tokenizer,
            args=args,
            user_prompt = question_prompt_list,
            question=question,
            answer=answer,
            verify_prompt=prompts_from_file[args.dataset.replace("_train", "")],
            num_repeats=args.num_repeats,
            depth_limit=args.depth_limit,
            n_iters=args.num_iteration,
            calc_q=np.mean,
            simulate_strategy='max',
            output_strategy='max_reward',
            disable_tqdm=True,
            save_path =  os.path.join(args.output_path,'{}.json'.format(index))
        )
        Search_agent()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation script using a pretrained model.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",help='Name or path of the model to use (e.g., meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, Qwen/Qwen2-7B-Instruct).')
    parser.add_argument('--dataset', type=str, default="gsm8k", help='Dataset name. Options: gsm8k, gsm_hard, strategyqa, svamp, aime_2024, aime_2025.')
    parser.add_argument('--max_new_tokens', type=int, default=300, help='The maximum numbers of tokens to generate')
    parser.add_argument('--seed', type=int, default=1, help='Seed value for reproducibility')
    parser.add_argument('--random_seed', type=int, default=0, help='Random choose a seed.')
    parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy decoding.')
    parser.add_argument('--use_cache', action='store_true', help='Use cache to speed up decoding.')
    parser.set_defaults(use_cache=True)
    parser.set_defaults(is_gaussian=True)
    parser.add_argument('--stop_threshold', type=float, default=0.01, help='stop_threshold')
    parser.add_argument('--dimention', type=int, default=50, help='projection dimention')
    parser.add_argument('--use_abs', type=int, default=0, help='Top-K sampling parameter.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling parameter.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-K sampling parameter.')

    parser.add_argument('--save_path', type=str, default=None, help='Path to save the generated text')
    parser.add_argument('--begin', type=int, default=0, help='Starting index of prompts to process.')
    parser.add_argument('--end', type=int, default=200, help='Ending index of prompts to process.')
    parser.add_argument('--continue_predict', type=int, default=0, help='Flag to continue from the last processed prompt.')
    parser.add_argument('--num_repeats', type=int, default=5, help='Numbw gnjer of times to repeat each prompt generation.')
    parser.add_argument('--num_iteration', type=int, default=1, help='Number of iterations for prompt generation.')
    parser.add_argument('--depth_limit', type=int, default=5, help='Number of iterations for prompt generation.')
    parser.add_argument('--add_best_node', type=int, default=1, help='Flag to include the best node in the output.')
    parser.add_argument('--num_tokens', type=int, default=1, help='Selection method for processing nodes.')
    parser.add_argument('--shot', type=int, default=0, help='Selection method for processing nodes.')
    parser.add_argument('--initial_vector_method', type=int, default= 3, help='Method to initialize vectors for decoding.')

    parser.add_argument('--output_base_path', type=str, default="./output",help='Base path for saving the output data.')
    parser.add_argument('--first_token', type=int, default=0,help='Special token placement at the first position (set to 1 to enable).')
    parser.add_argument('--middle_token', type=int, default=0,help='Special token placement at the middle position (set to 1 to enable).')
    parser.add_argument('--last_token', type=int, default=1,help='Special token placement at the last position (set to 1 to enable).')
    parser.add_argument('--ablation_type', type=int, default=0,help='Ablation experiment configuration (set different values for different setups).')
    parser.add_argument('--for_verifier', type=int, default=1,help='Whether to include verifier information in the reward (1: include, 0: exclude).')
    parser.add_argument('--for_coherence', type=int, default=1,help='Whether to include coherence information in the reward (1: include, 0: exclude).')
    parser.add_argument('--function_method', type=str, default="ei",help='Expected function selection for Bayesian optimization (e.g., "ei", "ucb").')
    parser.add_argument('--ucb_beta', type=float, default=2.0,help='Beta parameter for UCB if Upper Confidence Bound is used as acquisition function.')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use.')

    args = parser.parse_args()
    set_model_args(args)

    dataset2dir = {
        "gsm8k": "./data/gsm8k.json",
        "gsm_hard": "./data/gsm_hard.json",
        "strategyqa": "./data/strategyqa.json",
        "svamp": "./data/svamp.json",
        "aime_2024": "./data/aime_2024.json",
        "aime_2025": "./data/aime_2025.json",

    }
    if args.random_seed:
        timestamp = int(time.time() * 1000)
        random_seed = timestamp + random.randint(0, 1000)
        random_seed = random_seed % (2 ** 32)
        args.seed = random_seed
    args.json_file_path = dataset2dir[args.dataset]
    set_seed(args.seed)
    if args.continue_predict:
        args.output_path = args.output_base_path
    else:
        args.output_path = create_output_directory(args.output_base_path, args.dataset, args.model_name.split("/")[-1])
    save_config(args)
    main(args)
    check_answer(args.output_path)
