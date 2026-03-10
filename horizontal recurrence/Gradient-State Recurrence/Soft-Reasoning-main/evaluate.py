import json
import os
import numpy as np
import re
import itertools
from collections import Counter, defaultdict, deque

from utils.my_node import NODE

def get_number(string):
    if re.match(r'.*\..*\.\d$', string) and string.endswith('.0'):
        string = string[:-2]
    # Updated regex pattern to handle negative numbers
    pattern = r'-?\d+/\d+|-?\d+\.\d+|-?\d+'
    numbers_in_string = re.findall(pattern, string)

    if len(numbers_in_string) > 0:
        num = numbers_in_string[0]
        if '/' in num:
            numerator, denominator = map(float, num.split('/'))
            number_in_string = numerator / denominator
        else:
            number_in_string = float(num)
        return number_in_string
    return None


def evaluate_answer(args, string, number_to_compare, tolerance=0.01):

    if string is None:
        return False

    if "gsm8k" in args.dataset or "svamp" in args.dataset or "gsm_hard" in args.dataset:
        if isinstance(number_to_compare, str):
            number_to_compare = number_to_compare.strip()
            number_to_compare = number_to_compare.replace(',', '')
            number_to_compare = get_number(number_to_compare)
    elif "strategyqa" in args.dataset:
        if string.lower() == number_to_compare.lower():
            return True
        else:
            return False
    numbers_in_string = get_number(string)
    if numbers_in_string is not None and number_to_compare is not None:
        if abs(numbers_in_string - number_to_compare) <= tolerance:
            return True
    return False

def level_order_traversal(root):
    """
    Perform level-order traversal of the MCTS tree and return nodes grouped by level.

    :param root: The root node of the MCTS tree.
    :return: A list where each element is a list of nodes at the corresponding level.
    """
    if not root:
        return []

    result = []
    queue = deque([(root, 0)])  # Queue stores tuples of (node, level)

    while queue:
        current_node, level = queue.popleft()

        # If the current level does not exist in the result, add it
        if len(result) <= level:
            result.append([])

        # Append the current node to the corresponding level
        result[level].append(current_node)

        # Enqueue children with the next level
        if current_node.children:
            for child in current_node.children:
                queue.append((child, level + 1))

    return result


def load_tree_from_json(filename):

    def dict_to_node(node_dict, parent=None):
        node = NODE(
            question=node_dict["question"],
            model_answer=node_dict["model_answer"],
            guide_embedding=node_dict["guide_embedding"],
            standard_answer=node_dict["standard_answer"],
            parent=parent,
            cot_answer=node_dict["cot_answer"],
            clean_answer=node_dict["clean_answer"],
            is_true=node_dict["is_true"],
            reward=node_dict["reward"],
            is_terminal=node_dict["is_terminal"],
            depth=node_dict["depth"],
            cum_rewards=node_dict["cum_rewards"],
            eva_process=node_dict["eva_process"],
            is_verified_true=node_dict["is_verified_true"],
            is_verified=node_dict["is_verified"],
            consistency=node_dict["consistency"],
            prob_score=node_dict["prob_score"]


        )
        if "children" in node_dict and node_dict["children"]:
            node.children = [dict_to_node(child, parent=node) for child in node_dict["children"]]
        return node

    with open(filename, 'r') as file:
        result_dict = json.load(file)
    tree_dict = result_dict["tree_lists"]
    tree = dict_to_node(tree_dict)
    eva_processes = result_dict["eva_processes"]
    final_result = result_dict["final_result"]

    return tree,eva_processes,final_result


def have_key(root):
    is_true = []
    clean_answer_list = []
    def collect_answer_clean_recursive(node, answer_clean_list):
        if node is None:
            return
        if node.model_answer is not None:
            is_true.append(node.is_true)
        if node.clean_answer is not None:
            answer_clean_list.append(node.clean_answer)
        if node.children is None:
            return
        for child in node.children:
            collect_answer_clean_recursive(child, answer_clean_list)
    answer_clean_list = []
    collect_answer_clean_recursive(root, answer_clean_list)
    if any(is_true):
        return True
    return False

def collect_node( root ):
    clean_answer_list = []
    def collect_answer_clean_recursive(node):
        if node is None:
            return
        if node.clean_answer is not None:
            clean_answer_list.append(node.clean_answer)
        if node.children is None:
            return
        for child in node.children:
            collect_answer_clean_recursive(child)
    collect_answer_clean_recursive(root)

    return  clean_answer_list


def majority_vote(dataset, answers, answer):
    answer_counts = Counter(answers)
    most_common_answer, _ = answer_counts.most_common(1)[0]
    result = evaluate_answer(dataset, most_common_answer, answer)

    return result

class Args:
    def __init__(self, dataset):
        self.dataset = dataset

def check_answer(directory):
    for ds in ["gsm8k", "gsm_hard", "strategyqa", "svamp"]:
        if ds in directory:
            args = Args(ds)
            break
    file_list = [os.path.join(directory,file) for file in os.listdir(directory) if 'config' not in file ]
    any_result = []
    v_a_vote = 0
    print(f"len(file_list): {len(file_list)}")
    for file in file_list:
        tree, eva_processes, result = load_tree_from_json(file)
        ground_truth =tree.children[0].standard_answer
        node_list = level_order_traversal(tree)[1:]
        all_node = []
        for node in node_list:
            all_node += node
        any_result.append(have_key(tree))
        verify_clean_answer = [process['layer_clean_answer'] for process in eva_processes if 'layer_clean_answer' in process]
        all_clean_answer = collect_node(tree)
        v_a_vote_answer = int(majority_vote(args, verify_clean_answer+all_clean_answer, ground_truth))
        v_a_vote += v_a_vote_answer

    print(f"Any result Accuracy: {np.mean(any_result):.3%}")
    print(f"Accuracy: {v_a_vote / len(file_list):.3%}")


if __name__ == "__main__":

    check_answer("/cephfs/volumes/hpc_data_prj/inf_rate/219955ce-7670-4b47-901d-256fd2bfe358/qinglin/llama_decoding/decoding_token_position/output/gsm8k/Mistral-7B-Instruct-v0.3/d1_w5_b0_e250_r5_i1_n1_c1_v3_d0.5_04_185402/")
