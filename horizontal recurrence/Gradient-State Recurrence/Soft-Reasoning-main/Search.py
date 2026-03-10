import math
from typing import Optional, Callable
import itertools
import json
import numpy as np
from tqdm import trange
import torch
from utils.process_answer import get_cot_answer, get_clean_answer, evaluate_answer, get_number
from collections import Counter
from utils.guassian_inference import bayessian_optimisation_torch
from vllm_run import generate_with_vLLM_model, generate_with_vLLM_model_usually
from utils.my_node import NODE

def save_tree_to_json(root, filename):
    tree_dict = root.to_dict()
    with open(filename, 'w') as file:
        json.dump(tree_dict, file, indent=4)

class Search:
    def __init__(self,
                 model,
                 tokenizer,
                 args,
                 user_prompt,
                 question,
                 answer,
                 verify_prompt,
                 output_trace_in_each_iter: bool = False,
                 num_repeats: int = 1,
                 agent=None,
                 w_exp: float = 0.5,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = np.mean,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy = 'max',
                 output_strategy: str = 'max_reward',
                 disable_tqdm: bool = True,
                 save_path: str = ''
                 ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.user_prompt = user_prompt
        self.question = question
        self.answer = answer
        self.verify_prompt = verify_prompt
        self.agent = agent
        self.search_config = None
        self.stop_layer_flag = False
        self.A = torch.normal(0, 1, size=(args.dimention, model.llm_engine.model_config.hf_config.hidden_size))
        self.x_train_list = []


        self.num_repeats = num_repeats
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        self.save_path = save_path
        self.final_result = None
        self.each_layer_max_reward = []
        self.reward_list = []
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy
        self._output_iter: list[NODE] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[NODE]] = None
        self.root: Optional[NODE] = None
        self.disable_tqdm = disable_tqdm
        self.all_max_reward = 0
        self.eval_record = []
        self.verifier_analysis = []

    def collect_x_train_and_rewards(self, root_node):
        reward_list = []
        x_train = torch.cat(self.x_train_list, dim=0)
        reward = torch.tensor(self.reward_list, dtype=torch.float32, device=x_train.device).view(-1, 1)

        with torch.no_grad():
            top_5_points = bayessian_optimisation_torch(x_train, reward, self.args.dimention, self.args.num_repeats, self.args.function_method, self.args.ucb_beta)
            X_new = torch.mm(top_5_points, self.A)
        top_5_points_list = [point.view(1, -1) for point in top_5_points]
        self.x_train_list += top_5_points_list
        X_new_list = [new_point.view(1, -1) for new_point in X_new]

        return X_new_list, top_5_points_list

    def get_random_embedding(self, next_token):
        args = self.args
        token_list = []
        for i in range(args.num_tokens):
            if args.initial_vector_method == 0:
                token_list.append(torch.randn(1, self.model.lm_head.in_features))
            elif args.initial_vector_method == 1:
                token_list.append(torch.normal(mean=0, std=2 ** 0.5, size=(1, self.model.lm_head.in_features)))
            elif args.initial_vector_method == 2:
                token_list.append(torch.normal(mean=1, std=0, size=(1, self.model.lm_head.in_features)))
            elif args.initial_vector_method == 3:
                embedding_layer = self.model.get_input_embeddings()
                token_list.append(embedding_layer(next_token).cpu() + torch.randn(1, self.model.lm_head.in_features))

        return token_list

    def get_random_embedding_gaussian(self):
        args = self.args
        token_list = []
        for i in range(args.num_tokens):
            X_train = torch.randn(1, args.dimention)
            self.x_train_list.append(X_train)
            X_start = torch.mm(X_train, self.A)
            token_list.append(X_start)
        return token_list, X_train


    def generate_evaluate_prompt(self, eva_node_list):
        thought_list = []
        user_prompt = self.verify_prompt
        user_prompt += "\n"
        user_prompt = user_prompt.replace("{question}", self.question)
        i=0
        if self.args.add_best_node:
            for analysis in self.verifier_analysis:
                analysis = analysis.replace("\n", " ").strip()
                user_prompt += f"{i}. Thought: {analysis}\n"
                thought_list.append(analysis)
                i += 1
        j = 0
        for response in eva_node_list:
            process_answer = response.cot_answer.replace("\n", " ").strip()
            user_prompt += f"{i+j}. Thought: {process_answer}\n"
            thought_list.append(process_answer)
            j += 1

        user_prompt += "\nAnalysis:\nLet’s think step by step. "
        return user_prompt, thought_list

    def verifier(self, node, node_list):
        filter_node_list = [node for node in node_list if node.clean_answer is not None]
        eva_prompt,thought_list = self.generate_evaluate_prompt( filter_node_list)
        print("_" * 40 + "verifier" + "_" * 40)
        output_except_prompt = generate_with_vLLM_model_usually(model=self.model, input=eva_prompt, max_tokens=self.args.max_new_tokens, temperature=0, n=1,stop = ["Question:\n",'Here are some examples:',"Final Answer:","Please let me"])[0]
        try:
            layer_answer = output_except_prompt.split('Here are some examples:')[0].split('Question')[0].split('Final Answer:')[0].split('Please let me')[0]
            layer_answer = layer_answer.split('Question:')[0].strip()
            layer_clean_answer = layer_answer.split('Answer:')[1].strip()
            if self.args.dataset != "strategyqa":
                layer_clean_answer = str(get_number(layer_clean_answer))
            else:
                if "true" in layer_clean_answer.lower():
                    layer_clean_answer = "true"
                elif "false" in layer_clean_answer.lower():
                    layer_clean_answer = "false"
            # print("_" * 40 + "verifier" + "_" * 40)
            # print(layer_answer)
            self.verifier_analysis.append(layer_answer)
            for remove_char in [',', '$', '%', 'g']:
                layer_clean_answer = layer_clean_answer.replace(remove_char, '')
        except:
            layer_clean_answer = "-9999"
        print(layer_clean_answer)
        result = evaluate_answer(self.args, layer_clean_answer, self.answer)
        result_dict ={
            "prompt": eva_prompt,
            "output": output_except_prompt,
            "layer_answer": layer_answer,
            "layer_clean_answer": layer_clean_answer,
            "result": result,
            "any_correct": any([node.is_true for node in node_list]),
            "thought_list": thought_list,
            "ground_truth": self.answer
        }
        node.eva_process.append(result_dict)
        self.eval_record.append(result_dict)
        for node in node_list:
            node.verify_answer = layer_answer
            if evaluate_answer(self.args, node.clean_answer, layer_clean_answer):
                node.is_verified_true = 1
            else:
                node.is_verified_true = 0

        return

    def visualize_tree_symbols(self,node, prefix=""):
        tree_str = ""
        if prefix == "":
            tree_str += "●\n"
        else:
            tree_str += prefix[:-3] + "└── ●\n"

        if node.children:
            for i, child in enumerate(node.children):
                if i < len(node.children) - 1:
                    tree_str += self.visualize_tree_symbols(child, prefix + "|   ")
                else:
                    tree_str += self.visualize_tree_symbols(child, prefix + "    ")
        return tree_str

    def _is_terminal_or_depth(self, node: NODE):
        return node.is_terminal or node.depth >= self.depth_limit

    def iterate(self, node: NODE) -> list[NODE]:
        # return a select path
        path = self._select(node)
        # not reach terminal
        if not self._is_terminal_or_depth(path[-1]) and not self.stop_flag:
            self._expand(path[-1])
            self._simulate(path)
            path_all = [path + [child] for child in path[-1].children]
            for _path in path_all:
                cum_reward = self._back_propagate(_path)
        return path

    def _select(self, node: NODE) -> list[NODE]:
        path = []
        while True:
            path.append(node)

            if node.children is None or len(node.children) == 0 or self._is_terminal_or_depth(node) :
                return path
            is_verified_true = max([node.is_verified_true for node in node.children])
            if is_verified_true == 0:
                return path
            node = self._sub_select(node)


    def _sub_select(self, node: NODE) -> NODE:
        best_child_node = None
        best_uct_value = float('-inf')
        for child in node.children:
            exploration_term = np.sqrt(np.log(len(node.cum_rewards)) / max(1, len(child.cum_rewards)))
            uct_value = child.Q + self.w_exp * exploration_term
            if uct_value > best_uct_value:
                best_uct_value = uct_value
                best_child_node = child

        return best_child_node

    def _expand_child(self, node):
        if node.parent is None:
            for _ in range(self.num_repeats):
                guide_embedding, x_train = self.get_random_embedding_gaussian()
                node.children.append(NODE(guide_embedding=guide_embedding, parent=node, x_train = x_train))
        else:
            guide_embeddings, x_trains = self.collect_x_train_and_rewards(self.root)
            for i in range(self.num_repeats):
                node.children.append(NODE(guide_embedding=[guide_embeddings[i]], parent=node, x_train = x_trains[i]))

        answer_list = []
        new_nodes = [child for child in node.children if child.model_answer is None]
        guide_embedding = [node.guide_embedding for node in new_nodes]
        output_except_prompt,prob_scores = generate_with_vLLM_model(model=self.model, input=self.user_prompt,
                                                        temperature=0, n=self.args.num_repeats,
                                                        stop=["Question:\n", 'Here are some examples:', "Final Answer:",
                                                              "Please let me"], max_tokens=self.args.max_new_tokens,insert_embedding=guide_embedding,model_name = self.args.replace_name,special_token_id=self.args.special_token_id)
        for repeat, child in enumerate(new_nodes):
            child.question = self.question
            child.model_answer = output_except_prompt[repeat]
            child.cot_answer = get_cot_answer(child.model_answer)
            child.clean_answer = get_clean_answer(self.args, child.cot_answer)
            child.is_true = evaluate_answer(self.args, child.clean_answer, self.answer)
            child.standard_answer = self.answer
            child.prob_score = prob_scores[repeat]
            print("_" * 80 + "\n")
            print(child.cot_answer)
            answer_list.append(child.clean_answer)

        # calculate reward
        total = len(answer_list)
        probability_dict = {item: count / total for item, count in Counter(answer_list).items()}
        self.verifier(node, new_nodes)
        for child in new_nodes:
            child.reward +=  self.args.for_verifier * child.is_verified_true + self.args.for_coherence * child.prob_score
            self.reward_list.append(child.reward)
            child.consistency = probability_dict[child.clean_answer]
        # print([child.reward for child in new_nodes])

        return

    def _expand(self, node: NODE):
        self._expand_child(node)
        max_reward_node = max(node.children, key=lambda n: n.reward)
        if self.all_max_reward == 0:
            self.all_max_reward = max_reward_node.reward
        # self.all_max_reward = max_reward_node.reward
        self.each_layer_max_reward.append(max_reward_node.reward)
        if len(self.each_layer_max_reward) > 1:
            if self.args.use_abs == 0:
                if max_reward_node.reward - self.all_max_reward < self.args.stop_threshold:
                    self.stop_layer_flag = True
            else:
                if abs(self.each_layer_max_reward[-2] - self.each_layer_max_reward[-1]) < self.args.stop_threshold:
                    self.stop_layer_flag = True
            if max_reward_node.reward - self.all_max_reward > 0:
                self.all_max_reward = max_reward_node.reward

    def _simulate(self, path: list[NODE]):
        node = path[-1]
        while True:
            if node.depth + 1 < self.depth_limit and not self.stop_flag and not self.stop_layer_flag:
                rewards = [child.reward for child in node.children]
                chosen_index = self.simulate_choice(rewards)
                remaining_children = [child for i, child in enumerate(node.children) if i != chosen_index]
                for child_remain in remaining_children:
                    self._back_propagate(path+[child_remain])
                node = node.children[chosen_index]
                self._expand(node)
                path.append(node)
            else:
                print(self.visualize_tree_symbols(self.root))
                return

    def _back_propagate(self, path: list[NODE]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1]) # np.mean
            node.cum_rewards.append(cum_reward)
        return cum_reward

    def _dfs_max_reward(self, path: list[NODE]) -> tuple[float, list[NODE]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])
        return node_list

    def __call__(self):
        NODE.reset_id()
        self.stop_flag = False
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = NODE(calc_q=self.calc_q)
        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='Search iteration', leave=False):
            path = self.iterate(self.root)
        self.final_result = self.eval_record[-1]["result"] if self.final_result is None else self.final_result
        combined_results = {
            "tree_lists": self.root.to_dict(),
            "eva_processes": self.eval_record,
            "final_result": self.final_result,
        }
        with open(self.save_path, 'w', encoding="utf-8") as json_file:
            json.dump(combined_results, json_file, ensure_ascii=False)
        return