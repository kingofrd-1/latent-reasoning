import itertools
from typing import Optional, Callable
import itertools
import json
import numpy as np


class NODE:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self,
                 question=None,
                 model_answer=None,
                 guide_embedding=None,
                 standard_answer=None,
                 parent=None,
                 cot_answer=None,
                 clean_answer=None,
                 is_true=False,
                 reward: float = 0,
                 is_terminal: bool = False,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 depth=None,
                 children=None,
                 cum_rewards=None,
                 eva_process=None,
                 is_verified_true=False,
                 is_verified=False,
                 consistency=0,
                 prob_score=0,
                 x_train=None
                 ):
        self.id = next(NODE.id_iter)
        self.question = question
        self.model_answer = model_answer
        self.guide_embedding = guide_embedding
        self.standard_answer = standard_answer
        self.cot_answer = cot_answer
        self.clean_answer = clean_answer
        self.is_true = is_true
        self.cum_rewards = cum_rewards or []
        self.is_terminal = is_terminal
        self.reward = reward
        self.eva_process = eva_process or []
        self.consistency = consistency
        self.prob_score = prob_score
        self.x_train = x_train

        self.parent = parent
        self.children = children if children is not None else []
        self.calc_q = calc_q

        self.is_verified_true = is_verified_true
        self.is_verified = is_verified

        if depth is None:
            if parent is None:
                self.depth = 0
            else:
                self.depth = parent.depth + 1
        else:
            self.depth = depth

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:

        return self.calc_q(self.cum_rewards)


    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "depth": self.depth,
            # "guide_embedding": [i.tolist() for i in self.guide_embedding] if self.guide_embedding is not None else None,
            "guide_embedding": "[i.tolist() for i in self.guide_embedding] if self.guide_embedding is not None else None",
            "standard_answer": self.standard_answer,
            "model_answer": self.model_answer,
            "cot_answer": self.cot_answer,
            "clean_answer": self.clean_answer,
            "is_true": self.is_true,
            "reward": self.reward,
            "cum_rewards": self.cum_rewards,
            "is_terminal": self.is_terminal,
            "eva_process": self.eva_process,
            "is_verified_true": self.is_verified_true,
            "consistency": self.consistency,
            "prob_score": self.prob_score,
            "is_verified": self.is_verified,
            "children": [child.to_dict() for child in self.children],
        }



def set_model_args(args):
    # 设置模型相关参数
    if "Qwen" in args.model_name:
        args.replace_name = "qwen"
        args.special_token = "@"
        args.special_token_id = 569
    elif "mistralai" in args.model_name:
        args.replace_name = "mistral"
        args.special_token = "[control_760]"
        args.special_token_id = 762
    elif "llama" in args.model_name:
        args.replace_name = "llama"
        args.special_token = "<|reserved_special_token_20|>"
        args.special_token_id = 128025

    # 根据消融实验配置 ablation_type 设置 reward 信息
    if getattr(args, "ablation_type", 0) == 1:
        args.for_verifier = 0
        args.for_coherence = 1
    elif getattr(args, "ablation_type", 0) == 2:
        args.for_verifier = 1
        args.for_coherence = 0
