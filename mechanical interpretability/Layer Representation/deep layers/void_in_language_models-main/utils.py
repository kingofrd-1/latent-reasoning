import numpy as np
from transformers import AutoTokenizer


def get_text(tokens_layers: np.array, tokenizer: AutoTokenizer):
    return tokenizer.batch_decode(tokens_layers[0][..., 0])


def get_usage(token_layers: np.array):
    return token_layers[0][..., 1:].transpose(0, 2, 1).sum(axis=-1) / token_layers.shape[-2]
