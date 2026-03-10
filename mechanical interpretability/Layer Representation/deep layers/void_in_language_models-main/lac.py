import torch
from typing import Any


class L2AdaptiveComputation:
    "Implementation of L2 Adaptive Computation"

    def __init__(self, model, tokenizer, skip_layers: bool = False, l2_axis: int = -1,
                 state_slice: Any = slice(0, 2),
                 layer_module_tmp: str = "model.layers.{}"):

        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.skip_layers = skip_layers
        self.alpha = None
        self.l2_axis = l2_axis
        self.state_slice = state_slice
        self.max_num_layers = model.config.num_hidden_layers

        self.previous_state = None
        self.previous_norm = None
        self.previous_delta_norm = None
        self.delta_norms = None
        self.update_shape = None
        self.activated_layers = None
        self.l2_norms = None

        layers_name = [layer_module_tmp.format(layer) for layer in range(self.max_num_layers)]

        for name, module in model.named_modules():
            if name in layers_name:
                module.register_forward_hook(self._get_delta_norms(int(name.split(".")[-1])))

    def _reset_states(self):
        self.previous_state = None
        self.previous_norm = None
        self.previous_delta_norm = None
        self.delta_norms = None
        self.update_shape = None
        self.activated_layers = None
        self.l2_norms = None

    def _get_delta_norms(self, layer_id):

        def hook(_, input, output):
            if self.previous_state is None:
                self.update_shape = input[0].shape[self.state_slice]

                self.delta_norms = torch.zeros((self.max_num_layers,) + self.update_shape, dtype=output[0].dtype).to(
                    self.model.device)

                self.l2_norms = torch.zeros((self.max_num_layers,) + self.update_shape, dtype=output[0].dtype).to(
                    self.model.device)

                self.activated_layers = torch.zeros((self.max_num_layers,) + self.update_shape,
                                                    dtype=output[0].dtype).to(self.model.device)
                self.previous_delta_norm = torch.zeros(self.update_shape).to(self.model.device)

                if self.skip_layers:
                    self.previous_state = input[0]

                self.previous_norm = torch.linalg.norm(input[0], axis=self.l2_axis).to(self.model.device)

            state_norm = torch.linalg.norm(output[0], axis=self.l2_axis).to(self.model.device)

            curren_delta_norm = state_norm - self.previous_norm

            self.delta_norms[layer_id] = curren_delta_norm

            self.l2_norms[layer_id] = state_norm

            if self.delta_norms[:layer_id].shape[0] == 0:
                max_delta, _ = torch.max(self.delta_norms[layer_id], axis=0)
                min_delta, _ = torch.min(self.delta_norms[layer_id], axis=0)
            else:
                max_delta, _ = torch.max(self.delta_norms[:layer_id], axis=0)
                min_delta, _ = torch.min(self.delta_norms[:layer_id], axis=0)

            threshold = (max_delta - min_delta) * self.alpha

            reshape_value = curren_delta_norm.shape + (output[0].ndim - len(curren_delta_norm.shape)) * (1,)

            new_halted = torch.less(curren_delta_norm, threshold).reshape(reshape_value)

            self.activated_layers[layer_id] += torch.logical_not(new_halted).to(output[0].dtype).squeeze(-1)

            if self.skip_layers:
                new_state = output[0] * torch.logical_not(new_halted) + self.previous_state * new_halted
                self.previous_state = new_state

            self.previous_norm = state_norm
            self.previous_delta_norm = curren_delta_norm

            if self.skip_layers and layer_id == self.max_num_layers - 1:
                return (new_state,)
            else:
                return output

        return hook

    def generate(self, input_ids, max_seq_length: int = 512,
                 alpha: float = 0.8,
                 temperature: float = 0.6,
                 top_p: float = 0.9):

        self.alpha = alpha
        past_key_values = None

        prompt_processing_phase = []
        response_generation_phase = []
        prompt_l2_norms = []
        response_l2_norms = []
        with torch.no_grad():
            outputs = self.model(
                **input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            prompt_processing_phase.append(
                torch.cat([input_ids.input_ids.unsqueeze(-1),
                           self.activated_layers.permute(1, 2, 0).to(torch.int32)],
                          dim=-1).cpu().numpy().tolist())

            prompt_l2_norms.append(self.l2_norms.to(torch.float32).cpu().numpy().tolist())

            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = sample_top_p(probs, top_p)

            self._reset_states()
            for _ in range(max_seq_length):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                past_key_values = outputs.past_key_values

                logits = outputs.logits[:, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = sample_top_p(probs, top_p)

                response_generation_phase.append(
                    torch.cat([next_token.unsqueeze(-1), self.activated_layers.permute(1, 2, 0).to(torch.int32)],
                              dim=-1).cpu().numpy().tolist())

                response_l2_norms.append(self.l2_norms.to(torch.float32).cpu().numpy().tolist())

                self._reset_states()

                if torch.all(next_token == self.tokenizer.eos_token_id):
                    break

            return prompt_processing_phase, response_generation_phase, prompt_l2_norms, response_l2_norms



def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1).to("cuda")
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1).to("cuda")
    next_token = torch.gather(probs_idx, -1, next_token).to("cuda")
    return next_token
