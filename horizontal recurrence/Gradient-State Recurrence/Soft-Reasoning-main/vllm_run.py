from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
from utils.insert_embedding import *


def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    llm = LLM(
        model=model_ckpt,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        # disable_log_stats=False,
    )
    return tokenizer, llm


def generate_with_vLLM_model(
        model,
        input,
        temperature=0.8,
        top_p=1,
        top_k=50,
        # repetition_penalty=1.1,
        n=5,
        max_tokens=200,
        # logprobs=1,
        stop=[],
        insert_embedding=None,
        model_name="llama",
        special_token_id=128025

):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=1,
        logprobs=1,
        max_tokens=max_tokens,
        stop=stop,
    )
    if "72B" in model.llm_engine.model_config.hf_config.name_or_path:
        output = []
        for index in range(len(input)):
            with override_forward_with_over_zero(model, factory, [insert_embedding[index]], model_name=model_name,
                                                 special_token_id=special_token_id):
                output_index = model.generate([input[index]], sampling_params, use_tqdm=False)
                output.append(output_index[0])
    else:
        with override_forward_with_over_zero(model, factory, insert_embedding, model_name=model_name,
                                             special_token_id=special_token_id):
            output = model.generate(input, sampling_params, use_tqdm=False)

    io_output_list = []
    prob_list = []
    for i in range(5):
        io_output_list.append(output[i].outputs[0].text)
        logprob_list = [list(token.values())[0].logprob for token in output[i].outputs[0].logprobs]
        probabilities = [math.exp(logprob) for logprob in logprob_list]
        average_probability = sum(probabilities) / len(probabilities)
        prob_list.append(average_probability)
    return io_output_list, prob_list


def generate_with_vLLM_model_usually(
        model,
        input,
        temperature=0.8,
        top_p=1,
        top_k=50,
        n=1,
        max_tokens=200,
        # logprobs=1,
        stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=n,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    io_output_list = [o.text for o in output[0].outputs]
    return io_output_list


if __name__ == "__main__":
    model_ckpt = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=2, half_precision=False)
    input = "What is the meaning of life?"
    vllm_response = generate_with_vLLM_model(model, input)
    # breakpoint()
    io_output_list = [o.text for o in vllm_response[0].outputs]
    print(io_output_list)
