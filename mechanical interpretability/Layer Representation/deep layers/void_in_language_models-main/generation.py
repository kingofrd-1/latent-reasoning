import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from lac import L2AdaptiveComputation
from utils import get_text, get_usage


def generate_response(prompt, seq_length: int = 512, alpha: float = 0.8,
                      temperature: float = 0.6,
                      top_p: float = 0.9):
    formatted_inputs = tokenizer.apply_chat_template(prompt,
                                                     tokenize=False,
                                                     add_generation_prompt=True)

    inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(
        device)

    prompt_processing_phase, response_generation_phase, prompt_l2, response_l2 = lac.generate(inputs,
                                                                                              max_seq_length=seq_length,
                                                                                              alpha=alpha,
                                                                                              temperature=temperature,
                                                                                              top_p=top_p)

    return prompt_processing_phase, response_generation_phase, prompt_l2, response_l2, inputs


if __name__ == '__main__':
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    MAX_LENGTH = 2048
    ALPHA = 0.8
    SKIP = True
    TEMPERATURE = 0.6
    TOP_P = 0.9

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lac = L2AdaptiveComputation(
        model=model,
        tokenizer=tokenizer,
        skip_layers=SKIP,
        l2_axis=-1
    )

    prompt = [[{"role": "user",
                "content": """
    For how many positive integer values of $k$ does $kx^2+10x+k=0$ have rational solutions?
    """}]]

    prompt_processing_phase, response_generation_phase, prompt_l2, response_l2, inputs = generate_response(prompt,
                                                                                            seq_length=MAX_LENGTH,
                                                                                            alpha=ALPHA,
                                                                                            temperature=TEMPERATURE,
                                                                                            top_p=TOP_P)

    response = np.array(response_generation_phase).transpose((2, 1, 0, 3))
    prompt = np.array(prompt_processing_phase)

    texts = get_text(response, tokenizer)

    pp_usages = get_usage(prompt)
    rg_usages = get_usage(response)

    for text, pp_usage, rg_usage in zip(texts, pp_usages, rg_usages):
        print(text, end='\n')

        print(f"Mean PP usage: {pp_usage.mean()} \n Mean RG usage: {rg_usage.mean()} \n")
