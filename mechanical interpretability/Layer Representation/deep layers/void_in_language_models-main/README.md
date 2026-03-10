# Void in Language Models

## [arXiv](https://arxiv.org/abs/2505.14467)
This repository is the implementation of [Void in Language Models](https://arxiv.org/abs/2505.14467)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Supported models

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`

## Evaluation

To evaluate the proposed LMs on benchmarks(as stated in the paper), while skipping voids, do:

    python eval.py --model-id [MODEL_ID] --task [TASK] --batch-size [BATCH_SIZE] --alpha [ALPHA] --skip-layers True

## Citation

```
@misc{shemiranifar2025voidlanguagemodels,
      title={Void in Language Models}, 
      author={Mani Shemiranifar},
      year={2025},
      eprint={2505.14467},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14467}, 
}
```
