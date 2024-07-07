# AurA for Toxicity Mitigation

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This software project accompanies the research paper, [Whispering Experts: Neural Interventions for Toxicity Mitigation in Language Models](https://openreview.net/forum?id=2P6GVfSrfZ).

## Abstract
An important issue with Large Language Models (LLMs) is their undesired ability to generate toxic language. In this work, we show that the neurons responsible for toxicity can be determined by their power to discriminate toxic sentences, and that toxic language can be mitigated by reducing their activation levels proportionally to this power. We propose AurA, an intervention that can be applied to any pre-trained LLM to mitigate toxicity. As the intervention is proportional to the ability of each neuron to discriminate toxic content, it is free of any model-dependent hyperparameters. We show that AurA can achieve up to $2.2\times$ reduction in toxicity with only a $0.72$ perplexity increase. We also show that AurA is effective with models of different scale (from 1.5B to 40B parameters), and its effectiveness in mitigating toxic language, while preserving common-sense zero-shot abilities, holds across all scales. AurA can be combined with pre-prompting strategies, boosting its average  mitigation potential from $1.28\times$ to $2.35\times$. Moreover, AurA can counteract adversarial pre-prompts that maliciously elicit toxic content, making it an effective method for deploying safer and less toxic models.

## Getting Started 

### 1. Clone this repository

```bash
git clone https://github.com/apple/ml-aura.git
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

Optionally install this repository

```bash
pip install -e .
```

### 3. Download the jigsaw dataset

You can find it in [Kaggle](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge?select=train.csv)

Let's say you save the dataset in `DATA_DIR`. Your filesystem should look the following way:

```bash
> ls $DATA_DIR/jigsaw

train.csv
test.csv
...
```

## Usage

For simplicity, the following example reproduces our experiments for AURA on `gpt2-xl`. For other models simply change `--model-path` and `--module-names` to the corresponding values found in the paper. Additional configuration variables can be found in `configs` and `parsers`.

Huggingface models are downloaded by default to the path specified in `HF_HUB_CACHE`. For more information visit the official Huggingface website.

### 1. Extract Responses

```bash
python -m scripts.compute_responses \
    --config-path configs/responses.yaml \
    --data-dir $DATA_DIR \
    --device cpu \
    --model-path openai-community/gpt2 \
    --module-names 'transformer.h.*.mlp.c_fc' 'transformer.h.*.mlp.c_proj' \
    --tag toxicity-responses \
    --verbose 1
```

The output will be written in the following folder structure:

```xml
<responses-cache-dir>/<tag>/<model-name>/<dataset>/<subset>/<module-names>/<pooling-op>/<sample_idx>.pt
```

By default `args.responses-cache-dir` is set to `/tmp/cache`.

### 2. Compute AURA intervention

Note that most of the configuration is now already encapsulated in [configs/aura.yaml](configs/aura.yaml).

```bash
python -m scripts.learn_aura \
--config-path configs/aura.yaml \
--module-names 'transformer.h.*.mlp.c_fc' 'transformer.h.*.mlp.c_proj'
```

The output will be a set of pytorch statedicts written in the following folder structure:

```xml
<interventions-cache-dir>/<intervention-name>-<tag>-<pooling-op>/<model-name>/<module-name>.statedict
```

By default `args.interventions-cache-dir` is set to `/tmp/cache/model-interventions`

### 3. Generate with intervened model

```bash
python -m scripts.generate_with_hooks \
--intervention-name aura \
--intervention-state-path /tmp/cache/model-interventions/aura-toxicity-max/gpt2 \
--model-path openai-community/gpt2 \
--device cpu \
--verbose 1 \
--module-names 'transformer.h.*.mlp.c_fc' 'transformer.h.*.mlp.c_proj'
```

## Test

We include pytest unit tests to verify the integrity of the code.

```bash
pytest .
```
## Citation
```bibtex
@inproceedings{
suau2024whispering,
title={Whispering Experts: Neural Interventions for Toxicity Mitigation in Language Models},
author={Xavier Suau and Pieter Delobelle and Katherine Metcalf and Armand Joulin and Nicholas Apostoloff and Luca Zappella and Pau Rodriguez},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=2P6GVfSrfZ}
}
```

## Contact

Xavier Suau Cuadros (`xsuaucuadros@apple.com`)
