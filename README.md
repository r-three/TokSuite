<div align="center">
  <br>
  <h1>TokSuite</h1>
  <h4>A controlled suite for measuring the impact of tokenizer choice on language model behavior.</h4>
</div>

<img alt="toksuite_overview" src="figures/toksuite.png" />

<p></p>

<p align="center">
  <a href="https://github.com/r-three/TokSuite">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github"></a>
  <a href="https://huggingface.co/collections/toksuite/toksuite-model-collection">
    <img alt="HuggingFace Models" src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://huggingface.co/collections/toksuite/toksuite-benchmarks">
    <img alt="HuggingFace Benchmarks" src="https://img.shields.io/badge/🤗-Benchmarks-blue"></a>
  <a href="https://arxiv.org/abs/2512.20757">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.20757-b31b1b"></a>
  <a href="https://github.com/r-three/TokSuite/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

---

**TokSuite** is a collection of models and a benchmark designed for studying how tokenizer choice affects language model behavior. By training multiple 1B-parameter models with identical architectures, data, and training budgets, varying only the tokenizer, TokSuite enables clean scientific ablations that isolate tokenization effects from confounding variables.

- **Controlled by design**: Same architecture, dataset, training budget, and initialization — only the tokenizer changes.
- **Broad coverage**: 14 tokenizers evaluated, ranging from character-level and byte-level to subword tokenizers from major model families.
- **New robustness benchmark**: A custom multilingual evaluation dataset testing model sensitivity to real-world text perturbations that affect tokenization (orthographic noise, diacritics, OCR artifacts, Unicode variants, and more).
- **Multilingual focus**: The models are trained on English, Chinese, Turkish, Italian, and Farsi, and the parallel benchmark captures real-world perturbations across all five languages by applying them to the same canonical questions translated into each target language.
- **Fully open**: Code, models, datasets, and paper are all publicly released.

See our paper for details: https://arxiv.org/abs/2512.20757

## Models

We release 14 controlled 1B-parameter models, each trained with a different tokenizer under identical conditions (Llama-3.2-1B architecture, ~100B token training budget):

| Tokenizer | Method | Vocab. Size | Languages | HuggingFace |
|-----------|--------|-------------|-----------|-------------|
| ByT5 | Bytes | 259 | Language-agnostic | [toksuite/google-byt5-small](https://huggingface.co/toksuite/google-byt5-small) |
| TokenMonster | Custom | 32,000 | English-only | [toksuite/tokenmonster-englishcode-32000-consistent-v1](https://huggingface.co/toksuite/tokenmonster-englishcode-32000-consistent-v1) |
| Phi-3 | BPE | 32,064 | Multilingual | [toksuite/microsoft-Phi-3-mini-4k-instruct](https://huggingface.co/toksuite/microsoft-Phi-3-mini-4k-instruct) |
| GPT-2 | BPE | 50,257 | English-only | [toksuite/gpt2](https://huggingface.co/toksuite/gpt2) |
| Comma | BPE | 64,000 | Multilingual | [toksuite/common-pile-comma-v0.1](https://huggingface.co/toksuite/common-pile-comma-v0.1) |
| mBERT | WordPiece | 110,000 | Multilingual | [toksuite/google-bert-bert-base-multilingual-cased](https://huggingface.co/toksuite/google-bert-bert-base-multilingual-cased) |
| Llama-3.2 | BPE | 128,256 | Multilingual | [toksuite/meta-llama-Llama-3.2-1B](https://huggingface.co/toksuite/meta-llama-Llama-3.2-1B) |
| Tekken | BPE | 130,000 | Multilingual | [toksuite/mistralai-tekken](https://huggingface.co/toksuite/mistralai-tekken) |
| Qwen-3 | BPE | 151,646 | Multilingual | [toksuite/Qwen-Qwen3-8B](https://huggingface.co/toksuite/Qwen-Qwen3-8B) |
| GPT-4o | BPE | 200,000 | Multilingual | [toksuite/tiktoken-gpt-4o](https://huggingface.co/toksuite/tiktoken-gpt-4o) |
| BLOOM | BPE | 250,680 | Multilingual | [toksuite/bigscience-bloom](https://huggingface.co/toksuite/bigscience-bloom) |
| Aya | BPE | 255,029 | Multilingual | [toksuite/CohereLabs-aya-expanse-8b](https://huggingface.co/toksuite/CohereLabs-aya-expanse-8b) |
| XGLM | Unigram | 256,008 | Multilingual | [toksuite/facebook-xglm-564M](https://huggingface.co/toksuite/facebook-xglm-564M) |
| Gemma-2 | Unigram | 256,128 | Multilingual | [toksuite/google-gemma-2-2b](https://huggingface.co/toksuite/google-gemma-2-2b) |

All models share the same initialization via a **super-vocabulary** approach, ensuring fair comparison.

## Datasets

### TokSuite Pretraining Data
A multilingual corpus of ~100B tokens used to train all suite models:
- 40B tokens from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (English)
- 60B tokens distributed across Chinese, Turkish, Italian, and Farsi

Available at: [toksuite/toksuite-pretraining-data](https://huggingface.co/collections/toksuite/training-data-detokenized)

### TokSuite Robustness Benchmark
A parallel collection of multiple-choice text completion questions paired with a wide range of real-world surface-form perturbations that are known to interact strongly with tokenization covering English, Farsi, Turkish, Italian, and Chinese languages as well as STEM and Math domains.

Available at: [toksuite/toksuite-robustness](https://huggingface.co/collections/toksuite/toksuite-benchmarks)

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) (install with `pip install uv` if not already available).

#### Using UV (Recommended)

```bash
git clone https://github.com/r-three/TokSuite.git
cd TokSuite
uv venv --python 3.10
source .venv/bin/activate

# First-time setup: build dependencies (torch, etc.), then all extras
uv sync --extra build
uv sync --all-extras
```

On machines without a GPU, skip `flash-attn`:

```bash
uv sync --all-extras --all-groups --no-install-package flash-attn
```

#### Using pip

```bash
git clone https://github.com/r-three/TokSuite.git
cd TokSuite
pip install -e .
pip install -e .[compile]   # optional: adds flash-attn
```

### Optional Dependencies

- `flash-attn` — efficient attention kernels (requires matching CUDA/PyTorch versions)
- `vllm` — faster inference for large-scale evaluation

## Usage

Evaluation is config-driven. Sample configs for all supported tasks are in `toksuite/configs/`.

### Running Evaluation

TokSuite uses a fork of [lm-evaluation-harness](https://github.com/r-three/lm-evaluation-harness) with multilingual and robustness task support built in.

```bash
# Evaluate on MGSM (multilingual math reasoning)
uv run eval toksuite/configs/mgsm/mgsm_eval_llama8B.yaml

# Evaluate on the TokSuite Robustness Benchmark
uv run eval toksuite/configs/tokenization_robustness/eval_llama8B.yaml
```

You can override any config field from the command line, or create your own YAML config pointing to any HuggingFace model.

### Computing Intrinsic Tokenizer Metrics

Compare tokenizers on fertility, parity, vocabulary overlap, and other intrinsic properties:

```bash
python -m toksuite.scripts.calculate_intrinsic_tokenizer_metrics \
  --tokenizers meta-llama/Llama-3.2-1B google/byt5-small bigscience/bloom \
  --languages en zh tr it fa \
  --output_dir results/intrinsic_metrics/
```

### Comparing Tokenizers

Analyze and visualize differences across tokenizers:

```bash
token-alysis \
  --tokenizers meta-llama/Llama-3.2-1B Qwen/Qwen3-8B \
  --text "Your input text here"
```

### Building the Super-Vocabulary

To reproduce the TokSuite models, you first need to build the **super vocabulary** described in Section 3.2 of the paper. The super vocabulary is the union of all 14 tokenizer vocabularies (normalized to UTF-8 bytes), along with per-tokenizer alignment mappings used to initialize shared embedding weights across models.

**Step 1 — Extract the tiktoken file for GPT-4o** (required before building the super vocab, since tiktoken tokenizers are not available on HuggingFace):

```bash
python -m toksuite.scripts.create_tiktoken gpt-4o \
  --output vocabs/tiktoken-gpt-4o/gpt-4o.tiktoken
```

**Step 2 — Build the super vocabulary** for all 14 TokSuite tokenizers:

```bash
# Convenience script — runs all 14 tokenizers used in the paper
bash toksuite/scripts/build_super_vocab.sh

# Or invoke directly with a custom set of tokenizers
python -m toksuite.scripts.super_vocab \
  --tokenizers \
    google/byt5-small \
    toksuite/tokenmonster-englishcode-32000-consistent-v1 \
    microsoft/Phi-3-mini-4k-instruct \
    openai-community/gpt2 \
    nikandish/common-pile-comma-v0.1 \
    google-bert/bert-base-multilingual-cased \
    meta-llama/Llama-3.2-1B \
    mistralai/Mistral-7B-v0.3 \
    Qwen/Qwen3-8B \
    vocabs/tiktoken-gpt-4o/gpt-4o.tiktoken \
    bigscience/bloom \
    CohereLabs/aya-expanse-8b \
    facebook/xglm-564M \
    google/gemma-2-2b \
  --output_dir vocabs/
```

**Outputs** in `vocabs/`:

| File | Description |
|------|-------------|
| `super_vocab.json` | Master vocabulary mapping token string → super-vocab index |
| `{tokenizer}_super_mapping.json` | Per-tokenizer alignment: original token ID → super-vocab ID |
| `{tokenizer}_vocab.json` | Original vocabulary for each tokenizer |
| `{tokenizer}.yaml` | Tokenizer metadata |

The `super_vocab.json` and `*_super_mapping.json` files are then used as the embedding initialization for model training (see Section 3.2 of the [paper](https://arxiv.org/abs/2512.20757)).

## Results

TokSuite produces novel findings about how tokenizer choice affects:
- Downstream task performance across standard benchmarks (HellaSwag, ARC, PIQA)
- Multilingual capabilities (XNLI, MGSM across 11 languages)
- Robustness to real-world text perturbations
- The relationship between intrinsic tokenizer properties (fertility, vocabulary overlap) and model performance

Full results and analysis are available in the [paper](https://arxiv.org/abs/2512.20757).

## Citation

If you use TokSuite in your work please cite the paper below. BibTeX entries are provided for convenience.

```bibtex
@inproceedings{altintas2026toksuite,
  author       = {G{"u}l Sena Altınta\c{s} and Malikeh Ehghaghi and Brian Lester and Fengyuan Liu and Wanru Zhao and Marco Ciccone and Colin Raffel},
  title        = {{TokSuite}: Measuring the Impact of Tokenizer Choice on Language Model Behavior},
  booktitle    = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year         = {2026},
  eprint       = {2512.20757},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2512.20757}
}
```


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
