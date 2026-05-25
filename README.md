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
  <a href="https://huggingface.co/toksuite">
    <img alt="HuggingFace Models" src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://huggingface.co/datasets/toksuite">
    <img alt="HuggingFace Datasets" src="https://img.shields.io/badge/🤗-Datasets-blue"></a>
  <a href="https://arxiv.org/abs/2512.20757">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.20757-b31b1b"></a>
  <a href="https://github.com/r-three/TokSuite/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

---

**TokSuite** is a comprehensive, controlled benchmark suite for studying how tokenizer choice affects language model behavior. By training multiple 1B-parameter models with identical architectures, data, and training budgets — varying only the tokenizer — TokSuite enables clean scientific ablations that isolate tokenization effects from confounding variables.

- **Controlled by design**: Same architecture, dataset, training budget, and initialization — only the tokenizer changes.
- **Broad coverage**: 14 tokenizers evaluated, ranging from character-level and byte-level to subword tokenizers from major model families.
- **New robustness benchmark**: A custom evaluation dataset testing model sensitivity to real-world text perturbations that affect tokenization (orthographic noise, diacritics, OCR artifacts, Unicode variants, and more).
- **Multilingual focus**: Evaluation across English, Farsi, Turkish, Chinese, and Italian languages.
- **Fully open**: Code, models, datasets, and paper are all publicly released.

See our paper for details: https://arxiv.org/abs/2512.20757

## Models

We release 14 controlled 1B-parameter models, each trained with a different tokenizer under identical conditions (Llama-3.2-1B architecture, ~100B token training budget):

| Model | Tokenizer | Vocabulary Size | HuggingFace |
|-------|-----------|----------------|-------------|
| **TokSuite-Llama-GPT2** | GPT-2 | 50,257 | [toksuite/toksuite-llama-gpt2](https://huggingface.co/toksuite/toksuite-llama-gpt2) |
| **TokSuite-Llama-Llama3** | Llama-3.2-1B | 128,256 | [toksuite/toksuite-llama-llama3](https://huggingface.co/toksuite/toksuite-llama-llama3) |
| **TokSuite-Llama-Qwen3** | Qwen3-8B | 151,936 | [toksuite/toksuite-llama-qwen3](https://huggingface.co/toksuite/toksuite-llama-qwen3) |
| **TokSuite-Llama-Gemma2** | Gemma-2-2B | 256,000 | [toksuite/toksuite-llama-gemma2](https://huggingface.co/toksuite/toksuite-llama-gemma2) |
| **TokSuite-Llama-Phi3** | Phi-3-mini | 32,064 | [toksuite/toksuite-llama-phi3](https://huggingface.co/toksuite/toksuite-llama-phi3) |
| **TokSuite-Llama-Mistral** | Mistral Tekken | 131,072 | [toksuite/toksuite-llama-mistral](https://huggingface.co/toksuite/toksuite-llama-mistral) |
| **TokSuite-Llama-GPT4o** | tiktoken (GPT-4o) | 200,019 | [toksuite/toksuite-llama-gpt4o](https://huggingface.co/toksuite/toksuite-llama-gpt4o) |
| **TokSuite-Llama-Aya** | Aya-Expanse-8B | 255,000 | [toksuite/toksuite-llama-aya](https://huggingface.co/toksuite/toksuite-llama-aya) |
| **TokSuite-Llama-BLOOM** | BLOOM | 250,680 | [toksuite/toksuite-llama-bloom](https://huggingface.co/toksuite/toksuite-llama-bloom) |
| **TokSuite-Llama-mBERT** | mBERT | 119,547 | [toksuite/toksuite-llama-mbert](https://huggingface.co/toksuite/toksuite-llama-mbert) |
| **TokSuite-Llama-XGLM** | XGLM-564M | 256,008 | [toksuite/toksuite-llama-xglm](https://huggingface.co/toksuite/toksuite-llama-xglm) |
| **TokSuite-Llama-ByT5** | ByT5 (byte-level) | 384 | [toksuite/toksuite-llama-byt5](https://huggingface.co/toksuite/toksuite-llama-byt5) |
| **TokSuite-Llama-CommonPile** | Common Pile | 65,536 | [toksuite/toksuite-llama-commonpile](https://huggingface.co/toksuite/toksuite-llama-commonpile) |
| **TokSuite-Llama-TokenMonster** | TokenMonster | 32,000 | [toksuite/toksuite-llama-tokenmonster](https://huggingface.co/toksuite/toksuite-llama-tokenmonster) |

All models share the same initialization via a **super-vocabulary** approach, ensuring fair comparison.

## Datasets

### TokSuite Pretraining Data
A multilingual corpus of ~100B tokens used to train all suite models:
- 40B tokens from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (English)
- 60B tokens distributed across Chinese, Turkish, Italian, and Farsi

Available at: [toksuite/toksuite-pretraining-data](https://huggingface.co/datasets/toksuite/toksuite-pretraining-data)

### TokSuite Robustness Benchmark
A custom evaluation dataset testing model sensitivity to text perturbations that affect tokenization in real-world conditions. Covers:
- Orthographic and spelling variations
- Diacritic insertion and removal
- Keyboard and input-method noise
- Unicode homoglyphs and formatting artifacts
- OCR and spacing errors
- LaTeX and STEM-style formatting
- Morphological challenges

Available at: [toksuite/toksuite-robustness](https://huggingface.co/datasets/toksuite/toksuite-robustness)

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
uv run eval toksuite/configs/mgsm/mgsm_eval_llama3_1B.yaml

# Evaluate on the TokSuite Robustness Benchmark
uv run eval toksuite/configs/tokenization_robustness/eval_llama3_1B.yaml
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

## Results

TokSuite produces novel findings about how tokenizer choice affects:
- Downstream task performance across standard benchmarks (HellaSwag, ARC, PIQA)
- Multilingual capabilities (XNLI, MGSM across 11 languages)
- Robustness to real-world text perturbations
- The relationship between intrinsic tokenizer properties (fertility, vocabulary overlap) and model performance

Full results and analysis are available in the [paper](https://arxiv.org/abs/2512.20757).

## Citation

```bibtex
@article{altintas2025toksuite,
  title={TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior},
  author={Altıntaş, Gül Sena and Ehghaghi, Malikeh and Lester, Brian and Liu, Fengyuan and Zhao, Wanru and Ciccone, Marco and Raffel, Colin},
  year={2025},
  eprint={2512.20757},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.20757}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
