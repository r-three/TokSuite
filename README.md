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

## Set-up
We recommend using uv (install it with `pip install uv` if not already available).

### On Killarney
On the Killarney cluster, you need to first load the following modules:
```bash
module load slurm/killarney/24.05.7 StdEnv/2023  gcc/13.3  openmpi/5.0.3 cuda/12.6 python/3.10.13
```
and for the first time you run the code, you need to install the packages to the system:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# If you don't have a virtual environment already, you can either
# 1. Install the packages to the system
uv pip install -e . --system

# 2. Create a venv with uv
# make sure to load cuda (locally built with cuda-12.4)
uv venv --python 3.10
source .venv/bin/activate
## First run
uv sync --extra build
uv sync --all-extras
# on machines w/o cuda
uv sync --all-extras --all-groups  --no-install-package flash-attn
```
If you have another uv venv, you can add this package to the original projects `pyproject.toml` as below and run `uv sync --extra tokenizers` in the main directory:
```toml
[project.optional-dependencies]
tokenizers = ["toksuite"]

[tool.uv.sources]
toksuite = { path = "../tokenizers", editable = true }
```

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

#### Extracting tiktoken Vocabulary Files

For tiktoken-based tokenizers (gpt-4o, gpt-4), Lingua — the tokenizer backend used during evaluation — requires a local `.tiktoken` file. Generate one before running evaluation on those models:

```bash
python -m toksuite.scripts.create_tiktoken gpt-4o \
    --output vocabs/tiktoken-gpt-4o/gpt-4o.tiktoken
```

| Model / alias | Encoding |
|---|---|
| `gpt-4o`, `gpt-4o-mini` | `o200k_base` |
| `gpt-4`, `gpt-3.5-turbo` | `cl100k_base` |
| `gpt-3`, `gpt-2` | `r50k_base` |

HuggingFace-backed tokenizers (Llama, Mistral, BLOOM, etc.) do not need this step.

#### Running on an HPC Cluster (SLURM)

Convenience SLURM scripts are provided for batch evaluation on an HPC cluster:

```bash
# Evaluate a single model across all tasks
sbatch slurm_scripts/eval_supertoken_models.sh

# Batch-evaluate all 14 TokSuite models across all tasks
sbatch slurm_scripts/eval_supertoken_on_completion.sh
```

> **Before submitting**, update the paths, account name, and GPU partition at the top of each script. The defaults target the Killarney cluster.

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

### Computing Robustness-Specific Intrinsic Metrics

To measure fertility, parity, and vocabulary overlap scoped to the TokSuite Robustness Benchmark dataset (rather than arbitrary text), use the robustness-specific script:

```bash
python -m toksuite.scripts.tokenizer_robustness_intrinsic_metrics \
    --dataset_name toksuite/toksuite-robustness \
    --tokenizers meta-llama/Llama-3.2-1B bigscience/bloom
```

This complements `calculate_intrinsic_tokenizer_metrics.py` by showing how tokenizers behave specifically on the perturbation types present in the benchmark.

### Building the Super-Vocabulary

To reproduce the TokSuite models, you first need to build the **super vocabulary** described in Section 3.2 of the paper. The super vocabulary is the union of all 14 tokenizer vocabularies (normalized to UTF-8 bytes), along with per-tokenizer alignment mappings used to initialize shared embedding weights across models.

Run the convenience script, which handles the tiktoken extraction and vocab build in one step:

```bash
bash toksuite/scripts/build_super_vocab.sh
```

> **Before running**, update the `SCRATCH` path at the top of `build_super_vocab.sh` to point to your own scratch or cache directory. This keeps model downloads out of your home directory.

To use a custom set of tokenizers instead, invoke the Python module directly:

```bash
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

### Publishing Super-Vocabulary Tokenizers to HuggingFace

After building the super vocabulary locally, publish one tokenizer repository per model on HuggingFace (e.g. `toksuite/supervocab-gpt2`). Each repo bundles the shared `super_vocab.json` with the per-model alignment mapping so the conversion step below can download them as a unit.

`super_vocab.py` encodes `/` as `--` in output filenames (e.g. `openai-community/gpt2` → `openai-community--gpt2_vocab.json`). Use the same replacement when locating the files to upload.

```bash
# Set variables — model_hf_name must match the ID passed to super_vocab.py
model="gpt2"
model_hf_name="openai-community/gpt2"
tokenizer_name="toksuite/supervocab-$model"
safe_name="${model_hf_name//\/'/--'}"
staging="staging/$model"

# Collect the relevant files into a staging directory
mkdir -p "$staging"
cp vocabs/super_vocab.json                       "$staging/"
cp "vocabs/${safe_name}_vocab.json"              "$staging/"
cp "vocabs/${safe_name}_super_mapping.json"      "$staging/"
cp "vocabs/${safe_name}_info.json"               "$staging/"
cp "vocabs/${safe_name}.yaml"                    "$staging/"

# Create the HuggingFace repo and push (drop --private to make it public)
huggingface-cli repo create "$tokenizer_name" --type model --private
huggingface-cli upload "$tokenizer_name" "$staging" .
```

Repeat for every tokenizer listed in `build_super_vocab.sh`. The next step downloads these repos and reads `super_vocab.json` (matched as `*vocab.json`) to determine the embedding size and remap individual token embeddings via `*_super_mapping.json`.

### Training the TokSuite Models

Reproducing the full suite requires three training phases. The actual LLM training is not part of this repository — use any standard pre-training framework configured for the Llama-3.2-1B architecture and the TokSuite pretraining data (see [Datasets](#datasets)).

**Phase 1 — Train the superset model**

Train a single Llama-1B model whose embedding and output layers are sized to the full super vocabulary (`len(super_vocab.json)` tokens, approximately 600 K). Use `super_vocab.json` as the vocabulary. This model does not need to fully converge; its purpose is to produce meaningful shared initial embeddings for all 14 tokenizers. Save the raw checkpoint as a `.pth` file:

```
checkpoints/llama1b_supertoken/model.pth
```

**Phase 2 — Slice per-tokenizer starting checkpoints**

Once the superset checkpoint exists, run `create_checkpoint.py` to extract tokenizer-specific embedding rows from it. For each tokenizer it reads `{tokenizer}_super_mapping.json` to find which super-vocab rows belong to that tokenizer, slices `tok_embeddings.weight` and `output.weight` accordingly, and saves the result as a warm-started checkpoint.

Run all tokenizers at once with the convenience script:

```bash
bash toksuite/scripts/create_checkpoints.sh
```

> **Before running**, update `--superset` and `--model_dir` in `create_checkpoints.sh` to point to your superset checkpoint and output directory.

To run for a single tokenizer:

```bash
python -m toksuite.scripts.create_checkpoint \
    --tokenizers meta-llama/Llama-3.2-1B \
    --vocab_dir vocabs/ \
    --superset checkpoints/llama1b_supertoken/model.pth \
    --model_dir checkpoints/
```

Output: `checkpoints/supervocab-meta-llama--Llama-3.2-1B/model.pth`

**Phase 3 — Train per-tokenizer models**

Train each of the 14 models from its Phase 2 checkpoint. Each model uses its own tokenizer's vocabulary while sharing the same Llama-3.2-1B architecture, dataset, and ~100B token budget. Initialize `tok_embeddings.weight` and `output.weight` from the corresponding `supervocab-{tokenizer}/model.pth`. After training, convert each native checkpoint to HuggingFace format using the step below.

### Converting Models to the Super-Vocabulary

Once you have the super vocabulary, you can convert a trained model's weights to the super-vocabulary format using `toksuite/scripts/convert_supertoken_models.py`. This downloads the raw model and its tokenizer from HuggingFace, remaps the embeddings, and saves (or pushes) the converted model.

```bash
# Set your paths
model="gpt2"
model_name="toksuite/$model"
tokenizer_name="toksuite/supervocab-$model"
hf_model_path="models/$model_name"
tokenizer_path="tokenizers/$model"
output_dir="converted/$model"   # or a HuggingFace repo id if pushing

# Download model and tokenizer
huggingface-cli download $model_name --local-dir $hf_model_path
huggingface-cli download $tokenizer_name --local-dir $tokenizer_path

# Convert weights to HuggingFace format with the super-vocabulary tokenizer
python -m toksuite.scripts.convert_supertoken_models \
    --input_dir "$hf_model_path" \
    --model_size 1B \
    --llama_version 3 \
    --tokenizer_version 3 \
    --tokenizer_path "$tokenizer_path" \
    --output_dir "$output_dir"
```

To push directly to HuggingFace Hub instead of saving locally, add `--push_to_hub --public --only_model` and set `--output_dir` to your Hub repo id (e.g. `your-username/model-name`).

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
