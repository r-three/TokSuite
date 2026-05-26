#!/bin/bash
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

source /home/ehghaghi/projects/aip-craffel/ehghaghi/TokSuite/.venv/bin/activate


export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
export HF_HOME=$SCRATCH/huggingface


# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/calculate_intrinsic_tokenizer_metrics.py"

# Default configuration
DEFAULT_SAMPLE_SIZE=10000

# Default tokenizers (comma-separated)
TOKENIZERS="Comma,Llama-3.2,Phi-3,GPT-2,GPT-4o,BLOOM,XGLM,Tekken,ByT5,mBERT,Qwen-3,TokenMonster,Gemma-2,Aya"

# Default languages (comma-separated)
LANGUAGES="sentence_eng_Latn,sentence_zho_Hans,sentence_tur_Latn,sentence_pes_Arab,sentence_ita_Latn"

# Save flores200 to Arrow format once (requires datasets < 3.0 to support the Python loading script)
DATASET_PATH="$SCRATCH/flores200_dev"
if [ ! -d "$DATASET_PATH" ]; then
    echo "Saving flores200 to disk for the first time..."
    pip install "datasets==2.21.0"
    python -c "
from datasets import load_dataset
ds = load_dataset('Muennighoff/flores200', 'all', split='dev', trust_remote_code=True)
ds.save_to_disk('$DATASET_PATH')
print('Saved.')
"
    pip install "datasets==3.6.0"
fi

OUTPUT_DIR="$SCRIPT_DIR/../../results/intrinsic_tokenizers"

# Run the Python script with the tokenizers and languages
python3 calculate_intrinsic_tokenizer_metrics.py \
    --tokenizers "$TOKENIZERS" \
    --languages "$LANGUAGES" \
    --analyses all \
    --sample_size 10000 \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR"
