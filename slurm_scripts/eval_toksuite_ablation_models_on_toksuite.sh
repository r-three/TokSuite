#!/bin/bash
#SBATCH --job-name=eval_supertoken_models
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00

##############################################################################
# eval_toksuite_ablation_models_on_toksuite.sh
#
# Purpose:
#   Run `lm_eval` over a set of TokSuite models on multiple tasks using `srun`.
#	To reproduce Table 1 in the paper.
#
# Usage (run on a Slurm login node):
#   sbatch eval_toksuite_ablation_models_on_toksuite.sh
#   or for interactive testing (no sbatch):
#   ./eval_toksuite_ablation_models_on_toksuite.sh -o /tmp/out --mem 20G --gpus 1 --dry-run
##############################################################################

REPO_HOME_DIR="$(realpath "$(dirname "$(realpath "$0")")/..")"
cd "$REPO_HOME_DIR"
echo "$REPO_HOME_DIR"

set -euo pipefail
IFS=$'\n\t'

# --- Defaults (change if necessary) ---------------------------------------
OUT_DIR="${REPO_HOME_DIR}/results/toksuite"
MEM="18G"      		# memory for each srun job
GPUS_PER_JOB=1 		# number of GPUs to request for each srun job
CPUS_PER_TASK=1
TIME_PER_JOB="60" 	# minutes, toksuite is usually pretty quick to evaluate
DRY_RUN=0 			# Set to 1 to print commands without executing them (for testing)

# Ablation TokSuite models/tokenizers 
models=(
	"meta-llama-Llama-3.2-7B"
	"meta-llama-Llama-3.2-300M"
	"meta-llama-Llama-3.2-1B-textmatched"
	"Qwen-Qwen3-8B-textmatched"
	"common-pile-comma-v0.1-textmatched"
	"google-gemma-2-2b-textmatched"
)

tokenizers=(
	"meta-llama/Llama-3.2-1B"
	"meta-llama/Llama-3.2-1B"
	"meta-llama/Llama-3.2-1B"
	"Qwen/Qwen3-8B"
	"common-pile/comma-v0.1-1t"
	"google/gemma-2-2b"
)

TASKS="toksuite_math,toksuite_english,toksuite_stem,toksuite_turkish,toksuite_italian,toksuite_chinese,toksuite_farsi"

# Load modules and activate venv if needed (adapt paths to your cluster).
# These lines are cluster-specific and may be commented out on other systems.
# module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 cuda/12.6 python/3.10.13 
# source "$REPO_HOME_DIR/.venv/bin/activate" 

mkdir -p "$OUT_DIR/logs"

num_models=${#models[@]}
echo "Will evaluate $num_models models on tasks: $TASKS"

# Validate arrays
if [ ${#models[@]} -ne ${#tokenizers[@]} ]; then
	echo "ERROR: models and tokenizers arrays must have the same length." >&2
	exit 2
fi

# Iterate models and submit srun evaluation jobs
for i in "${!models[@]}"; do
	model="${models[i]}"
	tokenizer="${tokenizers[i]}"
	hf_out_path="toksuite/$model"
	run_name="$(basename "$model")-$(date +%s)"

		wandb_args=(
			--wandb_args
			"project=toksuite-reeval,name=$run_name"
		)
	common_args=(
		--model hf
		--model_args "pretrained=${hf_out_path},tokenizer=${tokenizer},trust_remote_code=true,dtype=bfloat16"
		--device cuda
		--verbosity DEBUG
		--batch_size 1
	)

	# Build the srun command as an array for safe quoting
	srun_cmd=(srun --ntasks=1 --nodes=1 --gres=gpu:l40s:${GPUS_PER_JOB} --cpus-per-task ${CPUS_PER_TASK} --mem=${MEM} --time ${TIME_PER_JOB} --job-name="${run_name}" -o "$OUT_DIR/logs/${run_name}_%j_%t.log" -e "$OUT_DIR/logs/${run_name}_%j_%t.err" lm_eval)

	# Append lm_eval arguments
		srun_cmd+=("${common_args[@]}" --log_samples --tasks "$TASKS" "${wandb_args[@]}" --output_path "$OUT_DIR")

	echo "----"
	echo "Model: $model"
	echo "Tokenizer: $tokenizer"
	echo "Task: $TASKS"
	printf 'Command: '
	printf '%s ' "${srun_cmd[@]}"
	echo

	if [ "$DRY_RUN" -eq 1 ]; then
		echo "(dry-run) skipping execution"
	else
		"${srun_cmd[@]}" &
	fi
done

wait

echo "All jobs submitted. Logs: $OUT_DIR/logs"
