#!/bin/bash

#SBATCH --gpus=1
#SBATCH --job-name=smolvla-prewarm
#SBATCH --account=proj_1912
#SBATCH --reservation=irs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=dzagonov@hse.ru
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="type_c"
#SBATCH --output=smolvla-prewarm-%j.out
#SBATCH --error=smolvla-prewarm-%j.err

set -euo pipefail

module load python
source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate vlab_pip

cd "$HOME/VLAb_v100"
export PYTHONPATH="$HOME/VLAb_v100/src:${PYTHONPATH:-}"

# Явно фиксируем постоянный HF cache
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Временные multiprocessing-файлы
JOB_TMP="/tmp/${USER}/vlab_prewarm_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export TMP="$JOB_TMP"
export TEMP="$JOB_TMP"

export REPO_IDS="$(cat "$HOME/VLAb_v100/examples/all_datasets_relative.txt")"

python "$HOME/VLAb_v100/prewarm_dataset.py" \
    --policy.type=smolvla2 \
    --dataset.repo_id="$REPO_IDS" \
    --dataset.root="$HOME/vla_datasets" \
    --dataset.use_imagenet_stats=false \
    --dataset.image_transforms.enable=true \
    --dataset.train_on_all_features=true \
    --dataset.features_version=2 \
    --policy.max_action_dim=32 \
    --policy.max_state_dim=32 \
    --dataset.max_num_images=2 \
    --dataset.max_image_dim=256 \
    --dataset.video_backend=pyav \
    --num_workers=4 \
    --dataset.min_fps=30 \
    --dataset.max_fps=30 \
    --policy.num_vlm_layers=16 \
    --policy.expert_width_multiplier=0.75 \
    --policy.causal_action_attention_mask=true \
    --policy.self_attn_every_n_layers=2 \
    --policy.attention_mode=cross_attn \
    --policy.prefix_length=0 \
    --policy.peft_method=lora \
    --policy.peft_config.r=32 \
    --policy.peft_config.target_modules=q_proj,v_proj \
    --policy.load_vlm_weights=true \
    --policy.repo_id=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.push_to_hub=false \
    --output_dir="$HOME/smolvla_outputs/prewarm_dummy" \
    --wandb.enable=false \
    --trackio.enable=false

echo "Dataset prewarming completed."
