#!/bin/bash
#SBATCH --job-name=qwen_srconly
#SBATCH --account=<hpc-project>
#SBATCH --partition=accel
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=qwen_vl_pipeline_rev/outputs/logs/%x_%j.out
#SBATCH --error=qwen_vl_pipeline_rev/outputs/logs/%x_%j.err
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export HF_HOME=$HF_CACHE
module purge; module load <site-gpu-stack>; module load <cuda-12.4>
source <conda>/bin/activate <env>
cd $PROJECT_DIR
python - <<PYEOF
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 17)
PYEOF
mkdir -p qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes
python qwen_vl_pipeline_rev/phase2_qwen_eval.py --dataset nuscenes \
  --ckpt qwen_vl_pipeline_rev/outputs/checkpoints/tvg_qwen/tvg_merged \
  --out qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/eval_source_only.json
echo "SRC-ONLY EVAL DONE"
cat qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/eval_source_only.json | head -30
