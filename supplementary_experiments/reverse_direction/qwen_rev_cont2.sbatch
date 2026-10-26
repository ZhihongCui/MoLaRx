#!/bin/bash
#SBATCH --job-name=qwen_rev_c2
#SBATCH --account=<hpc-project>
#SBATCH --partition=accel
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
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
echo "preflight OK"
echo "--- STEP 4: phase2 CISR (2 strategies) ---"
python qwen_vl_pipeline_rev/phase2_cisr_qwen.py --dataset nuscenes --strategy all
echo "--- STEP 5: evaluate ---"
CKPT_DIR=qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/checkpoints
for ck in "$CKPT_DIR"/*_final; do
  name=$(basename "$ck")
  echo "eval: $name"
  python qwen_vl_pipeline_rev/phase2_qwen_eval.py --dataset nuscenes \
    --ckpt "$ck" --out "qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/eval_${name}.json"
done
echo "=== QWEN REV CONT2 COMPLETE: $(date) ==="
