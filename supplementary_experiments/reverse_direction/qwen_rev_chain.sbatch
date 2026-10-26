#!/bin/bash
# Reverse-direction chain (paper App. C.6): nuScenes SG->BS, Qwen2-VL-2B, full pipeline
#SBATCH --job-name=qwen_rev
#SBATCH --account=<hpc-project>
#SBATCH --partition=accel
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=qwen_vl_pipeline_rev/outputs/logs/%x_%j.out
#SBATCH --error=qwen_vl_pipeline_rev/outputs/logs/%x_%j.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # keep Slurm value; fall back to 0
export PYTHONUNBUFFERED=1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export HF_HOME=$HF_CACHE

module purge
module load <site-gpu-stack>
module load <cuda-12.4>
source <conda>/bin/activate <env>

cd $PROJECT_DIR
echo "=== qwen_rev chain start: $(date) node=$(hostname) ==="

# Guard: reversed jsonl data must exist and point SG->BS
python - <<PYEOF
import json
m = json.load(open("lfm2_vl_pipeline_rev/outputs/data/meta.json"))
assert m["id_city"] == "singapore" and m["ood_city"] == "boston", m
print("data OK:", m["id_city"], "->", m["ood_city"])
PYEOF

echo "--- STEP 1: phase0_qwen_finetune ---"
python qwen_vl_pipeline_rev/phase0_qwen_finetune.py

echo "--- STEP 2: visual_grounding_qwen_finetune (TVG, 3 epochs) ---"
python qwen_vl_pipeline_rev/visual_grounding_qwen_finetune.py --num_epochs 3

echo "--- STEP 3: phase1 CIS probing (--tvg) ---"
python qwen_vl_pipeline_rev/phase1_qwen_probing.py --tvg

echo "--- STEP 4: phase2 CISR all strategies ---"
python qwen_vl_pipeline_rev/phase2_cisr_qwen.py --dataset nuscenes --strategy all

echo "--- STEP 5: evaluate all checkpoints ---"
CKPT_DIR=qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/checkpoints
for ck in "$CKPT_DIR"/*_final; do
  name=$(basename "$ck")
  echo "eval: $name"
  python qwen_vl_pipeline_rev/phase2_qwen_eval.py --dataset nuscenes \
    --ckpt "$ck" --out "qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/eval_${name}.json"
done
# Source-only reference: TVG merged checkpoint
TVG_CKPT=qwen_vl_pipeline_rev/outputs/checkpoints/tvg_qwen/tvg_merged
echo "eval: source-only ($TVG_CKPT)"
python qwen_vl_pipeline_rev/phase2_qwen_eval.py --dataset nuscenes \
  --ckpt "$TVG_CKPT" --out "qwen_vl_pipeline_rev/outputs/phase2_cisr_nuscenes/eval_source_only.json"

echo "=== QWEN REV CHAIN COMPLETE: $(date) ==="
