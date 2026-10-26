#!/bin/bash
# Reverse-direction chain (paper App. C.6): nuScenes SG->BS, LFM2-VL-1.6B, full pipeline
#SBATCH --job-name=lfm2_rev
#SBATCH --account=<hpc-project>
#SBATCH --partition=accel
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=lfm2_vl_pipeline_rev/outputs/logs/%x_%j.out
#SBATCH --error=lfm2_vl_pipeline_rev/outputs/logs/%x_%j.err

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
# Pre-flight: fail fast if this node/GPU cannot init CUDA
python - <<PYEOF
import torch, os, sys
ok = torch.cuda.is_available()
print("preflight: CVD=", os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_available=", ok)
sys.exit(0 if ok else 17)
PYEOF
echo "=== lfm2_rev chain start: $(date) node=$(hostname) ==="

echo "--- STEP 0: prepare_vl_data ---"
python lfm2_vl_pipeline_rev/prepare_vl_data.py
python - <<PYEOF
import json
m = json.load(open("lfm2_vl_pipeline_rev/outputs/data/meta.json"))
assert m["id_city"] == "singapore" and m["ood_city"] == "boston", m
print("meta OK:", m["id_city"], "->", m["ood_city"], "| id_kf:", m["id_keyframes"], "ood_kf:", m["ood_keyframes"])
PYEOF

echo "--- STEP 1: phase0_vl_finetune ---"
python lfm2_vl_pipeline_rev/phase0_vl_finetune.py
if [ ! -f lfm2_vl_pipeline_rev/outputs/checkpoints/phase0_vl_merged/config.json ]; then
  echo "phase0_vl_merged missing -> merge_lora"
  python lfm2_vl_pipeline_rev/merge_lora.py
fi

echo "--- STEP 2: visual_grounding_v2_finetune ---"
python lfm2_vl_pipeline_rev/visual_grounding_v2_finetune.py
test -f lfm2_vl_pipeline_rev/outputs/checkpoints/visual_grounding_v2/vg2_merged/config.json

echo "--- STEP 3: phase1 CIS (100 pairs) ---"
python lfm2_vl_pipeline_rev/phase1_vl_counterfactual_vg2.py --max_pairs 100
echo "--- reverse partition: ---"
cat lfm2_vl_pipeline_rev/outputs/phase1_vl_vg2/layer_partition.json || true

echo "--- STEP 4: phase2 all strategies ---"
python lfm2_vl_pipeline_rev/phase2_vl_finetune_vg2.py --strategy all

echo "--- STEP 5: evaluate ---"
python lfm2_vl_pipeline_rev/phase2_vg2_evaluate.py

echo "=== LFM2 REV CHAIN COMPLETE: $(date) ==="
