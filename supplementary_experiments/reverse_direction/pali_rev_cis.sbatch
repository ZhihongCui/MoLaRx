#!/bin/bash
# Reverse-direction PaliGemma2: TVG (2 epochs, matching the other reverse cells) -> CIS partition only
#SBATCH --job-name=pali_rev_cis
#SBATCH --account=<hpc-project>
#SBATCH --partition=accel
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=paligemma_vl_pipeline_rev/outputs/logs/%x_%j.out
#SBATCH --error=paligemma_vl_pipeline_rev/outputs/logs/%x_%j.err
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
python - <<PYEOF
import json
m = json.load(open("lfm2_vl_pipeline_rev/outputs/data/meta.json"))
assert m["id_city"] == "singapore" and m["ood_city"] == "boston", m
print("data OK:", m["id_city"], "->", m["ood_city"])
PYEOF
echo "--- STEP 1: phase0 TVG (2 epochs) ---"
python paligemma_vl_pipeline_rev/phase0_paligemma_tvg.py --dataset nuscenes --num_epochs 2
echo "--- STEP 2: phase1 CIS (100 pairs) ---"
python paligemma_vl_pipeline_rev/phase1_cis_paligemma.py --dataset nuscenes --K 100
echo "--- reverse Pali partition: ---"
cat paligemma_vl_pipeline_rev/outputs/phase1_*/layer_partition*.json 2>/dev/null || find paligemma_vl_pipeline_rev/outputs -name "*partition*" -exec cat {} \;
echo "=== PALI REV CIS COMPLETE: $(date) ==="
