#!/bin/bash
# Reverse-direction chain (paper App. C.6): nuScenes SG->BS on LFM2-VL
set -euo pipefail
cd .
PY=python
export CUDA_VISIBLE_DEVICES=0
D=lfm2_vl_pipeline_rev/outputs
mkdir -p "$D"
log(){ echo "[$(date +"%F %T")] $*" | tee -a "$D/rev_chain.log"; }

log "===== REVERSE CHAIN START (SG->BS, LFM2-VL, GPU0) ====="

log "STEP 0: prepare_vl_data"
$PY lfm2_vl_pipeline_rev/prepare_vl_data.py > "$D/prepare.log" 2>&1
$PY - <<PYEOF
import json
m = json.load(open("lfm2_vl_pipeline_rev/outputs/data/meta.json"))
assert m["id_city"] == "singapore" and m["ood_city"] == "boston", m
print("meta OK:", m["id_city"], "->", m["ood_city"], "| id_kf:", m["id_keyframes"], "ood_kf:", m["ood_keyframes"])
PYEOF
log "STEP 0 done"

log "STEP 1: phase0_vl_finetune (source planner, Singapore)"
$PY lfm2_vl_pipeline_rev/phase0_vl_finetune.py > "$D/phase0_finetune.log" 2>&1
if [ ! -f lfm2_vl_pipeline_rev/outputs/checkpoints/phase0_vl_merged/config.json ]; then
  log "phase0_vl_merged missing -> running merge_lora"
  $PY lfm2_vl_pipeline_rev/merge_lora.py > "$D/merge_lora.log" 2>&1
fi
log "STEP 1 done"

log "STEP 2: visual_grounding_v2_finetune (TVG recipe)"
$PY lfm2_vl_pipeline_rev/visual_grounding_v2_finetune.py > "$D/vg2_finetune.log" 2>&1
test -f lfm2_vl_pipeline_rev/outputs/checkpoints/visual_grounding_v2/vg2_merged/config.json
log "STEP 2 done"

log "STEP 3: phase1 CIS diagnosis (100 pairs)"
$PY lfm2_vl_pipeline_rev/phase1_vl_counterfactual_vg2.py --max_pairs 100 > "$D/phase1_cis.log" 2>&1
test -f lfm2_vl_pipeline_rev/outputs/phase1_vl_vg2/layer_partition.json
log "STEP 3 done. Partition:"
cat lfm2_vl_pipeline_rev/outputs/phase1_vl_vg2/layer_partition.json | tee -a "$D/rev_chain.log" || true

log "STEP 4: phase2 all strategies (cis_selective + 4 ablations)"
$PY lfm2_vl_pipeline_rev/phase2_vl_finetune_vg2.py --strategy all > "$D/phase2_all.log" 2>&1
log "STEP 4 done"

log "STEP 5: evaluate all models"
$PY lfm2_vl_pipeline_rev/phase2_vg2_evaluate.py > "$D/eval.log" 2>&1
log "===== REVERSE CHAIN COMPLETE ====="
