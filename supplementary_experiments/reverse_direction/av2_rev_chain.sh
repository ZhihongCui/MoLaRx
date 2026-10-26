#!/bin/bash
# AV2 reverse chain (paper App. C.6): Miami -> Pittsburgh on LFM2-VL
# CIS diagnosis -> cis_selective + full_lora_only repair -> evaluation.
set -euo pipefail
cd .
PY=python
export CUDA_VISIBLE_DEVICES=0
D=argoverse2_pipeline_rev/outputs
log(){ echo "[$(date +"%F %T")] $*" | tee -a "$D/av2_rev_chain.log"; }

log "===== AV2 REVERSE CHAIN START (MIA->PIT) ====="

# Guard 1: the reversed data splits really are MIA(source)/PIT(target).
$PY - <<PYEOF
import json
r = json.loads(open("argoverse2_pipeline_rev/outputs/data/train.jsonl").readline())
assert r["city"] == "MIA", r["city"]
r2 = json.loads(open("argoverse2_pipeline_rev/outputs/data/test_ood.jsonl").readline())
assert r2["city"] == "PIT", r2["city"]
print("AV2 rev data OK: train=MIA, ood=PIT")
PYEOF
log "data direction verified"

# Guard 2: the phase-2 script resolves the *reversed* pipeline config, not the
# forward one (import order matters when both pipeline copies are importable).
$PY - <<PYEOF
import sys
sys.path.insert(0, "argoverse2_pipeline_rev")
import config_av2
assert "rev" in str(config_av2.AV2_OUTPUT_DIR), config_av2.AV2_OUTPUT_DIR
assert config_av2.ID_CITY == "MIA", config_av2.ID_CITY
print("rev config import OK:", config_av2.ID_CITY, "->", config_av2.OOD_CITY)
PYEOF
log "rev config import verified"

log "STEP 1: phase1 CIS (100 pairs)"
$PY argoverse2_pipeline_rev/phase1_cis_av2.py --max_pairs 100 > "$D/phase1_cis.log" 2>&1
find argoverse2_pipeline_rev/outputs -name "*partition*" -exec cat {} \; | tee -a "$D/av2_rev_chain.log" || true

log "STEP 2: phase2 cis_selective"
$PY argoverse2_pipeline_rev/phase2_cisr_av2.py --strategy cis_selective > "$D/phase2_cis.log" 2>&1
log "STEP 3: phase2 full_lora_only"
$PY argoverse2_pipeline_rev/phase2_cisr_av2.py --strategy full_lora_only > "$D/phase2_fl.log" 2>&1

log "STEP 4: evaluate (source-only + 2 strategies)"
$PY supplementary_experiments/reverse_direction/av2_rev_eval.py > "$D/eval.log" 2>&1
log "===== AV2 REVERSE CHAIN COMPLETE ====="
