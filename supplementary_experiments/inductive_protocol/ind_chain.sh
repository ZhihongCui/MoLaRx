#!/bin/bash
# Inductive-protocol control experiment (paper App. C.4):
# CIS + MMD target data from adapt_ood.jsonl (disjoint from test_ood).
set -euo pipefail
cd .
PY=python
export CUDA_VISIBLE_DEVICES=1
D=lfm2_vl_pipeline_ind/outputs
log(){ echo "[$(date +"%F %T")] $*" | tee -a "$D/ind_chain.log"; }

log "===== INDUCTIVE CHAIN START (BS->SG, LFM2, GPU1) ====="

# Wait until GPU1 is free (KITScenes probe may still be running)
while pgrep -f "kitscenes_probe.py --eval" > /dev/null; do sleep 120; done
log "GPU1 free"

# Sanity: adapt_ood exists and test_ood identical to forward run
$PY - <<PYEOF
import json, hashlib
def h(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
a = "lfm2_vl_pipeline/outputs/data/test_ood.jsonl"
b = "lfm2_vl_pipeline_ind/outputs/data/test_ood.jsonl"
assert h(a) == h(b), "ind test_ood differs from forward test_ood!"
adapt = [json.loads(l) for l in open("lfm2_vl_pipeline_ind/outputs/data/adapt_ood.jsonl")]
test_tokens = {json.loads(l)["sample_token"] for l in open(b)}
overlap = sum(1 for r in adapt if r["sample_token"] in test_tokens)
assert overlap == 0, f"overlap={overlap}"
print(f"SANITY OK: test_ood identical to forward; adapt_ood={len(adapt)} records, overlap=0")
PYEOF
log "sanity passed"

log "STEP 1: phase1 CIS on adapt_ood (100 pairs)"
$PY lfm2_vl_pipeline_ind/phase1_vl_counterfactual_vg2.py --max_pairs 100 > "$D/phase1_cis.log" 2>&1
cat lfm2_vl_pipeline_ind/outputs/phase1_vl_vg2/layer_partition.json | tee -a "$D/ind_chain.log" || true

log "STEP 2: phase2 cis_selective"
$PY lfm2_vl_pipeline_ind/phase2_vl_finetune_vg2.py --strategy cis_selective > "$D/phase2_cis.log" 2>&1
log "STEP 3: phase2 full_lora_da"
$PY lfm2_vl_pipeline_ind/phase2_vl_finetune_vg2.py --strategy full_lora_da > "$D/phase2_flda.log" 2>&1
log "STEP 4: phase2 mmd_only"
$PY lfm2_vl_pipeline_ind/phase2_vl_finetune_vg2.py --strategy mmd_only > "$D/phase2_mmd.log" 2>&1

log "STEP 5: evaluate (TVG ref + 3 target-consuming strategies)"
$PY lfm2_vl_pipeline_ind/phase2_vg2_evaluate.py --models TVG_baseline CISR Full_LoRA_MMD MMD_only --skip_ablation > "$D/eval.log" 2>&1

log "===== INDUCTIVE CHAIN COMPLETE ====="
