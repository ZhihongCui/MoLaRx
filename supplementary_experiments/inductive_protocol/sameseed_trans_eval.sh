#!/bin/bash
# Same-seed transductive leg of the protocol control (paper App. C.4).
#
# The inductive control (ind_chain.sh) trains against the disjoint adaptation
# split. To compare protocols with the seed held fixed, this script retrains the
# transductive MoLaRx (cis_selective) in the *forward* pipeline — whose CIS/MMD
# target pool is the unlabeled test_ood inputs — using the identical default seed,
# schedule, and hyperparameters as the inductive run, then evaluates both the
# source-only baseline and the retrained checkpoint on the fixed 300-sample
# ID/OOD splits with the standard protocol.
#
# Reported comparison (OOD ADE, LFM2-VL/nuScenes, same seed):
#   Source-only: transductive 4.466 = inductive 4.466 (no target data; sanity check)
#   MoLaRx:      transductive 4.266 vs inductive 4.273 (difference 0.007 m)
set -euo pipefail
cd .
PY=python
export CUDA_VISIBLE_DEVICES=0
D=lfm2_vl_pipeline/outputs
log(){ echo "[$(date +"%F %T")] $*" | tee -a "$D/sameseed_trans.log"; }

log "STEP 1: retrain cis_selective (transductive pool, shared default seed)"
$PY lfm2_vl_pipeline/phase2_vl_finetune_vg2.py --strategy cis_selective > "$D/phase2_cis_sameseed.log" 2>&1

log "STEP 2: evaluate baseline + cis_selective (fixed 300-sample splits)"
$PY lfm2_vl_pipeline/phase2_vg2_evaluate.py > "$D/eval_sameseed_trans.log" 2>&1
log "DONE"
