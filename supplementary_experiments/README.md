# Supplementary experiments

Code for the supplementary experiments reported in Appendices C.4–C.6 of the paper.
All scripts assume the repository root as working directory, with data prepared as
in the main pipelines (`lfm2_vl_pipeline/prepare_vl_data.py`). Site-specific paths,
accounts, and module names are replaced by placeholders (`$PROJECT_DIR`, `<env>`, …).

## `kitscenes_longtail/` — zero-shot long-tail probe (paper App. C.5)

- `kitscenes_probe.py --prepare` converts the KITScenes-LongTail **val** split
  (100 nine-second scenarios, 5 Hz) into the pipeline's 6-waypoint / 0.5 s ego-frame
  format. Protocol: front-camera frames at indices 15/18/20 (t = −1.0 / −0.4 / 0.0 s,
  i.e. the model's (t−0.5, t, t+0.5) triplet with ≤ 0.1 s jitter; reference time
  t_ref = −0.4 s); the expert trajectory is linearly interpolated at
  t_ref + {0.5, …, 3.0} s and rotated into the ego frame at t_ref (heading from a
  centred difference of the past track). KITScenes coordinates are x-forward /
  y-left; the pipeline convention is (dx = left+, dy = forward+).
- `kitscenes_probe.py --eval` runs the four forward-direction checkpoints zero-shot.
- `kitscenes_mms.py --prepare-neg` resamples the annotated alternative trajectories
  (`wrong_speed`, `neglect_instruction`) with the identical transform.
- `kitscenes_mms.py --eval` computes the MMS-inspired **closest-mode accuracy**
  (prediction closer in ADE to the expert mode than to any alternative; ties — none
  occurred — would count as incorrect) and the **mode margin**
  (nearest-alternative ADE − expert ADE; positive = correct-mode preference).

## `planning_proxies/` — planning-oriented proxies (paper App. C.5)

`planning_proxies.py` re-evaluates the forward checkpoints on the fixed
300-sample ID/OOD splits, saving per-sample predictions, and reports
- lateral error at the 3 s horizon: |dx_pred − dx_gt| at the final waypoint;
- mean |jerk| of the predicted trajectory: third finite difference over the
  waypoints (origin prepended), Δt = 0.5 s; ground-truth jerk reported as reference.
These are proxies, not closed-loop safety metrics.

## `reverse_direction/` — reverse transfer (paper App. C.6)

The reverse pipelines are byte-identical copies of the forward pipelines with only
the direction flipped in the config (`ID_CITY`/`OOD_CITY` and location lists swapped,
outputs redirected to `*_rev/`). `rev_chain_lfm2.sh` is the single-GPU SG→BS chain
(prepare → phase 0 → visual grounding → CIS → repair strategies → evaluation);
the `*.sbatch` files are the HPC equivalents, including the truncated-schedule
continuations reported in App. C.6 (`qwen_rev_cont*.sbatch`) and the
PaliGemma2 diagnosis-only run (`pali_rev_cis.sbatch`).

`av2_rev_chain.sh` is the Argoverse 2 MIA→PIT chain (CIS → cis_selective +
full_lora_only → evaluation). It asserts both the data direction and that the
phase-2 script resolves the reversed pipeline config before training.
`av2_rev_eval.py` evaluates Source-only / Full-LoRA / MoLaRx on the first 300
records of the reversed `test_id` (Miami) and `test_ood` (Pittsburgh) splits;
AV2 records carry their own 3-frame image paths and VG2 question. These numbers
use this harness and are internally comparable only.

## `inductive_protocol/` — inductive adaptation control (paper App. C.4)

`ind_chain.sh` re-runs CIS diagnosis and all target-consuming repair strategies
against a disjoint unlabeled adaptation split (`adapt_ood.jsonl`: all target-city
keyframes not sampled into `test_ood.jsonl`; zero-overlap asserted at runtime),
evaluating on the untouched target test split. The pipeline copy differs from the
forward one in three one-line changes: `prepare_vl_data.py` additionally emits
`adapt_ood.jsonl`, and the CIS / MMD target-sampling paths read it instead of
`test_ood.jsonl`.

`sameseed_trans_eval.sh` is the matching transductive leg: it retrains
`cis_selective` in the forward pipeline (target pool = unlabeled `test_ood`
inputs) with the identical default seed, schedule, and hyperparameters as the
inductive run, then evaluates on the same fixed 300-sample splits. Together the
two scripts produce the seed-controlled transductive-vs-inductive comparison
(Source-only 4.466 = 4.466; MoLaRx 4.266 vs 4.273 OOD ADE, difference 0.007 m).
