# Experiments — Sprint 2 Leaderboard Optimization

## 2026-09-22 — Local background review viewer

Added scripts/review_background_app.py: real Z-slice display, adjustable contrast,
explicit box proposals, reviewer notes and confirmation, GEFF conflict checks,
source-image hash validation, sampled-negative check and exclusive reviewed-file
save. Empty/uncertain regions are never labeled automatically. Streamlit AppTest
render passed, with confirmation unchecked by default. No real region has been
reviewed and no training started. Expert visual review remains a data dependency.
Launch procedure in docs/BACKGROUND_REVIEW.md.


## 2026-09-22 — Reviewed-background supervision workflow

Implemented background_review.py and prepare_background_review.py. Sparse trainer
now accepts image-hashed JSON review records instead of anonymous NPY masks.
Unreviewed/missing-provenance/empty/out-of-bounds/conflicting labels are rejected;
all unspecified voxels remain unknown.120 tests pass;61 baseline files unchanged.
One existing local Biohub frame exported as TIFF plus unreviewed template under
C:/Projects/cell-tracking/reviews/background. No new data download, no fabricated
review or new training run. Real knowledgeable review still required; this does
not complete supervision acquisition or support full199 training yet. Procedure:
docs/BACKGROUND_REVIEW.md. Background labels alone do not establish precision
evaluation coverage.


## 2026-09-22 — Dependency/gating fixes and successful preflight

Sparse preflight Version4 (351787566) completed:199 paired volumes inventoried,
two patches from distinct embryo groups, TeslaT4 forward/backward passed,
unknown-output gradient exactly zero. Zero optimizer steps; no trained model.
Verified negative labels absent. Artifacts: results/kaggle_sparse_preflight_v4.

Checkpoint existence now checked before torch import. Learned linker now gates
before feature allocation and inference, preserving full-source ranks/counts;
dense distance/cost matrices remain. Regression covers both10/25-feature models,
rank ties, gated inference and missing torch/checkpoint ordering. Removed float64
promotion from detector inference. No accuracy improvement or timeout cause
claimed without profiling. CHO improvement was sequence01 validation, not untouched
held-out sequence02. Full training still needs defensible supervision.


## 2026-09-22 — Sparse preflight Version2 failed; Version3 repair dispatched

Version2 failed before GPU checks: GEFF loader returned None when zarr was unavailable, then nodes.empty raised AttributeError. No training or compatibility pass. Source/logs/hashes preserved in results/kaggle_sparse_preflight_v2. Fixed loader to attempt existing explicit compatibility reader when zarr is missing and raise FileNotFoundError for missing GEFF. Added two regression tests;114 pytest tests pass,61 baseline files unchanged. Version3 installs missing codec dependencies and reports versions, then runs same bounded checks. Unknown labels and no-optimizer policy unchanged; full training remains blocked by absent verified negative supervision.


## 2026-09-22 — Combined Biohub sparse-supervision/GPU preflight

Private dushi2/biohub-sparse-detector-preflight Version1 dispatched after no
matching existing notebook found. Seven sparse target/trainer tests passed.
Frozen 26-file source manifest: notebooks/kaggle_sparse_preflight. Uses mounted
competition data; inventories paired volumes and selects first volume from each
of two embryo prefixes before metrics, one annotated-frame patch per sample.
Checks actual image I/O, GPU model forward/backward and zero unknown-output
gradient. Same frame duplicated for tensor compatibility; no temporal claim.

This is NOT useful detector training: no optimizer step, no trained checkpoint.
No verified background labels have been supplied; positive-only BCE can collapse
to all-positive predictions. Expected successful state is compatibility passed
but training blocked. Need validated alternative sparse-label objective or
verified background annotation before full training. Do not count this run as
completion of training step1 or full199 training. No raw data downloaded locally.


## 2026-09-22 — Training-side detector failure audit and padding ablation

Reproduced selected checkpoint validation counts exactly on CPU: TP29, FP69,
FN20 across sequence01 frames70/77/84/91. Saved per-observation 3D matching,
localization distances, boundary and annotated-neighbor measurements plus four
XY projection overlays in results/cho_validation_failure_audit. Projection
inspection suggests multiple bright intranuclear peaks; this is not proof of
a unique cause. Combined 3D boundary measure is uninformative here because
all points lie within1um of a boundary; separate XY distances are recorded.

Confirmed training bug: real depth5 was padded to8 with replicated images but
zero target labels. Fixed crop size to available image dimensions, removing
contradictory artificial negative slices. Six CTC tests pass, including new
regression;112 pytest tests pass,61 baseline files unchanged.

Next bounded ablation uses same notebook Version3, same seed, max30 epochs,
30 steps, model, losses and threshold grid, but corrected crops. Frozen source
and protocol in notebooks/kaggle_cho_detector. Evaluation is sequence01
VALIDATION ONLY; sequence02 report stays frozen, no repeat held-out selection.
Compare best validation F1 to Version2's0.3945578 with recall/precision; this
is a development diagnostic, not an adoption test or independent improvement.
No biological or Biohub accuracy claim.


## 2026-09-22 — CHO Version 2 completed, limited detector feasibility

Kaggle version 351776811 COMPLETE. Tesla T4; 19,425-parameter detector,
27 epochs (early stopping); selected epoch 15 and threshold 0.4 using sequence
01 validation only. Sequence 02 evaluated once on 19 frames / 126 annotated
cell observations. Learned TP/FP/FN = 92/328/34, precision 0.219048, recall
0.730159, F1 0.336996. Shallow-volume classical control = 111/1167/15,
precision 0.086854, recall 0.880952, F1 0.158120. Original defaults F1
0.077922. Learned inference 0.815 s vs adapted control 1.523 s across 19 frames
on this run; hardware paths differ, not a general speed claim.

Decision: preserve checkpoint; do not promote to Biohub. Higher F1 comes with
19 fewer true detections than adapted control, 328 false detections remain,
and no tracking/division or zebrafish validation exists. One held-out sequence
cannot establish an embryo-level confidence interval or satisfy Biohub adoption
rule. No post-test threshold tuning. Next: verify image overlays/localization,
label coverage and boundary/density failure modes, then freeze training-side
improvements and obtain independent validation before any adoption.

Evidence: results/kaggle_cho_detector_v2 contains source, logs, checkpoint,
history, provenance, frozen protocol and report with SHA256 manifest. All 22
packaged source hashes, checkpoint/report hash and 57 row aggregates verified.
All 61 original baseline files verified unchanged. This advances Phase 9/G
feasibility only; it does not complete detection or the broader roadmap.


## 2026-09-22 — CHO Version 1 infrastructure failure

Kaggle reported ERROR. Training failed while reading the first LZW TIFF: missing imagecodecs. No epoch or held-out result was produced. Source and log preserved in results/kaggle_cho_detector_v1 with SHA256.json. Version 2 preparation adds pinned imagecodecs 2024.12.30 (without changing other dependencies), codec roundtrip and actual raw/annotation read preflight. Training, split, thresholds and scoring remain unchanged.

## 2026-09-22 — Solver equivalence and detector-training repairs

Preserved solver reference in `references/joint_solver_before_optimization`.
Sixteen randomized configurations compare exact ordered assignments and constraint
matrices against it. Sparse CSC construction and distance-gated pair enumeration
preserve the objective and hypothesis order. Synthetic 160-cell benchmark:
median 0.3920246 s before, 0.0220875 s after (17.75x); matrix storage 819200 vs
8964 bytes. Evidence: `results/joint_solver_optimization.json`. This measures
synthetic efficiency, not biological accuracy or full-run speed.

Sparse detector targets no longer label dark unannotated voxels as negative or
accept teacher predictions as confirmed positives. Explicit background masks
are required. Fixed actual model 5-D output handling, verified backward pass,
and removed held-out-recall epoch selection from the sparse trainer. Full suite:
112 passed; baseline verifier: all 61 files unchanged.

CHO feasibility Version 1 dispatched to private Kaggle notebook
`dushi2/biohub-cho-detector-feasibility`. Five existing CTC tests passed before
packaging 22 hashed source files. Frozen protocol is under
`notebooks/kaggle_cho_detector`: sequence 01 training/validation, sequence 02
held-out evaluation once, max 30 epochs x 30 steps. Human tracking-marker
centroids supervise this experiment, not sparse segmentation-mask background.
No model result yet; no production detector replacement. CHO is not Biohub.


## 2026-09-22 — Frozen expansion completed; no division gain

Eight volumes, 24 unique arm records. Trackastra candidate composite delta
+0.00039128, but three volumes regressed, nine correct edges were removed and
15 added. Of five GT divisions, zero recovered; one confirmed false division.
1,930 edits increased unknown division predictions from 1,787 to 3,716.
Geometry arm exactly tied classical. Source hashes verified. No promotion or
automatic full-data run; the intended division improvement was not reproduced.
Artifacts: `results/kaggle_joint_expansion_v1/`.

## 2026-09-22 — Frozen additional-volume joint evaluation launched

Hypothesis: the fixed joint candidate's four-volume gain extends to eight
additional volumes without unacceptable correct-link removal or unknown
division growth. Three arms, fixed parameters and source hashes are specified
in `JOINT_EXPANSION_PROTOCOL.md` and the frozen manifest. Kaggle Version 1 of
`dushi2/biohub-joint-expansion` is RUNNING. This is exploratory benchmark reuse,
not independent biological validation; no outcome is yet available.

Evidence-based engineering log for the classical competition pipeline.

## Real detected-data learned-linker evaluation — completed 2026-09-21

**Hypothesis:** an image-aware parent cost trained on real DoG detections improves
the current rich Hungarian tracker. **Baseline:** production classical Hungarian
linking. **Candidate:** the same detections with the 25-feature neural edge cost.
The split was two-fold leave-one-embryo-out across all 199 logical samples (71
`44b6`, 128 `6bba`); only two embryos are independent biological groups.

The pre-registered decision rule required the embryo-level paired 95% CI lower
bound for composite-score delta to exceed zero. The candidate failed on both
embryos. Composite mean changed 0.726048 -> 0.703995; paired sample delta was
-0.022054 [-0.024782, -0.019467]. Edge precision delta was -0.007480
[-0.009013, -0.006010], edge recall delta -0.057868
[-0.065031, -0.051093], and edge-F1 delta -0.045807
[-0.051203, -0.040685]. Candidate division precision and recall were both zero.
Detection recall delta was exactly zero, which confirms that the linker did not
alter detections. **Decision: reject; do not deploy.**

Artifacts and executed source are preserved in
`results/kaggle_detected_linker_full/` with SHA-256 manifest. A post-run audit
confirmed `BCEWithLogitsLoss(pos_weight=negative/positive)` was already used and
unknown sparse-label candidates were excluded. It also found a real
missed-parent false-negative leak in training labels; that bug is fixed locally
and covered by regression. Joint parent/daughter-pair optimization is being
validated as a separate candidate with this evaluator's scoring, split, and
decision rule unchanged.

The full 199-volume joint-hypothesis validation was launched privately as
`dushi2/biohub-joint-division-linker` Version 1. It remains opt-in. Continuation
and division choices compete in one constrained objective; one division variable
claims its parent and both daughters and incorporates both edge costs, midpoint
consistency, daughter separation, and symmetry. Results are pending.

## Trackastra zero-shot evaluation — completed 2026-09-21

**Hypothesis:** the externally pretrained 3D Trackastra `ctc` transformer improves
linking and mitosis recovery over the current Hungarian tracker when both receive
the same real DoG detections. DoG centroids were converted to deterministic,
image-supported watershed instances because Trackastra requires raw images and
instance masks.

The private GPU run evaluated every paired volume: 71 `44b6` and 128 `6bba`.
Trackastra generated candidate edges independently; Hungarian was only the control.
The run completed in 14,182.07 seconds. Two earlier dependency setup failures did
not execute the model and are preserved separately as integration failures.

| Metric | Hungarian mean | Trackastra mean | Paired sample delta (95% CI) |
|---|---:|---:|---:|
| Composite | 0.726051 | 0.684527 | -0.041524 [-0.050396, -0.032925] |
| Edge precision | 0.994568 | 0.995095 | +0.000527 [-0.000913, +0.002094] |
| Edge recall | 0.645024 | 0.536326 | -0.108698 [-0.126634, -0.091471] |
| Edge F1 | 0.774234 | 0.674741 | -0.099493 [-0.117755, -0.082896] |
| Division precision | 0.030151 | 0.041625 | +0.011474 [-0.020352, +0.042462] |
| Division recall | 0.009631 | 0.062395 | +0.052764 [+0.024696, +0.084590] |

The embryo-level composite delta was -0.049129 with 95% CI
[-0.075678, -0.022579]. Trackastra recovered more evaluable divisions but dropped
far more correct continuation edges. **Decision: reject zero-shot Trackastra and
keep classical Hungarian as the production default.** This does not test a Biohub
fine-tuned Trackastra model. Artifacts and hashes are in
`results/kaggle_trackastra_full_v2/`.

### Trackastra division-only follow-up — running

Because zero-shot Trackastra's only clear improvement was division recall, a
pre-specified hybrid candidate is running on all 199 volumes. It preserves the
classical continuation graph exactly. Trackastra may add one second daughter only
when its other daughter equals the classical continuation, the new target has no
incoming edge, both daughters occur in the next frame, and all physical division
gates pass. This isolates the supported signal and prevents the measured
Trackastra continuation-edge regression from entering the candidate graph.

## Exact-observation tracking backend check — 2026-09-20

**Hypothesis:** the experimental multi-frame continuation optimizer preserves or
improves graph accuracy across controlled motion, density, entry/exit, division,
and low-SNR scenarios. **Baseline:** framewise sparse Hungarian linking.
**Candidate:** sequence-wide maximum-weight continuation/gap matching with the same
division gates. Both consume the identical exact synthetic coordinates, excluding
detector error.

Across all nine pre-defined scenarios, both backends achieved mean edge F1 1.0.
Measured total runtime in this cold Python process was 30.44 ms for framewise and
491.75 ms for global; the global measurement includes its first NetworkX import.
The result is a tie and provides no evidence to adopt global tracking as the
default. It does verify graph behavior before real-data evaluation. Daughter pairs
are still inferred after the global continuation objective, so Phase H's requested
joint mitosis objective remains unfinished. Raw results are in
`results/tracking_backend_comparison.json`.

## Phase A corrected-evaluator rerun — 2026-09-19

The previous DoG and border conclusions were treated as historical because the old
proxy matcher could match one prediction to more than one ground-truth cell and did
not distinguish sparse-label unknowns from false positives. The rerun uses the
production `link_frames` path, one-to-one Hungarian node matching, sparse-GT edge
accounting, GT-confirmed division precision/recall, and paired bootstrap confidence
intervals on the per-sample score delta.

Protocol: three real `44b6` training samples, first 40 frames for the DoG comparison
and first 8 frames for the border sweep. Score weights were 0.45 node recall,
0.45 evaluable edge F1, and 0.10 evaluable division F1. The sample is too small for
a production decision; a candidate is accepted only when the paired 95% bootstrap
delta interval is strictly above zero.

| Experiment | Baseline mean | Candidate mean | Paired delta (95% CI) | Decision |
|---|---:|---:|---:|---|
| V10 DoG vs V4 classical | 0.172500 | 0.346875 | +0.174375 [0.000000, 0.320625] | Reject: interval includes zero |
| Border `q=0.20` vs `q=0.80` | 0.262500 | 0.431250 | +0.168750 [0.000000, 0.450000] | Reject: interval includes zero |
| Border `q=0.00` vs `q=0.80` | 0.262500 | 0.431250 | +0.168750 [0.000000, 0.450000] | Reject: interval includes zero |
| Border `q=0.50` vs `q=0.80` | 0.262500 | 0.281250 | +0.018750 [0.000000, 0.056250] | Reject: interval includes zero |

The corrected runs show promising recall gains, especially for DoG, but do not
support enabling either change. Edge and division F1 were zero in these short
windows because no predicted edge had two endpoints that could both be mapped to
the sparse annotations; thousands of predicted edges and all predicted divisions
were correctly reported as unknown rather than false positives or evidence of
quality. This makes the composite score a recall-only signal in this rerun and is
the main limit on the conclusion.

Raw summaries are written to `results/dog_ab_test_phase_a.csv` and
`results/border_ab_test_phase_a.csv`; per-sample measurements are in the matching
`*_details.csv` files. The evaluator regression suite covers the four required
failure modes and the full suite passes (30 tests after the later roadmap work).

## Leaderboard evidence

| Notebook version | Public score | Pipeline basis |
|------------------|-------------|----------------|
| V2 | 0.607 | Early v1.x calibration |
| **V4** | **0.659** | **v1.4 baseline** |
| V5 | 0.648 | v1.5 — **regression** |
| **V7** | **0.659** | **v1.6 v4 preset — regression recovered** ✅ |

**Rule:** Do not add features without A/B evidence. V7 confirms Sprint 2 hypothesis: revert v1.5 defaults, keep v4 behavior.

| V8–V9 one-knob tuning | 0.659 | No gain — classical knobs exhausted |

**Next:** V10 DoG band-pass detection (opt-in via `competition_v10_dog_preset()`).

---

## V10 — DoG band-pass detection

Public notebooks report ~0.85 with DoG preprocessing. V10 adds an **opt-in** detection front-end while keeping V7 tracking/division settings.

```python
CFG = Config().competition_v10_dog_preset()
# detector=peaks_dog, use_dog_bandpass=True
# dog_sigma_small_um=1.0, dog_sigma_large_um=2.8
```

Notebook: set `USE_V10_DOG = True` in section 1.

Optional single-knob DoG tuning:

```python
V10_DOG_OVERRIDES = {"dog_sigma_large_um": 3.2}  # one at a time
```

**V7 remains default** (`USE_V10_DOG = False`) — do not break the validated 0.659 baseline.

---

## Phase 2 — V4 vs V5 code comparison

V5 used repository commit `6682370` (pipeline v1.5). V4 aligned with v1.4 (`8c19f55`) plus notebook-specific `thresh_rel` from density calibration.

### Changes in v1.5 (V5) not present in v1.4 (V4)

| Change | File | Expected effect | Likely LB impact |
|--------|------|-----------------|------------------|
| **Gap closing** (`t-2 → t`, 15 µm gate) | `tracking.py`, `submission.py` | Recover missed links | **Negative** — long-range false edges, wrong topology, extra division candidates via `link_frames` |
| **Soft orphan pruning** | `tracking.py` | Keep nodes near neighbors without edges | **Negative** — extra unmatched nodes hurt node precision / graph consistency |
| **Division symmetry penalty** (`weight=0.35`) | `tracking.py` | Penalize asymmetric daughters | **Mixed/negative** — may skip true mitosis or reorder daughter assignment |
| **Adaptive threshold retry** (bug fix) | `detection.py` | Tighten threshold when over count cap | **Mixed** — can reduce FPs but also drop true cells vs V4 (where retry never ran) |
| Hyperparameter grid expansion | `tuning.py` | Offline only if disabled | Neutral if `run_hyperparameter_search=False` |

### Regression hypothesis (ranked)

1. **Gap closing** — highest risk: adds edges outside the v4 Hungarian window, pollutes edge Jaccard and division graph.
2. **Soft pruning** — retains detections that v4 would drop; increases false nodes.
3. **Adaptive retry** — changes per-frame counts vs frozen v4 behavior.
4. **Division symmetry** — smaller effect; may reduce division recall.

**Conclusion:** V6 defaults **disable all four** via `competition_v4_preset()`.

---

## Phase 3 — Error analysis (metric mapping)

| Issue | Affects |
|-------|---------|
| False positive detections | Node precision, spurious edges |
| False negative detections | Node recall, broken tracks |
| False edges (gap close, long gate) | Edge Jaccard |
| Missing edges | Edge recall |
| False divisions | Division metric precision |
| Missed divisions | Division metric recall |
| Isolated nodes after prune | Node recall if true cells unlinked |

---

## Phase 4 — Improvement ranking (post-regression)

| Improvement | Expected LB gain | Difficulty | Runtime | Risk |
|-------------|------------------|------------|---------|------|
| **Revert to v4 preset (V6)** | Recover ~0.659 baseline | Low | Same | Low |
| **Hyperparameter search on train** (thresh, link, div) | +0.01–0.03 | Medium | Offline | Low |
| **Single-knob thresh_rel sweep** after calibration | +0.005–0.015 | Low | Low | Low |
| Re-enable gap close with tighter gate (13 µm) | Unknown | Low | +5% | **High** |
| Re-enable soft prune | Unknown | Low | Same | **High** |
| Learned detector | Large | High | High | Medium |

**Implement now:** v4 preset only. Tune via `run_hyperparameter_search.py`, one knob per submission.

---

## Phase 5–7 — What we did NOT implement

No new detection/tracking/division algorithms in Sprint 2. Feature adds caused V5 regression.

---

## Phase 8 — Hyperparameter search

```bash
python scripts/run_hyperparameter_search.py --train-dir /path/to/train
```

Output: `results/hyperparameter_search.csv`

Apply **one** parameter change per submission attempt. Compare against V4 baseline.

### Single-knob sweep (recommended after V7)

```bash
python scripts/run_single_knob_sweep.py --train-dir /path/to/train --samples 5 --frames 6
```

Output: `results/single_knob_sweep.csv`

The Kaggle notebook now runs the same low-risk sweep by default and can apply
the best proxy candidate automatically. Manual overrides are guarded so only
one parameter can change. Explicit `thresh_rel` overrides disable density
calibration; otherwise `build_submission()` would recalibrate the threshold
back to the train-density choice (for example 0.26).

If `mean_score`, `mean_recall`, and `mean_edge_proxy` are all `NaN`, GEFF
labels were not found/readable in the runtime. Treat that table as
non-diagnostic; do not use it to pick a winner. In that case, keep V7 as the
known baseline and only spend leaderboard submissions on one clearly defined
manual candidate at a time.

---

## Phase 9 — Local diagnostics

```bash
# V4 preset (default)
python scripts/run_diagnostics.py --synthetic --preset v4

# Compare against V5 settings locally
python scripts/run_diagnostics.py --synthetic --preset v5

# With train data
python scripts/run_diagnostics.py --train-dir /path/to/train --preset v4
```

Plots in `results/diagnostics/`: detections/frame, track lengths, link distances, divisions, node density, scores.

### Synthetic A/B (local)

| Metric | v4 preset | v5 preset | Interpretation |
|--------|-----------|-----------|----------------|
| Nodes | 109 | 163 | Soft prune + gap close retain extra graph nodes |
| Edges | 95 | 149 | Gap closing adds long-range false links (+57%) |
| Divisions | 5 | 5 | Same on synthetic; real data may differ |
| Detections/frame | ~13.6 | ~13.6 | Adaptive retry did not change count on synthetic |

---

## V6 submission workflow

```bash
python scripts/build_notebook.py
```

Notebook applies `CFG.competition_v4_preset()` automatically.

`build_submission()` applies the same preset when `use_competition_preset=True` (default).

### Opt-in v1.5 experiments (offline A/B only)

```python
CFG = Config(use_competition_preset=False).copy_with(
    gap_close_enabled=True,  # test one flag at a time
)
```

---

## Track A vs Track B

| Track | Focus |
|-------|--------|
| **A — Competition** | `submission.py`, notebook, `EXPERIMENTS.md`, preset tuning |
| **B — Research** | Streamlit app, paper, `main` docs — no experimental flags in defaults |

---

## Runtime comparison

| Setting | Relative runtime |
|---------|------------------|
| V4 preset | 1.0× (baseline) |
| V5 (+ gap close) | ~1.05–1.10× |

---

## Risk assessment (V6)

| Risk | Mitigation |
|------|------------|
| V6 matches V4 but not beat it | Run hyperparameter search; single-knob commits |
| Overfitting train proxy | Validate with public LB one change at a time |
| Re-enabling v1.5 flags | Only via explicit opt-in, never default |

---

## Expected leaderboard impact

| Action | Estimate |
|--------|----------|
| V6 = V4 preset | **~0.659** (recover V5 regression) |
| V7 submitted (confirmed) | **0.659** ✅ matches V4 |
| + tuned thresh_rel / link gate | +0.005–0.02 (next experiment) |

Validate with **one submission** (V6). Do not batch multiple changes.
## 2026-09-20 — Expanded-roadmap correctness audit

**Hypothesis.** The earlier A–L summary may hide unfinished requirements in the
new Phase 0–34 roadmap.

**Verified finding.** `validation.track_length_distribution` was an out-degree
proxy, so the Phase 0/6 track-length claim was false. It now traverses maximal
non-branching graph segments and splits at divisions. Sparse evaluation now
reports `edge_pred_evaluable`, `edge_pred_unknown`, and `edge_gt_evaluable`.
Raw detector scores and intensities are preserved on observations; predicted
divisions are labeled `predicted_candidate`; graph validation now checks the
allowed gap and daughter timing.

**Synthetic coverage.** Added acceleration, missed-observation, temporary
occlusion, high-noise, and anisotropic-Z exact-truth scenarios plus primitive
physical NMS, scale conversion, and centroid tests.

**Validation.** Full local suite: **64 passed**. Untouched baseline verifier:
**61 original files passed**. These are local correctness checks, not evidence
of improved real-data accuracy or a state-of-the-art result.

**Decision.** Keep the learned detector opt-in. Do not promote any candidate
until the active all-199 Kaggle run and held-out real-data comparisons finish.

## 2026-09-21 — Division-only Trackastra hybrid audit

**Hypothesis.** Trackastra's measured division-recall gain can be added without
losing any continuation edges from the classical graph.

**Version 1 gate.** The running all-199 notebook accepts a proposal only when
Trackastra selects exactly two children, the classical parent has exactly one
child and it is one of those two, the second child has no incoming edge, all
three nodes exist at adjacent times, and parent/daughter distance, sister
separation, and daughter-midpoint residual pass the existing physical gates.
It has no model-score threshold and no post-division persistence requirement.

**Local audit and preparation.** The merge now refuses duplicate baseline edges,
keeps the complete ordered baseline edge list as the candidate prefix, records
stable hashes, and raises if those hashes differ. Regression tests reject a far
false daughter and show that accepting a valid split cannot disturb unrelated
continuations. Trackastra 0.5.3 exposes normalized selected-edge weights, so the
local evaluator now supports an opt-in threshold that both daughter edges must
pass. It also supports requiring each daughter to continue for a specified number
of classical transitions. These options were added after Version 1 launched and
are not part of its result.

**Validation.** Focused adapter suite: 9 passed. Full local suite: 75 passed.
Untouched baseline verifier: 61 original files passed. The active hybrid was
healthy at 11/199; its first 11 scores tied the control exactly. No decision is
made until all 199 samples and embryo-level paired confidence intervals finish.

### Final Version 1 result

The run completed all 199 samples in 14,545.32 seconds and was **rejected**.
Trackastra produced 129,476 split proposals; 101,811 failed the graph-conflict
gate and 27,665 failed geometry. Zero passed, so zero daughter edges were added.
Every candidate metric equalled the control: composite 0.726051, edge precision
0.994568, edge recall 0.645024, edge F1 0.774234, division precision 0.030151,
and division recall 0.009631. Every paired sample and embryo delta interval was
exactly `[0.0, 0.0, 0.0]`. The adoption rule was not met.

The executed source began from a de-duplicated baseline edge list and appended
only accepted daughters. It did not contain the post-launch exact-prefix hash,
score threshold, or persistence gate. In the observed run, zero acceptance makes
the candidate and baseline edge sets identical on every sample. A stricter score
or persistence setting cannot create an eligible proposal, so a duplicate run is
not justified. `TRACKASTRA_DIVISION_FOLLOWUP_PROTOCOL.md` freezes a preflight-first
protocol for any redesigned gate.

## 2026-09-21 — Joint hypothesis full-run timeout

**Outcome.** Kaggle canceled Version 1 after 43,200.6 seconds (exit code 137).
Detection completed for all 199 volumes and the first fold trained for 15 epochs.
Only 54 of 71 held-out `44b6` samples completed both arms; the `6bba` fold never
ran. The completed prefix had mean composite delta +0.026499, edge precision
delta +0.005119, edge recall delta +0.080539, and edge F1 delta +0.058887.
Division precision and recall were zero in both arms.

These numbers are a time-ordered partial prefix with no held-out second embryo,
no paired confidence interval, and no pre-registered verdict. They must not be
used to adopt the model. The fold checkpoint, status, executed source, notebook,
log, and SHA-256 manifest are preserved in
`results/kaggle_joint_division_linker_v1/`.

**Artifact and repository verification.** The completed hybrid archive contains
52 hashed files; its manifest hash is
`a931a80b1f6c6307c263527e96152882a184299805c09218a4dc093f2dc9640d`. The timed-out
joint archive contains 47 hashed files; its manifest hash is
`ee7ba811bea70b3aaf23e74a05cc35af14015e71b3ee91bf2b97771d9f39f5ac`. After the
records were updated, the full local suite passed 81 tests and the untouched
baseline verifier passed all 61 original files.

**Cheaper classical control.** The existing one-knob sweep was rerun on the three
locally available `44b6` volumes for six frames. Parent gates 11/12/13 um all
produced 5.33 candidate divisions per sample; changing the sister gate from 7.5
to 7.0 or 8.0 changed the count from 5.33 to 3.00 or 7.67. All sparse-label
recall, edge, and composite values were zero in this short window, so every delta
was zero and no gate can be selected from this smoke test. The raw table is
`results/division_gate_local_smoke.csv`. A fair all-199 paired evaluation remains
necessary before crediting Trackastra or changing the cheaper classical gates.

## 2026-09-21 — Comparative-development foundation

**Goal.** Add a falsifiable normal-versus-condition analysis layer without claiming
that a single movie identifies a genetic disorder.

**Implementation.** Canonical-cell event tables can now be compared for missing,
unexpected, or time-shifted events and the first measured divergence. Separate
functions measure joint daughter orientation and asymmetry, build robust division-
timing atlases, classify track ends conservatively, and calculate local crowding.
Run-local numeric IDs are rejected as cross-embryo identities unless the caller
provides canonical names or a validated mapping.

**Evidence boundary.** A disappearing track remains an unresolved termination;
even a known-lineage match is called an annotated death candidate until confirmed
by suitable imaging or biological annotation. No mutant, stress, disease, or
diagnostic result has been produced. Real control and perturbation replicates have
not yet been integrated.

**Validation.** Six comparative-development unit tests and the existing scientific
export tests pass. The complete local suite is **81 passed**, and the untouched
baseline verifier still passes all 61 original files. An end-to-end synthetic
example under `results/comparative_development_demo/` reports one timing shift,
one missing event, one unexpected event, and a first divergence at 10 seconds.


## 2026-09-21 — Diagnostic-first follow-up launched

Private notebook `dushi2/biohub-division-diagnostics` Version 1 was pushed and
Kaggle API confirmed RUNNING. It selects the lexicographically first two paired
volumes per embryo before metrics (four total), requires CUDA, and makes no
training, tuning, adoption, or full-dataset-completion claim. The mounted dataset
stays on Kaggle. Sources and SHA-256 hashes are preserved in
`notebooks/kaggle_division_diagnostics/`.

Important correction to the earlier conversational diagnosis: 101,811 is an
aggregate conflict count, NOT a verified count of already-owned second daughters.
The executed gate also combines wrong baseline child count, missing anchor,
missing node, and timing failures. No single root cause is proven yet.
The new diagnostic separates those categories, saves up to 20 deterministic
examples per conflict category per volume, and preserves compressed raw graphs.

The previous joint notebook metadata has enable_gpu=false. Its learned cost
loader and inference explicitly use CPU, and scipy.optimize.milp is a CPU
assignment step. Toggling the notebook GPU alone will not accelerate those paths.
A bottleneck profile is needed before investing in a GPU rewrite or assuming it
will solve the timeout.

Division coverage reports raw distinct daughter detections, evaluator-matched
daughters, and matched parent-plus-daughters, with exact event counts and the
existing physical matching gate. A merged single detection cannot count as two
daughters. Zero annotated divisions gives null coverage, not an invented rate.
This is a sparse annotated-event coverage diagnostic, not exhaustive cell recall.

Validation: 85 pytest tests passed; untouched baseline passed all 61 file checks.
Evaluator scoring/splits/rule and production linking were not modified.
Next: preserve outputs, inspect actual conflicts and coverage, then design the
smallest supported fix. If pilot outcomes select thresholds using both embryos,
the same embryos cannot subsequently be called untouched validation. Use only
the opposite embryo for each fold's parameter selection; prior repeated use of
this two-embryo benchmark must still be disclosed as exploratory reuse.


## 2026-09-21 — Four-volume diagnostic completed (Version 351638414)

Kaggle status COMPLETE; all four per-volume files and compressed graphs, logs,
notebook, metadata and SHA-256 manifest preserved in
`results/kaggle_division_diagnostics_v1`. All ten entries in the run's output
hash manifest verified. No model was trained or promoted.

Proposal totals: 1,218 second daughters already owned, 240 missing continuation
anchors, 203 parents with baseline child count other than one, 388 geometry
rejections (2,049 total). All 227 saved conflict examples have unknown sparse
support; none can be called a confirmed true or false division.

The first three selected volumes have no GT divisions. `6bba_05db0fb1` has three
adjacent binary GT divisions; two have distinct detected daughters and matched
parent/daughters (2/3), one lacks daughter coverage. This is three events in one
embryo, not a dataset-wide estimate. Inspection of saved graphs shows Trackastra
proposes neither complete GT-supported split for the two fully detected events:
GT parent 25000381 has no Trackastra outgoing edge; GT parent 63001217 has only
one outgoing daughter edge. Hungarian has one correct daughter for each and
assigns the other to a competing parent. Thus conflict reassignment alone cannot
recover these events if it relies exclusively on selected Trackastra splits.

Next diagnostic must inventory annotation-supported divisions first and select
fixed division-bearing samples per embryo before model outcomes. Distinguish
Trackastra's unselected candidate edge weights from its selected greedy graph;
inspect proposal recall before redesigning merge rules. Do not rerun all 199
or claim improved tracking from this diagnostic. The lexicographic sample choice
was insufficient for a cross-embryo division ceiling and is recorded as such.


## 2026-09-21 — Division-bearing candidate diagnostic launched

Private notebook `dushi2/biohub-division-bearing-diagnostics` Version 1 was
successfully pushed. It inventories adjacent binary GT divisions in mounted
annotations, then selects the lexicographically first two eligible volumes per
embryo before model inference. It fails if groups lack sufficient eligible
volumes or selected events lie outside processed frames. This enriched subset
is diagnostic, not an unbiased full-dataset evaluation or threshold-tuning set.

Instrumentation targets verified Trackastra 0.5.3 wheel source. It captures
model weights and the default candidate graph (max_distance=256 in model
coordinates, max_neighbors=10, delta_t=1), then calls the original selector.
For GT daughter edges it separates missed detection, absent weight, candidate
pruning, score below default 0.5, greedy conflict, and selected edge. Raw weights
are scores, not calibrated probabilities. Full weights and graph snapshots are
saved for offline inspection. The original production/evaluation code is unchanged.

Validation: 87 tests passed; baseline verifier passed 61 files. Source hashes
and launch metadata: `notebooks/kaggle_division_bearing_diagnostics/`.
No accuracy improvement, completed training, or full-data verdict is claimed.

## 2026-09-22 — Cached joint evaluation Version 1 failed before scoring

Version 1 lacked the optional GEFF/Zarr reader dependencies. Annotation loading
returned no tables, causing `AttributeError: 'NoneType' object has no attribute
't'` before any arm was scored. Treat this solely as an integration failure.
Version 2 uses pinned offline readers and adds explicit missing-table errors;
experiment samples, parameters, arms, and evaluation logic remain unchanged.

## 2026-09-22 — Cached joint evaluation Version 2

The corrected run completed. On four previously inspected division-enriched
volumes, joint Trackastra improved mean composite by +0.014752, edge recall by
+0.005483 and edge precision by +0.002533. It recovered one of six confirmed
divisions with one evaluable false positive. All four sample composite deltas
were positive. However, 2,009 edits increased unknown division candidates from
1,281 to 3,288; sparse labels cannot judge them. Geometry-only made no edits.
Decision: retain as an experimental candidate only. No adoption, full-data win,
confidence interval or model-training claim follows from this reused subset.


## 2026-09-21 — Division-bearing diagnostic completed, Version 351643668

Kaggle COMPLETE. Preserved outputs, raw model-weight/graph snapshots, executed
source/notebook, metadata, logs and SHA-256 manifest under
`results/kaggle_division_bearing_diagnostics_v1`. All 10 upstream output hashes
verified. Annotation inventory covers 199 volumes, 87 with adjacent binary
annotated divisions and 151 such events in total (not necessarily unique
biological events across volumes).

Four annotation-selected volumes contain 6 GT divisions: 2 in 44b6 and 4 in
6bba. Matched parent and both daughters: 2/2 and 3/4 respectively, 5/6 total.
Of 12 GT daughter edges: 5 selected, 6 below the default greedy score threshold
0.5, and 1 missing detection. No observed GT edge was lost to absent model
weights, candidate pruning, or greedy conflict. Only one full GT split had both
edges selected. Below-threshold weights were 0.279275, 0.084878, 0.131177,
0.057243, 0.050240, and 0.462717.

This small enriched diagnostic supports inspecting model scoring/domain mismatch
and joint daughter evidence before blaming greedy conflicts for GT misses.
It does NOT justify globally lowering the threshold, predict precision after
lowering it, or establish dataset-wide division recall. The competing-edge
reassignment hypothesis alone cannot fix low-scored true edges. Next candidate
should use raw scores plus joint geometry and compare competing hypotheses;
parameter selection must use only the opposite embryo in each fold, with a
cheap classical control and unchanged final decision rule. Missing detections
remain a separate limitation. No model was trained or adopted by this run.


## 2026-09-22 — Experimental joint scoring and geometry ablation

Implemented opt-in `src/biohub/joint_division.py`: preserve the parent anchor,
compare second-edge score with the competing incoming score, charge geometry and
termination penalties, require two transitions of daughter persistence, validate
the entire graph after every edit and assert unrelated ordered edges unchanged.
Displaced parents are explicitly unresolved terminations, not invented relinks.
This is an experimental heuristic, not a trained joint probability model.

Original 7.5 um sister gate: 358 edits, zero of six annotated divisions recovered.
All five fully detected true pairs exceed that gate (7.557 to 12.488 um).
Exploratory sister gate 24 um (triangle bound from parent gate 12 um): 2,009 edits,
one of six divisions recovered. The independent geometry-score control made
zero edits. Reports: `results/joint_division_cached_pilot.json`,
`results/joint_division_gt_gate_audit.json`, `results/joint_division_gate_ablation.json`.
These previously inspected enriched samples cannot provide a clean held-out win.
Do not promote the candidate or infer precision from event recovery.

92 tests passed and baseline verification passed all 61 original files.
Launched private CPU notebook `dushi2/biohub-cached-joint-evaluation` Version 1
using saved graph outputs plus mounted annotation tables only, without image
inference or model training. It runs the unchanged sparse evaluator on classical,
joint Trackastra and independent geometry arms across the same four volumes.
Scoring will establish actual continuation damage before any expensive full run.
Source hashes and metadata: `notebooks/kaggle_cached_joint/`.
`n## 2026-11-13 — External repository reuse review`nReviewed PyUAT, LapTrack, Ultrack and two Biohub-specific repositories; see docs/GITHUB_REUSE_REVIEW.md. No external code or model promoted. A like-for-like, embryo-disjoint benchmark and license/dependency checks are required before adoption. The image-label supervision blocker remains.


## 2026-11-19 — larger real pair and failed domain-transfer check

Prepared 30 native-grid nucleus frames per recording: uncompressed wild-type EmbSAM Emb8 (612,434,048 bytes, 30x56x512x712), uncompressed CMap wee-1.1 MT2 (1,028,014,208 bytes, 30x94x512x712). Source manifests and hashes: data/real_control_variant_large/PROVENANCE.json. The control source is figshare 29064530; mutant source is figshare 27872238. Both embryos are uncompressed, but biological stage, frame interval and voxel grids are NOT matched. No genetic effect can be inferred from raw count differences.

Actual fixed-checkpoint native-grid early inference on Emb8 returned 66,60,60,70,58,64,62,63 centres in its first eight frames. The authors' CellTracing CSV lists two nuclei per frame there. Stopped this run; no complete-pair score was computed. A one-frame fixed metadata-driven physical resampling pilot produced 88 predictions against two reference nuclei (zero matches within three target voxels under an unverified one-based Z assumption). That pilot was rejected; source scale alone did not resolve transfer failure. Evidence: results/large_pair_review_20261119/native_grid_failure.json and scale_pilot.json. Current checkpoint is not validated for these acquisitions. App blocks this exact failed model/control hash combination instead of presenting it as a valid lineage comparison.

UI now supports memory-mapped local .npy paths for large files and six comparison views: images, division proposals, continuity, observed track durations, aligned differences (gated), family trees. Unaligned descriptive graphs warn about acquisition confounding; individual cell homology and genetic diagnosis remain unvalidated.


## 2026-11-19 — experimental pretrained submission review
Official Cellpose 3.1.1.1 nucleus and size weights were installed with provenance and BSD attribution. No new training was performed. Tested full 3D Cellpose, 2D nucleus projections, denoising, disabling flow filtering, and cyto3. Full 3D, denoising and cyto3 did not solve the variant errors and were not adopted. Selected the standard nucleus projection pipeline as an explicitly experimental review option, without claiming validated genetic analysis. Original Biohub evaluator and acceptance rule remain unchanged.

Five-frame control: 2,2,2,2,2 predictions versus 2,2,2,2,2 published counts. Five-frame variant: 1,1,4,4,5 versus 2,3,4,4,4. Control tracking produced eight links; variant produced ten. 30-frame inference also completed for both files; exact outputs and count disagreements are under results/submission_large_demo_20261119. All claims are count checks only; reference coordinate registration and biological identity matching remain unvalidated.

Model adapter retains real raw Cellpose logits, estimates Z from source intensities, and clearly exposes 2D projection/overlap/mitosis limitations. UI uses source spacing for linking on recognized files and suppresses scientific time/speed exports for unknown time calibration. End-to-end AppTest passed Volume, Pipeline, Detection, Lineage, Exports and Compare Lineages using actual saved outputs. Control/Variant switching did not rerun inference or cross-bind results.


## 2026-11-19 — division-bearing clips
Prepared native five-frame clips using published lineage transitions: control source frames 50,60,70,80,90 (reference counts 2,2,3,3,4), variant source frames 8-12 (reference counts 4,4,6,6,6). No links were inserted from annotations. Cellpose projection + classical tracking produced control counts 1,1,1,1,3 with zero proposed divisions; variant counts 5,4,5,6,6 with three proposed division parents (node 9 at clip frame 1; nodes 11 and 14 at clip frame 2). These are unvalidated candidates, not three confirmed biological divisions. Earlier control frames 1-5 precede the first published AB division at 63->64. Outputs: results/division_clip_check_20261119.


## 2026-11-19: EmbSAM control/RNAi candidate checked
Downloaded Emb20 wild-type and Emb21 nop-1 RNAi TIFF members using bounded HTTP ranges; ZIP CRC and full pixel decode passed. Shapes are (165,2,280,281) and (67,2,460,376), both encoded ZCYX with no separate T/Z axes. Published table lists 66 Emb21 timepoints; actual file has 67 planes. Visual review shows different developmental stages. Public Cell Tracing Result and Dataset nucloc archives contain Emb1-9 only, not this pair. Rejected for current 3D lineage workflow; no fabricated depth, annotations, inference, or training. Files and SHA256 in data/embsam_control_rnai_candidate/INSPECTION.json.


## 2026-11-19: local Biohub recordings in comparison workspace
Converted all 100 frames of 44b6_0113de3b and 44b6_0b24845f from existing Desktop Zarr data to memory-mapped NPY, 838,860,928 bytes each. Every pixel verified against source; SHA256 and calibration recorded in data/local_biohub_pair/PROVENANCE.json. Added comparison preset and original classical detector option; unsupported C. elegans engines are blocked for these known files. Same prefix and unverified genotypes are explicitly disclosed; no control/mutant labels established. No new training or paired inference result claimed. 139 tests pass; baseline verifier PASS 61 files.


## 2026-11-19: submission reliability fixes and local zebrafish verification
See SUBMISSION_STATUS.md for item-by-item audit disposition. Fixed input/result invalidation, empty schemas, calibration/provenance, independent timing verification, atlas replication, run-specific exports, source hashes and sparse GEFF checks. Windows reference matching preserves uint64 IDs; global topology preview filters edges to its bounded node subset. No evaluator threshold/split/decision-rule change and no Kaggle training or cloud action. Local complete 100-frame A/B inference: A 18805 observations/17912 edges, 42/52 annotations matched; B 10708/8397, 3/51 matched. All 260 division proposals remain unverified; neither local reference contains an annotated division parent. B remains a measured failure, not an improvement. Detailed artifacts/checksums: results/submission_biohub_20261119. No validated genetic diagnosis, independent cross-embryo association or full-detection claim. Baseline remains separate.
