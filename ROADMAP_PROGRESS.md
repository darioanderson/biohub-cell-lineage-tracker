# Roadmap progress — 2026-09-20

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

Update 2026-09-22 (detector feasibility): fixed sparse detector output-rank
contract and removed inferred dark-voxel background supervision. Biohub training
now requires explicit background masks; it remains blocked without those labels.
Joint MILP uses equivalent sparse constraints and physically gated daughter-pair
enumeration. 112 tests passed and all 61 baseline files verified unchanged.
Private CHO detector Version 1 dispatched with sequence-disjoint held-out testing;
model quality and Biohub integration remain unverified. Remaining roadmap phases
are not marked complete by these implementation changes.


Update 2026-09-22: frozen eight-volume expansion finished. Zero of five divisions
recovered, one false division, nine correct links removed versus 15 added.
Candidate remains unapproved; phase H/11 remains partial. See latest result in
`KAGGLE_PROGRESS.md`; older RUNNING notes below are historical.

Latest (2026-09-22): eight-volume frozen joint expansion is RUNNING on Kaggle,
with per-edit sparse-support auditing. 94 local tests and 61 baseline file
checks passed. Phase H/11 remains experimental pending measured results;
see `JOINT_EXPANSION_PROTOCOL.md`.

The expanded Phase 0–47 audit is maintained in `FULL_ROADMAP_AUDIT.md`.
This older A–L view is retained as a compact project summary and must not be
read as proof that every item in the expanded roadmap is complete.

This file records verified repository state. A phase is marked complete only when
its listed deliverables and tests exist in this checkout.

| Phase | State | Verified evidence / remaining limit |
|---|---|---|
| A — evaluator | Complete | One-to-one Hungarian matching, sparse unknown edges, GT-confirmed divisions, production linker path, paired bootstrap, corrected DoG/border reruns, regression tests |
| B — data correctness | Complete | Global-ID overlays, observation/track/lineage/generation columns, float centroids, non-destructive QC, graph invariants |
| C — synthetic truth | Complete | Nine controlled scenarios return exact images, observations, tracks, edges, divisions, and lineages |
| D — real revalidation | Limited by local data | Corrected reruns cover all three locally available paired samples (1.51 GB); paired intervals include zero. More local samples are not present |
| E — scientific tables | Implemented | OME-Zarr physical metadata and observation/track/division/lineage/QC tables with path-based motion statistics. Probability calibration remains pending because no suitable held-out detector probabilities exist |
| F — scalability | Implemented and profiled | Sparse `cKDTree` candidate components, no global signature tensor, content/config detection cache. At 800 cells: exact assignment match, 3.23x measured speedup; see `results/linking_profile.json` |
| G — learned detection | Not complete | Ground-truth-seeded geometry and image linkers tied their baselines on top-1 parent accuracy. The all-199 real-DoG evaluation rejected the learned linker: composite delta -0.022054 [-0.024782, -0.019467]; neither linker is an image detector. Zero-shot Trackastra was also rejected after an all-199 evaluation; a learned image detector still must be trained and validated |
| H — global tracking | Partly complete | A tested multi-frame continuation optimizer jointly selects continuation and gap edges, then applies constrained adjacent-frame division inference. The all-199 joint run timed out after only 54/71 first-fold samples, so it has no valid verdict. The Trackastra division-only hybrid completed all 199 but accepted zero proposals and was rejected. Optional score and persistence gates remain locally tested but cannot rescue a zero-eligibility gate |
| I — reproducibility | Complete locally | Editable installation in the project venv works without `PYTHONPATH`; config validation, library logging, cache, run manifests, and package metadata are tested. Experiment scripts print progress intentionally |
| J — scientist UI/exports | Complete locally | Cell Explorer exposes identity, generation, physical coordinates/time, QC, parent and calibrated-confidence availability; audited corrections and downloadable scientific CSV/JSON/manifest bundles are implemented |
| K — release consistency | Complete locally | README, changelog, citation, manuscript, package version, and actual Streamlit entry point agree on v1.8. Published deployment was not changed in this local run |
| L — comparison study | In progress | Exact-observation synthetic comparison exists in `results/tracking_backend_comparison.json`; both tracking backends reached mean edge F1 1.0. The real detected-data learned linker and zero-shot Trackastra were rejected. The division-only Trackastra candidate also failed to change any edge; the joint candidate timed out before completing either embryo fold |
| M — comparative development | Foundation implemented | Canonical-identity event comparison, first-divergence reporting, termination candidates, division orientation/asymmetry, interdivision atlas statistics, and local-density features are tested and exported. Real replicated control/perturbation data, identity registration, time alignment, biological validation, and uncertainty-aware inference remain |

## Current evidence boundaries

- Kaggle full-annotation training used all 199 GEFF graphs and produced a small
  parent-link checkpoint. Parent accuracy did not improve over nearest distance.
- The completed image-aware Kaggle experiment used all 199 paired volumes but only
  extracted patches at annotated coordinates. It tied distance on top-1 parent
  accuracy and is not an official competition evaluation.
- `scripts/evaluate_detected_linker.py` implements the required detector-output
  training and end-to-end comparison. One- and three-volume-per-embryo pilots lacked
  safe negatives in the `44b6` training fold. The all-199 Kaggle run completed and rejected the candidate under the pre-registered rule.
- Kaggle lists 87.61 GB and about 24.9k physical files. The training inventory is
  199 complete `.zarr`/`.geff` pairs (71 `44b6`, 128 `6bba`); chunk files are not
  independent biological samples. The active command has no sample limit.
- The exact synthetic backend comparison is deliberately not evidence of better
  real tracking: both backends tied at mean edge F1 1.0, and the global path was
  slower in the measured cold-process run.
- No result in this repository establishes state-of-the-art or world-best status.
- Comparative event differences are research measurements, not diagnoses. Numeric
  IDs from separate runs are never treated as homologous cells without a canonical
  identity or validated mapping.
- A bounded three-volume classical division-gate smoke test changed proposal
  counts but had no evaluable sparse division signal; it cannot select a cheaper
  threshold alternative. Its table is `results/division_gate_local_smoke.csv`.
- Trackastra division-only Version 1 processed all 199 samples but accepted zero
  of 129,476 proposals, so every metric delta was exactly zero. A stricter score
  or persistence gate cannot change that result; the next protocol starts with a
  zero-cost eligibility preflight and aborts if the count remains zero.
- The joint hypothesis Version 1 hit Kaggle's 12-hour CPU limit after 54/71
  first-fold samples. Its positive partial deltas are incomplete and are not an
  adoption result.
- The original baseline under `C:\Projects\cell-tracking\baseline` remains unchanged;
  `C:\Projects\cell-tracking\tools\verify-baseline.ps1` verifies its 61 files.


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

The cached four-volume joint Trackastra evaluation completed with exploratory
positive mean deltas, but only one of six confirmed divisions was recovered and
2,009 edits created 3,288 unknown division candidates. Phase H/11 remains
partial until frozen, embryo-disjoint validation meets the existing CI rule.


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


## 2026-11-19: submission reliability fixes and local zebrafish verification
See SUBMISSION_STATUS.md for item-by-item audit disposition. Fixed input/result invalidation, empty schemas, calibration/provenance, independent timing verification, atlas replication, run-specific exports, source hashes and sparse GEFF checks. Windows reference matching preserves uint64 IDs; global topology preview filters edges to its bounded node subset. No evaluator threshold/split/decision-rule change and no Kaggle training or cloud action. Local complete 100-frame A/B inference: A 18805 observations/17912 edges, 42/52 annotations matched; B 10708/8397, 3/51 matched. All 260 division proposals remain unverified; neither local reference contains an annotated division parent. B remains a measured failure, not an improvement. Detailed artifacts/checksums: results/submission_biohub_20261119. No validated genetic diagnosis, independent cross-embryo association or full-detection claim. Baseline remains separate.
