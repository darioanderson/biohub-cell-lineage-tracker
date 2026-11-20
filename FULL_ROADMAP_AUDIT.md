# Expanded roadmap audit — 2026-09-20

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

Update 2026-09-22: solver efficiency repaired with exact equivalence tests;
sparse detector supervision and real-model output shape repaired. Verification:
112 tests pass, original 61-file baseline unchanged. A private, frozen CHO
centroid detector feasibility run has been dispatched, not yet validated.
Biohub image training needs justified supervision; CHO success alone will not
complete that phase. No rejected linker promoted, no accuracy gain claimed,
and no remaining biological-analysis phase marked complete without evidence.


Update 2026-09-22: expansion complete, 24 unique sample/arm rows and executed
source hashes verified. No division recovery on five annotated events; correct
link losses and growth in unknown divisions prevent promotion. Division/global
tracking work remains experimental. Older RUNNING notes below are historical.

Latest (2026-09-22): fixed candidate and source hashes frozen for eight
additional-volume validation, now RUNNING on private Kaggle. The same two
embryos remain reused. Per-edit loss/recovery auditing is implemented and
tested; total local suite 94 tests, baseline verification 61 files. This does
not complete the division/global tracking phases or authorize adoption.

This audit is based on the files and tests in `C:\Projects\cell-tracking\advanced`.
The roadmap's stated OneDrive path and Flask UI are stale: this checkout uses
`app.py` with Streamlit. A phase is **partial** when any requested deliverable
or real-data validation is absent.

| Phase | State | Verified evidence and remaining work |
|---|---|---|
| Known bugs | Verified locally | Hungarian detection matching, sparse unknown edges, GT-supported division F1, global IDs, and paired bootstrap have regression tests. Prior all-data runs finished or timed out; the four-volume diagnostic is now running. |
| 0 | Substantially complete | True traversed track lengths, observation/track/lineage identity, candidate division labels, raw detector scores, non-destructive QC, float centroids, explicit GEFF errors, and graph timing/gap invariants exist. |
| 1 | Partial | Core sparse evaluator is correct and exposes evaluable counts. All-199 classical/linker evaluation exists; distribution metrics, fragmentation, ID switches, completeness, and DET/TRA remain. |
| 2 | Substantially complete | 85 tests cover physical NMS/scale/centroids, known-blob detection, identity/translation/gates/division/gaps, graph invariants, sparse GT, comparative-development primitives, and byte-deterministic complete-pipeline output. Exact comparison against Kaggle's sample-submission dtypes still requires the mounted data. |
| 3 | Implemented locally | Exact truth includes stationary, translation, acceleration, crossing, dense, entering/leaving, missed observation, occlusion, noise, low SNR, anisotropic Z, one/multiple mitosis, and no-division controls. Full pipeline accuracy thresholds remain Phase 2 work. |
| 4 | Partial | Observation/track/division/lineage/QC tables exist. Background correction, uncertainty, parent/daughter detail, morphology evidence, and validation status need expansion. |
| 5 | Partial | OME-Zarr voxel and frame-time metadata are read and propagated. Per-run calibration is recorded; metadata coverage needs real-data validation. |
| 6 | Partial | Lifetime, displacement, path length, speed, and persistence exist. Acceleration, turning angle, gaps, neighbor density, interdivision interval, lineage depth, and survival remain. |
| 7 | Missing | No held-out probability calibration, reliability diagram, ECE, Brier score, or alternative-link ambiguity. Raw scores are not called probabilities. |
| 8 | Partial | Classical arm is preserved and corrected local reruns exist on three local samples. All-199 linker comparisons exist; detector-specific expanded comparisons remain. Several hidden detector constants remain unpromoted. |
| 9 | In progress, opt-in | A real TemporalUNet3D adapter, sparse-safe targets, tests, and all-data trainer exist. No trained image checkpoint or fair held-out win exists yet. |
| 10 | Partial | Framewise Hungarian and multi-frame global continuation/gap optimization exist. Kalman filtering and calibrated ambiguity remain. |
| 11 | Partial | Division geometry and a constrained joint two-daughter candidate exist. Its first full run timed out after 54/71 first-fold samples. The Trackastra division-only full run accepted zero proposals and changed no edges. The local successor preserves the ordered baseline prefix and supports two-edge score and persistence gates, but a preflight must first find an eligible proposal. |
| 12 | Partial | Physical NMS tests exist. The two-pass redundancy and density-stratified real-data recall are unresolved. |
| 13 | Implemented locally | `cKDTree` gated connected components feed sparse Hungarian problems and match the dense reference in tests. |
| 14 | Partial | The production sparse path avoids a global signature tensor, but `_signature_cost` still materializes dense tensors inside components; maximum-scale memory benchmarks remain. |
| 15 | Partial | Linking profile exists. Vectorized link statistics and complete runtime/peak-memory component profiling remain. |
| 16 | Implemented locally | Detection cache uses content/config identity and is tested. Feature cache remains optional future work. |
| 17 | Partial | Run manifest records environment/config/input provenance. Every artifact is not yet bound to a run ID. |
| 18 | Partial | Stable ordering and seeded synthetic/training paths exist. Byte-identical complete-pipeline regression exists; GPU nondeterminism reporting remains. |
| 19 | Implemented locally | `Config.__post_init__` rejects invalid scale, timing, thresholds, sigmas, gates, backends, and learned-detector settings. More cross-field constraints can be added as features grow. |
| 20 | Partial | Core modules use logging, while experiment/training scripts intentionally print progress. Logging coverage and CLI verbosity are incomplete. |
| 21 | Partial | Package and training requirements exist. Bounded lock files and a formal core/app/dev split remain; Flask is irrelevant because the actual UI is Streamlit. |
| 22 | Implemented locally | `pyproject.toml` supports editable installation in the project venv without manual `PYTHONPATH`. |
| 23 | Improved, partial | CI compiles the real Streamlit app, runs all pytest tests, and checks an exact one-mitosis lineage. Ruff and type checking remain. |
| 24 | Partial | Baseline and advanced trees are separate and caches/venv are ignored. UI choice is documented as Streamlit; naming cleanup and repository artifact review remain. |
| 25 | Partial | Global IDs, raw image overlays, time stepping, track/lineage views, QC, and candidate labeling exist. A true 3D uncertainty view and GT/confirmed-division toggles remain. |
| 26 | Implemented locally | Cell Explorer and non-destructive approve/reject/relink/mitosis curation exist. Alternate-link candidates are not yet calculated. |
| 27 | Partial | Required CSV/JSON/manifest bundle exists. Units/schema version in every export and TrackMate/Mastodon formats remain. |
| 28 | Partial | `EXPERIMENTS.md`, scripts, paired bootstrap, and negative results exist. A single automated experiment-record writer remains. |
| 29 | In progress | Synthetic backend comparison exists. Identical held-out real comparisons require the Kaggle output and trained image detector. |
| 30 | Not complete | The system cannot yet provide validated calibrated confidence or embryo-level accuracy variability for all scientific questions. |
| 31 | Substantially complete locally | Evaluator regression tests cover one-to-one matching, sparse unknowns, GT divisions, and paired bootstrap; production/evaluator parity for gap/global paths needs a dedicated integration test. |
| 32 | Missing/partial | Border tests exist indirectly, but a current-best real failure-mode report and density analysis based on image/detected neighbors are absent. |
| 33 | Partial | Manifests/checksums and embryo-disjoint folds exist. A reusable label-support map and frozen subset manifest remain. |
| 34 | Substantially complete | README, changelog, citation, paper, version, and Streamlit app were aligned for v1.8. The supplied roadmap's Flask references are intentionally rejected as stale. |
| 35 — cross-embryo identity | Partial | Comparison code requires canonical cell names or an explicit validated mapping and refuses to guess from run-local IDs. Automated registration and mapping confidence remain. |
| 36 — lineage divergence | Implemented locally, data pending | Missing, unexpected, and time-shifted canonical events plus earliest divergence are reported. Replicated normal/perturbation validation remains. |
| 37 — programmed death | Partial | Track ends are classified as censored, division, linked boundary, annotated-death candidate, or unresolved. Image/marker confirmation is required before calling apoptosis. |
| 38 — division orientation | Implemented locally | Joint daughter axis, embryo-axis angle, sister distance, and midpoint residual are exported. Embryo-axis registration and real controls remain. |
| 39 — interdivision atlas | Implemented locally, data pending | Canonical-cell control medians, MAD, and quantiles are supported. A replicated normal reference atlas has not been imported. |
| 40 — asymmetric division | Partial | Daughter intensity asymmetry and optional validated volume asymmetry are exported. Segmentation-derived cell volume and biological thresholds remain. |
| 41 — crowding accuracy | Partial | Same-frame physical neighbor density is implemented. It still needs matched GT errors and held-out density-stratified confidence intervals. |
| 42 — perturbation datasets | Data acquisition pending | Digital Development is a promising gene-perturbation source; matched images, stable identities, licensing, stage alignment, and replicate structure must be verified. Stress datasets remain candidates, not integrated evidence. |
| 43 — developmental atlas viewer | Partial | Timeline, 3D observations, lineages, and cell explorer exist. Cross-embryo alignment and atlas overlays remain. |
| 44 — cross-species comparison | Missing | Requires organism-specific identity ontologies and comparable developmental landmarks; direct numeric-ID comparison is invalid. |
| 45 — gene-expression fusion | Missing | Requires a lineage-resolved expression dataset and explicit cell/time mapping. No molecular claims are made from images alone. |
| 46 — trust map and correction learning | Partial | QC and audited curation exist. Calibrated spatial uncertainty, alternative-link probabilities, and correction-to-training provenance remain. |
| 47 — external benchmark | Pending | Cell Tracking Challenge formatting and local validation remain. Any public submission requires separate user authorization and comparable official scoring. |

## Latest external experiments

The division-only Trackastra hybrid completed all **199** paired volumes (71
`44b6`, 128 `6bba`) but accepted zero of 129,476 proposed splits, so every metric
delta was zero and the pre-registered rule rejected it. The joint two-daughter
run detected all 199 volumes and trained its first fold, then Kaggle canceled it
at 12 hours after only 54/71 first-fold evaluations. Its incomplete positive
prefix has no decision status. Neither run is image-detector training or an
official competition score. Preserved artifacts and hashes are under `results/`.

## Current gate

Do not promote a learned model until an embryo-disjoint, same-detections,
same-metrics paired comparison has a positive lower confidence bound. The full
87.61 GB dataset remains mounted in Kaggle and is not downloaded locally.


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

The cached joint Trackastra candidate improved all four inspected samples and
recovered one confirmed division, but its enriched/reused selection and large
unknown-proposal burden make it exploratory. Global tracking/division phases
remain partial; production defaults are unchanged.


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
