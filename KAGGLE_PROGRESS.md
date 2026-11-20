# Kaggle training progress - 2026-09-19

## 2026-11-13 — Licensed pretrained Biohub baseline inference dispatched

Goals frozen in `PROJECT_GOALS.md`. Private notebook
`dushi2/biohub-organizer-pretrained-submission-check` Version 1 is RUNNING,
using the organizer's published `cellmot-baseline-artifacts` dataset (trained
3D temporal U-Net plus transformer and bundled wheels). It reads mounted test
images, predicts all test volumes, writes `submission.csv`, and checks graph
structure and SHA-256. Source notebook and metadata are in
`notebooks/kaggle_organizer_baseline/`. This is **inference with existing weights**,
not new full-data training or an official submission. The organizer's reported
0.810 public score belongs to their notebook, not this project. A monitor is
active; do not launch a duplicate.

The organizer's training script explicitly includes a detection-negative
weight on unannotated voxels, even though their edge loss masks sparse labels.
Therefore it does not by itself solve our verified-negative supervision problem.
Do not run a claimed sparse-safe full-199 detector training based on that script
without changing and validating its detection objective.


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

## 2026-09-22 — CHO detector feasibility Version 1 dispatched

Private notebook: https://www.kaggle.com/code/dushi2/biohub-cho-detector-feasibility
Version 1 push succeeded. No accuracy result is available yet. This is a new
image-centroid detector experiment, not a duplicate joint-linker run.
Frozen source hashes and protocol: `notebooks/kaggle_cho_detector/frozen_protocol.json`.
Kaggle downloads the checksum-pinned 108 MB public CTC CHO archive; no Biohub
87 GB download. Train sequence 01, choose checkpoint and threshold using declared
sequence-01 validation frames, then evaluate sequence 02 once. Maximum 30 epochs
and 30 steps/epoch. CHO results cannot establish zebrafish tracking performance.


## Latest result — joint expansion complete, candidate not promoted

Version 351694868 completed all eight samples and three arms: 24 unique records,
no duplicated sample/arm output despite repeated log lines. Frozen source hashes
verified against the executed notebook payload. Outputs/source/logs/checksums:
`results/kaggle_joint_expansion_v1/`.

Mean composite: classical 0.74902131, joint Trackastra 0.74941259 (delta
+0.00039128). Four sample scores improved, three declined, one tied. Correct
edges removed: 9; correct edges added: 15 (net +6). Candidate made 1,930 edits;
unknown division candidates increased 1,787 to 3,716. Zero of five annotated
divisions recovered; one evaluable false division. Geometry control made zero
edits and tied classical. Mean edge recall increased only 0.00077036.

Decision: do not promote or scale this candidate to all 199 based on these
results. The intended division gain did not generalize to this small expansion,
and correct continuations were lost. This is exploratory benchmark reuse, not
independent biological evidence or an official score. Preserve negative results
and reassess division modeling/label support before another experiment.

## Latest — frozen joint expansion running, 2026-09-22

Private `dushi2/biohub-joint-expansion` Version 1 is confirmed RUNNING.
Eight additional volumes, balanced across both embryos and annotated-division
strata, exclude recent diagnostic/design samples. Exact names, fixed parameters
and runtime-verified source hashes are in
`notebooks/kaggle_joint_expansion/frozen_manifest.json`.
See `JOINT_EXPANSION_PROTOCOL.md`. The benchmark and both embryos have already
been used; do not describe this as independent biological validation.
The new audit separately counts correct edges removed, correct edges added,
inconsistent and unknown edits. Parameters, production defaults and the sparse
evaluator are unchanged. No training or adoption claim. Local checks: 94 tests
passed; all 61 original baseline files verified.

Notebook: https://www.kaggle.com/code/dushi2/biohub-stage-1-small-data-training

Latest execution was observed in the interactive Kaggle notebook. Training completed after 42 epochs, best validation checkpoint at epoch 27. A Save & Run All version named "Stage 1 - real sparse-label linker" was created; persistence of its output files must be checked separately.

The competition dataset is mounted at /kaggle/input/competitions/biohub-cell-tracking-during-development. No 87 GB download to this laptop was performed. Current training uses only GEFF annotations, not image volumes. Runtime reported Torch 2.10.0+cpu and CUDA False.

Model: 10-feature, 32/16-hidden-unit neural edge classifier. Features: signed/absolute physical displacement, distance, squared distance, distance rank, source count. Maximum 16 candidate parents within 20 micrometres. Negative examples are alternative parents for a target with one annotated parent; this assumes one biological parent per target.

Training: first 12 sorted samples from embryo 44b6. Validation: first 6 sorted samples from embryo 6bba. Training contains 3,075 candidate pairs, 2,443 labeled positive edges. Validation contains 6,342 candidates. Candidate coverage of eligible validation parents was 1.0. Normalization is fitted on training only.

Measured validation results:
- Nearest-distance baseline average precision: 0.9998817601024923
- Learned classifier average precision: 0.9999919752778803
- Both methods parent-selection accuracy: 0.9996032533227535
- Best checkpoint training loss: 0.0007358443030030825
- Interactive-run checkpoint SHA-256: 440f82cec57ed97b807984bb1e31afb773d29b7037101675fd5842e9aaf988e8

There is NO parent-accuracy improvement. The near-perfect results reflect a restricted, easy candidate set built from ground-truth detections. This is not end-to-end tracking, official competition performance, SEG, TRA, or evidence of world-best performance. Checkpoint selection uses this validation set; no independent test score exists. The nearest-distance comparator is not the complete original Biohub repository pipeline.

Files written in Kaggle /kaggle/working/biohub_stage1: best_linker.pt, protocol.json, history.json, status.json. The local baseline remains untouched. Local CHO experiments are prepared but have not trained. The large local PyTorch wheel has been repaired and verified against the official SHA-256 but has not been installed.

## Full-annotation run (saved as Kaggle Version 3)

The full annotation linker completed on all 199 GEFF graphs: 180,268 candidate
pairs and 128,866 positive edges, with 27 fixed epochs. The final checkpoint was
downloaded to `models/full_annotation_linker/final_all_annotations.pt`; its SHA-256
is `5daa0a8cd3084fe37b9bf2e62e0a172e371af5ab27044d0918c254f3dd0a5810`.

Embryo-disjoint results:

- Held-out `44b6`: learned AP 0.9999049424200822 versus distance AP
  0.9991353628198598; both parent accuracies 0.9993947341874306.
- Held-out `6bba`: learned AP 0.9996628629534867 versus distance AP
  0.999505677374154; both parent accuracies 0.9985512163364112.

The learned model did not improve parent-selection accuracy. This remains a
ground-truth-coordinate linker and is not a detector or an official competition
evaluation.

## Full paired-image run (completed; saved as Kaggle Version 4)

A second Kaggle stage read all 199 paired `.zarr` image volumes and GEFF graphs.
It extracted local intensity patches only at annotated cell coordinates, added 15
image features to the 10 geometry features, trained embryo-disjoint folds, and fit
a final model on all samples. Kaggle reported CPU execution and 1,961.61 seconds.

Measured embryo-disjoint results:

- Held-out `44b6`: image AP 0.9998466921697537 versus distance AP
  0.9991353628198598; both parent accuracies 0.9993947341874306.
- Held-out `6bba`: image AP 0.9998164812496537 versus distance AP
  0.999505677374154; both parent accuracies 0.9985512163364112.

The final all-199 checkpoint hash printed by Kaggle was
`94a537574a453d9a3cb619eaa51fdc8ab4517f93f137408964ca42fd767d4a8a`.
Version 4 is a Quick Save: the notebook and observed output are preserved, but the
Kaggle API exposed only its log, not the `/kaggle/working` checkpoint. The hash is
recorded as provenance; the model is not claimed as locally integrated.

This stage uses every paired sample but does not learn from every voxel. Sparse
unlabeled voxels are never invented as background negatives. It is an image-aware
parent linker seeded by ground-truth coordinates, not an end-to-end image detector.
The image features improved candidate ranking AP but did not change top-1 parent
accuracy. This is a null result for the decision that matters to linking.

## Real detected-data evaluation (completed; rejected)

The private full run `dushi2/biohub-real-detected-linker-full` completed in
8,507.94 seconds on all 199 paired samples (71 `44b6`, 128 `6bba`). The exact
outputs, executed source, notebook, and run log are preserved under
`results/kaggle_detected_linker_full/`; `artifact_manifest.json` records SHA-256
hashes and sizes.

The pre-registered rule was: adopt only if the embryo-level paired 95% CI lower
bound for composite-score delta is greater than zero. It failed. Mean composite
score fell from 0.726048 (classical) to 0.703995 (learned); paired sample delta
was -0.022054, 95% CI [-0.024782, -0.019467]. Edge recall fell by -0.057868,
95% CI [-0.065031, -0.051093], and division precision and recall collapsed to
zero. Detection recall delta was exactly zero, as expected because only linking
changed. Decision: **REJECT; do not integrate this checkpoint.**

Post-run code audit found that BCE positive-class weighting was already present
and sparse unknown candidates were already excluded. The large unknown counts
therefore are not a 13:1 negative-class ratio. A real label bug was found:
targets whose true parent was missed could contribute detected alternative
parents as false negatives. The builder now drops every label for such a target,
with a regression test. A separate opt-in joint hypothesis solver is under
validation; it scores a parent and two daughters as one constrained hypothesis
rather than adding a second daughter after a 1:1 assignment. No improvement is
claimed until the unchanged held-out evaluator measures it.

The private notebook `dushi2/biohub-joint-division-linker` Version 1 attempted
that full held-out validation but Kaggle stopped it at the 12-hour CPU limit
(43,200.6 seconds, exit code 137). It detected all 199 volumes, completed 15
training epochs for the first fold, and evaluated 54 of the 71 held-out `44b6`
samples before timing out. The incomplete 54-sample prefix had mean composite
delta +0.026499 and edge-recall delta +0.080539, but division precision and
recall remained zero. This prefix is time-ordered, omits 17 `44b6` samples and
all `6bba` evaluation, has no paired confidence interval, and cannot support an
adoption decision. The partial model, status, executed source, notebook, log,
and hashes are preserved in `results/kaggle_joint_division_linker_v1/`.

## Trackastra full-data evaluation (completed; rejected)

The competition Models page exposes Jirka Borovec's public Trackastra artifact.
It contains the 3D-capable pretrained `ctc` checkpoint (`model.pt`, 109,906,058
bytes), configuration, and offline packages. The roughly 3.12 GB artifact stayed
mounted on Kaggle and was not downloaded locally. Two setup attempts failed before
model execution because the public artifact's installation recipe had drifted from
Kaggle's runtime; their preserved logs are integration failures, not model results.

The corrected private GPU run `dushi2/biohub-trackastra-full-evaluation` Version 2
completed all 199 paired volumes (71 `44b6`, 128 `6bba`) in 14,182.07 seconds.
Trackastra and classical Hungarian linking received identical production DoG
detections. Hungarian was only the independent control and never fed or repaired
the Trackastra graph. Exact outputs, executed source, notebook, log, and SHA-256
manifest are preserved in `results/kaggle_trackastra_full_v2/`.

The pre-registered rule rejected zero-shot Trackastra. Composite mean fell from
0.726051 to 0.684527. Paired sample delta was -0.041524, 95% CI
[-0.050396, -0.032925]; embryo-level delta was -0.049129, 95% CI
[-0.075678, -0.022579]. Edge recall fell by -0.108698
[-0.126634, -0.091471] and edge F1 by -0.099493
[-0.117755, -0.082896]. Edge precision was essentially tied: +0.000527
[-0.000913, +0.002094].

Trackastra did improve mitosis sensitivity: division recall rose from 0.009631 to
0.062395, delta +0.052764 [0.024696, 0.084590]. Division precision rose from
0.030151 to 0.041625, but its paired sample interval crossed zero:
+0.011474 [-0.020352, 0.042462]. Detection recall was identical, as required.
The division gain did not compensate for the much larger continuation-edge loss.
Decision: **REJECT zero-shot Trackastra; do not switch the production default.**

The private follow-up `dushi2/biohub-trackastra-divisions-full` Version 1
completed all 199 volumes in 14,545.32 seconds. It used Trackastra only as a
division proposal source. A proposed split was eligible only
when one daughter is the existing continuation, the second daughter is unmatched,
and parent distance, sister separation, midpoint, timing, in-degree, and out-degree
constraints pass. This targets Trackastra's measured division-recall gain without
reusing its continuation graph, which caused the rejected edge-recall loss.

The result is conclusive and uninformative for accuracy: **REJECT**. Across
129,476 Trackastra split proposals, 101,811 failed the graph-conflict gate and
27,665 failed geometry; zero were accepted. The candidate therefore equalled
the control on all 199 samples. Composite, edge precision/recall/F1, and division
precision/recall deltas and both paired intervals were exactly 0.0. Baseline and
candidate means were: composite 0.726051, edge precision 0.994568, edge recall
0.645024, edge F1 0.774234, division precision 0.030151, and division recall
0.009631. The pre-registered lower-bound rule was not met.

Executed Version 1 source starts from a de-duplicated baseline edge list and can
only append an accepted daughter. Because no daughter was accepted, the actual
candidate edge set was identical to the baseline for every sample. Version 1 did
not contain the later exact ordered-prefix assertion, score threshold, or temporal
persistence gate. Outputs, source, notebook, log, and hashes are preserved under
`results/kaggle_trackastra_divisions_full_v1/`.

The local post-launch audit strengthened the next evaluator without modifying the
running notebook. Baseline edges are now preserved as an exact ordered prefix,
with per-sample SHA-256 hashes and a hard failure on any mutation or silent
deduplication. Tests cover rejection of a false geometric split and preservation
of unrelated continuations. Trackastra 0.5.3 source confirms selected graph edges
retain normalized `weight` attributes; optional two-edge score and daughter-track
persistence gates are therefore implemented locally for a future pre-registered
run. They are not retroactively attributed to Version 1. Because stricter gates
cannot rescue zero eligible proposals, no duplicate GPU run is justified. The
preflight and nested held-out protocol is frozen in
`TRACKASTRA_DIVISION_FOLLOWUP_PROTOCOL.md`; it requires a nonzero eligibility
count before launch and the same all-199 classical division-gate control.

Post-preservation verification on 2026-09-21: the hybrid artifact directory has
52 hashed files (manifest SHA-256
`a931a80b1f6c6307c263527e96152882a184299805c09218a4dc093f2dc9640d`), and the
joint-timeout directory has 47 hashed files (manifest SHA-256
`ee7ba811bea70b3aaf23e74a05cc35af14015e71b3ee91bf2b97771d9f39f5ac`). The
complete local suite passed 81 tests, and the baseline verifier passed all 61
original files.



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

## Cached joint evaluation integration failure and corrected rerun — 2026-09-22

`dushi2/biohub-cached-joint-evaluation` Version 1 failed before scoring. The
minimal runtime did not contain the optional GEFF/Zarr readers, so
`load_geff_graph` returned no annotation tables and the script raised an
`AttributeError`. This is an environment-integration failure, not evidence for
or against any tracking arm. Its executed source and log are preserved under
`results/kaggle_cached_joint_evaluation_v1/`.

Version 2 adds only the pinned offline GEFF/Zarr packages from the already-used
Trackastra artifact and explicit failures for a missing GEFF path or empty graph
tables. The scoring logic, four fixed samples, three arms, sparse evaluator, and
fixed parameters are unchanged. Eleven focused tests passed before launch.
Kaggle accepted Version 2 and reported it queued. No metric claim is made until
its output completes and is preserved.

## Cached joint evaluation Version 2 completed — 2026-09-22

Version 2 completed all three arms on the four fixed, previously inspected,
annotation-enriched samples. Outputs, notebook, log, metadata and SHA-256
manifest are preserved under `results/kaggle_cached_joint_evaluation_v2/`.
This was a cached CPU evaluation: no image inference or training occurred.

Mean classical metrics were composite 0.732193, edge precision 0.984428, edge
recall 0.621149 and edge F1 0.755983. The joint Trackastra arm measured composite
0.746945 (delta +0.014752), edge precision 0.986961 (+0.002533), edge recall
0.626633 (+0.005483), edge F1 0.760987 (+0.005004), mean division precision
0.25, mean division recall 0.083333 and mean division F1 0.125. Its composite
delta was positive in all four samples. Aggregated evaluable counts changed from
2066 TP / 15 FP / 1065 FN to 2078 TP / 13 FP / 1053 FN. It recovered one of six
confirmed divisions and produced one confirmed division false positive.

The candidate made 2,009 local edits and increased unknown division candidates
from 1,281 to 3,288. Those unknown predictions are correctly excluded from
precision, so sparse metrics cannot establish their correctness. The independent
geometry arm made zero edits and exactly tied classical. Joint scoring took
101.77 seconds across four cached graphs versus 0.12 seconds for scoring the
unchanged baseline.

Verdict: **promising exploratory result; do not adopt or launch all 199 yet**.
These samples were selected for annotated divisions and repeatedly inspected;
there are only two embryo groups and no held-out CI. The result does not meet the
pre-registered adoption rule. The next evaluation must freeze this candidate,
use opposite-embryo selection or untouched volumes, quantify unknown proposal
burden, and retain the independent classical control.


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
`nCHO Version 2 push succeeded; Kaggle status verified RUNNING. Monitor updated to Version 2.
Preflight Version1 failed before execution: missing notebook kernelspec. Preserved log/source in results/kaggle_sparse_preflight_v1. Added python3 kernelspec and dispatched Version2; existing heartbeat updated to monitor this run and preserve completed CHO Version3.

## 2026-11-14 — C. elegans detector Version 3 completed

Kaggle version 369271457 completed 10 epochs and 3,900 optimizer updates
using silver segmentation labels from 195 frames of recording 01. Checkpoint,
notebook, metadata, log, and summary are preserved with SHA-256 hashes in
results/kaggle_ce_detector_v3.

Recording 02 was held out. Across 190 labeled frames, the decoder returned
37,399 predictions for 22,418 annotated observations. In 122 frames it
returned fewer detections than annotations. Summing the per-frame minimum
of those counts gives a 28.91% upper bound on one-to-one annotated recall,
regardless of distance. This is a count bound, not measured recall. The
median of the per-frame median nearest-prediction distances was 21.39 voxels.
At frame 189 the decoder returned 49 centers for 360 annotations.

Keep this checkpoint for diagnosis. The next comparison should inspect raw
probability maps and test whether separating neighboring peaks improves the
count deficit. Tune decoding on training-side data and use an independent
recording for a clean confirmation. Linking was not launched.

## 2026-11-14 — CE center detector, Kaggle version 369386500

Both recording folds completed with 30,800 optimizer updates. Training on
195 frames of 01 matched 10,054 of 22,418 annotations on 02 (44.85%). Training
on 190 frames of 02 matched 4,558 of 23,802 on 01 (19.15%). Matching was
one-to-one within three voxels. Partial annotations preclude treating all
unmatched predictions as false positives. These are localization results,
with no linking or division evaluation.

The 01-to-02 fold exceeded the previous detector's 28.91% count-only recall
ceiling on the same recording. The reverse fold shows weak transfer.
Preserved both checkpoints, summary, source, metadata and log with SHA-256
hashes in results/kaggle_ce_centers_v1.

## 2026-11-15 — CShaper detector and periphery comparison

Both private runs completed 13,440 updates on 168 volumes from Samples04–17.
Both evaluated 36 volumes from Samples18–20, with identical input manifests
and one-to-one matching within three voxels against generated nucleus masks.
The center-only run matched 5,050 of 5,241 reference centers across 5,454
predictions (92.59% precision, 96.36% recall, 94.44% F1 against those masks).
The center/periphery run matched 659 of 5,241 across 33,347 predictions
(1.98% precision, 12.57% recall, 3.42% F1 with its fixed decoder).
Retain center-only as the detection candidate. These outputs do not evaluate
linking or divisions. Sources, logs, checkpoints and hashes are preserved in
results/kaggle_ce_cshaper_v2 and results/kaggle_ce_cshaper_periphery_v1.

## CShaper lineage audit Version 4

Native SegNuc IDs resolved through the published dictionary for all 2,952
nucleus observations sampled across nine adjacent frame pairs. Shared names
across these pairs totaled 1,462. Samples18–20 have 159, 159, and 169 available
adjacent pairs respectively. This checks reference identity coverage on sampled
frames. Tracking and division scores remain pending. Notebook, log, report and
checksums are in results/kaggle_cshaper_link_audit_v4.

## 2026-11-17 — CShaper joint assignment benchmark

The paired exploratory benchmark completed on 81 transitions from the same
three previously evaluated embryos. The 9-voxel baseline recovered 4 of 122
reference divisions. Joint assignment recovered 26 of 122, with 38 true
divisions reachable among its candidates and 99 with all endpoints detected.
Continuation counts fell from 12,024 to 11,990 of 12,920. Division outputs
fell from 235 to 96; 12 baseline and 28 joint outputs had all endpoints
identified, so precision conditional on those endpoints is 4/12 and 26/28.
The unscored outputs and partial reference prevent a complete precision claim.
Artifacts and hashes are in results/kaggle_cshaper_joint_v1.

## 2026-11-17 — CShaper all-pairs detector

Training completed on 2,304 paired volumes across 14 embryos with 27,648
optimizer updates. All 36 evaluation volumes and mask/image hashes matched
the 168-volume baseline. Against generated masks, recall increased from
96.36% to 98.66% (5,171/5,241), precision fell from 92.59% to 89.59%,
and F1 fell from 94.44% to 93.91%. The larger run used three epochs
and the earlier run used twenty, so this compares recipes, not data size alone.
Keep the earlier checkpoint as the F1 control. Evaluate tracking end to end
before selecting the higher-recall candidate. Artifacts and hashes are in
results/kaggle_cshaper_all_pairs_v1.


## 2026-11-19: submission reliability fixes and local zebrafish verification
See SUBMISSION_STATUS.md for item-by-item audit disposition. Fixed input/result invalidation, empty schemas, calibration/provenance, independent timing verification, atlas replication, run-specific exports, source hashes and sparse GEFF checks. Windows reference matching preserves uint64 IDs; global topology preview filters edges to its bounded node subset. No evaluator threshold/split/decision-rule change and no Kaggle training or cloud action. Local complete 100-frame A/B inference: A 18805 observations/17912 edges, 42/52 annotations matched; B 10708/8397, 3/51 matched. All 260 division proposals remain unverified; neither local reference contains an annotated division parent. B remains a measured failure, not an improvement. Detailed artifacts/checksums: results/submission_biohub_20261119. No validated genetic diagnosis, independent cross-embryo association or full-detection claim. Baseline remains separate.
