# Submission status — 19 November 2026

This release is a microscopy analysis/review prototype. It supports real model
outputs and full local zebrafish recordings. It is not a genetic diagnostic.

## Audit remediation

| Audit item | Current disposition |
|---|---|
| 1. Matched genetic experiment absent | Explicitly scoped out. Unverified zebrafish files display Recording A/B; no mutant labels invented. |
| 2. Stale input/results | Fixed: new input/source clears results/review state; single-recording configuration changes require rerun; failed replacement inference clears old output. |
| 3. Wrong model provenance | Fixed: exports use actual backend, checkpoint hash where relevant, per-recording config and input hash. |
| 4. Empty predictions | Fixed: stable empty schema, empty scientific tables and valid graph export. |
| 5. Incorrect timing verification | Fixed: separate source-hash-bound metadata per recording, never verification from another input or numerical interval. |
| 6. Calibration mismatch | Fixed for reporting: effective config retained with output; scientific tables use it. OME units converted; missing units remain unverified. |
| 7. Alignment checkbox | Input-hash-bound reset; alignment disabled for supplied exploratory pairs. No automatic stage registration claimed. |
| 8. Missing GEFF evaluation | Implemented sparse-safe checks for available Biohub GEFF; unmatched predictions remain unknown. |
| 9. Projection limitations | Retained experimental limitation; original classical 3D peak workflow available for Biohub. No claim of full 3D Cellpose segmentation. |
| 10. Detection/division quality | Measured and disclosed, not solved. Low annotated recall warns visibly; unvalidated division proposals are never confirmed events. |
| 11. Genetic diagnosis absent | Explicitly outside validated scope. No classifier/causal claim added. |
| 12. Atlas pseudoreplication | Fixed: embryo IDs required; duplicate embryo/cell measurements rejected. |
| 13. Large-run UI delays | Progress added, file-backed inputs retained, unnecessary float checks/copies reduced; complete saved runs reload without inference. Background scheduling and mid-run cancellation are not implemented. |
| 14. Dense learned matrices | Bounded at 8 million pairs before allocation; unnecessary output copies removed. No claim of a new sparse solver. |
| 15. Export overwrites | Run-ID and graph-revision folders; original paired predictions saved separately from review edits. |
| 16. Stale submission/docs | Updated paths/workflow; new archive and extraction check are recorded in results/submission_verification_20261119. |
| 17. Reproducibility | Dependency constraints aligned, tested Windows environment recorded, CI restored, source content hashes added to manifests. No remote CI pass or fresh dependency install on another PC claimed. |
| 18. Missing regressions | Targeted reliability tests added; suite and real-output UI checks recorded in verification artifacts. |

## Additional bugs discovered during release verification

- GEFF IDs were truncated by platform-dependent NumPy integer conversion on
  Windows. Fixed using Python integer IDs; regression covers IDs above 32 and
  63 bits. Matching/scoring thresholds and evaluation logic remain unchanged.
- Global graph preview reintroduced all nodes through unfiltered edges,
  bypassing the node limit and making full-recording review extremely slow.
  Fixed induced-subgraph selection and bounded layout; original result unchanged.

## Actual full-recording result

Source: two already-local Biohub zebrafish recordings, all 100 frames each,
classical peak detection and classical linking. No training or Kaggle run.

| Measurement | Recording A | Recording B |
|---|---:|---:|
| Predicted observations | 18,805 | 10,708 |
| Predicted edges | 17,912 | 8,397 |
| Unverified division proposals | 96 | 164 |
| Annotated nuclei matched | 42 / 52 | 3 / 51 |
| Annotated-cell recall | 80.77% | 5.88% |
| Annotated edges recovered | 28 / 50 | 1 / 49 |
| Inference runtime on this PC | 40.48 s | 42.76 s |

These annotations are sparse. Most edges and all division proposals lack an
evaluable reference. Neither file contains an annotated division parent, so
division accuracy cannot be established. Conditional precision is not full
tracking performance. Recording B is inadequate for a reliable lineage
comparison; keep its failure visible. File size and a plausible graph do not
establish accuracy. Both recordings have the same embryo prefix and unverified
genotypes. Artifacts: results/submission_biohub_20261119 with checksums.

## Judge demonstration

Follow SUBMISSION_GUIDE.md. For a trained neural detector demonstration, choose
the experimental official pretrained Cellpose nuclei backend and the small real
wild-type/wee-1.1 recordings; show Reference check and its errors. For a full
100-frame zebrafish workflow, choose the Biohub classical option and load the
saved analysis. Distinguish these two pipelines explicitly.

Show the input images, predictions, parent/daughter inspection, comparison graphs,
reference-supported metrics, provenance and limitations. Do not claim genetic
causation, complete detection, world-best performance or an official score.

## Home overview update
Replaced external promotion buttons and static pipeline constants with the project purpose, detection/tracing/comparison feature cards, actual selected-recording counts, review navigation and the explicitly prospective genetic-comparison research question. Theme and main navigation retained. New Home/navigation regression passes.
