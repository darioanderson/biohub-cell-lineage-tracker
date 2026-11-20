# Project audit — 19 November 2026

Scope: current advanced project, local data/provenance, saved experiment outputs,
app loaders/comparison/exports, inference and scientific tables, packaging and
tests. No fresh training, Kaggle status check, full-199 inference, or official
benchmark was performed. This is an audit, not a claim that the listed fixes are
implemented. Production code was not changed during this audit.

## Overall assessment

The project is a usable microscopy tracking/review prototype. It is not a
validated genetic-defect detector or a validated complete-cell lineage system.
The most urgent problems concern attaching the right result to the right input,
truthful provenance, calibration, and data suitable for genetic comparison.

## Prioritized findings

### 1. No matched zebrafish control/mutant experiment is established — P1, scientific blocker

Evidence: data/local_biohub_pair/PROVENANCE.json identifies two recordings with
the same 44b6 prefix, both with unverified genotypes. Their Control/Variant UI
slots are not biological labels. The C. elegans control/wee-1.1 pair comes from
different studies and is explicitly unaligned. The Emb20/21 alternative has
unresolved image axes/stages and no matching lineage labels in the two checked
archives. Janelia wild-type/MZoep data is a source candidate, not an imported,
validated image dataset.

Consequence: differences cannot be attributed to genotype. A true research
comparison needs source-provided conditions, multiple independent embryos,
developmental alignment, and matched measurements. Use Recording A/B until the
conditions are established.

### 2. New single-recording inputs retain the old analysis result — P1, reproduced bug

Evidence: app/ui.py:57, :68, :80 update images/name/source but do not invalidate
result or curation state. A controlled call to load_uploaded_npy preserved an
existing result object after replacing the image. Evidence saved in
results/project_audit_20261119/reproductions.json.

Consequence: old detections/lineages can remain attached to new microscopy. The
single-recording rerun failure path also retains the preceding result
(app.py:260). Centralize input-change invalidation and bind results to an input
and effective-configuration fingerprint; invalidate failed replacement runs.

### 3. Paired provenance exports name the wrong detector — P1, confirmed code defect

Evidence: app/paired_movies.py:304 hard-codes detector to Kaggle-trained CShaper
centre detector, although selectable backends include Cellpose and classical
Biohub. The export also omits the full effective per-recording configurations.

Consequence: downloaded run details can misstate how results were generated.
Build provenance from the actual backend, weights, calibration and source data;
use no checkpoint label for a classical detector.

### 4. Zero-detection recordings break scientific/lineage views — P1, reproduced bug

Evidence: src/biohub/analysis.py:242 builds an empty DataFrame without columns
when no nuclei are detected. A real pipeline invocation on two all-zero volumes
returned zero columns; build_scientific_tables then raised a missing-identity-
columns error. The Lineage page calls that builder before its empty-table check.

Consequence: valid empty predictions become a page error. Preserve an explicit
empty schema and support empty summaries, exports and figures.

### 5. Time verification is inferred from numbers and from either input — P1, confirmed code defect

Evidence: app/paired_movies.py:174 uses any known zebrafish hash to classify the
pair; :226 marks both results time-verified if that flag is true, or if their
configured interval is numerically 10 or 100. A number alone does not establish
metadata provenance. Mixing one known Biohub file with an unknown file can mark
the unknown file's timing verified.

Consequence: unverified seconds/speed exports may become available. Carry
independent spatial/time verification and units for each recording; do not
infer verification from a numeric value or the other recording.

### 6. Effective calibration is not returned with pipeline results — P1, confirmed design defect

Evidence: src/biohub/analysis.py:85-91 reads source calibration into a local copy
of cfg, but returns no effective config/calibration in result.stats.
src/biohub/scientific.py:36 uses the externally supplied cfg for micrometers and
seconds. The uploaded single-recording path does not recognize the paired
provenance manifests. data.read_physical_metadata also reads scale values without
converting declared units.

Consequence: tracking and exported physical measurements can use different
calibrations when metadata differs from the UI config. Store calibration and
units with each result, and require explicit calibration for unknown uploads.

### 7. Alignment is a persistent checkbox, not a stored transform — P2, confirmed limitation

Evidence: app/paired_movies.py:241,:290 and app/pair_dashboard.py matched-frame
panel. Preset buttons clear the checkbox; arbitrary replacement file paths do
not explicitly clear it. Comparisons merge by frame index when checked.

Consequence: an earlier alignment assertion can carry to a different pair; the
app does not estimate stage/time alignment or establish homologous cell IDs.
Bind alignment to input hashes and store its evidence, time mapping and method.

### 8. Biohub reference graphs are not connected to paired reference checking — P2, confirmed gap

Evidence: local Biohub manifests record reference_geff, but
app/pair_dashboard.py:reference_counts_for_pair only recognizes C. elegans
manifests and their tracing CSVs. The new local Biohub preset therefore does not
use its available GEFF annotations in the comparison's Reference check.

Consequence: rich source annotations do not automatically validate the displayed
predictions. Add sparse-safe location, edge and division matching; do not treat
unknown detections as false positives/background.

### 9. Cellpose projection cannot represent all overlapping 3D nuclei — P1, model limitation

Evidence: src/biohub/cellpose_projection.py segments one XY maximum projection
and derives a single Z estimate per 2D mask. It cannot resolve two nuclei merged
into one projected mask. The Z estimate is intensity-based, not learned 3D
segmentation. The UI already discloses this limitation.

Consequence: a larger image file cannot alone produce complete 3D detections.
Use an acquisition-appropriate 3D detector, or retain this mode as experimental
review and quantify its limitations against independent spatial annotations.

### 10. Detection and division recovery remain empirically weak on some target clips — P1, saved evidence

Evidence: results/division_clip_check_20261119/summary.json: control reference
counts [2,2,3,3,4], predictions [1,1,1,1,3], zero predicted division parents.
The variant's three proposals are not confirmed true divisions. The native-grid
CShaper failure report records 66 predictions versus two reference nuclei.
The 30-frame variant count check disagrees with references in 25/30 frames.

Consequence: family trees can be wrong even when the page renders correctly.
Measure location/link/division accuracy on an independent recording before
adopting a model; do not manufacture branches to improve presentation.

### 11. No validated genetic-defect inference model is attached — P1, scope gap

Evidence: paired dashboard compares predictions descriptively; it does not
establish cross-embryo biological identities or infer genotype. The separate
canonical-event analysis requires externally supplied canonical identities.
Existing CShaper and Cellpose checkpoints are detectors, not genetic classifiers.

Consequence: graphs cannot currently identify genetic causes or diagnose a
defective cell. Implement supported phenotypes and event correspondence before
claiming defect association; causal claims require additional experimental evidence.

### 12. Reference atlas overcounts independent embryos — P2, reproduced statistical bug

Evidence: src/biohub/comparative_development.py:159 sets n_control_embryos to
len(values), without requiring or deduplicating embryo IDs. Two observations
from the same embryo were reported as two control embryos in a controlled check.

Consequence: replication can be overstated. Require embryo identifiers, one
defined measurement per embryo/cell/event, and embryo-level uncertainty.

### 13. Large paired inference blocks the Streamlit interaction — P2, runtime risk; not benchmarked here

Evidence: app/paired_movies.py runs both complete movies synchronously in one
button handler. It validates every local frame for finiteness and performs a
whole-array equality comparison for equal-shaped movies. The reused-control
upload path serializes a complete movie into an in-memory byte buffer.

Consequence: UI pauses, extra scans, and memory duplication can increase for
large recordings. File-backed inputs help but do not prove smooth performance.
Add bounded frame loading, progress/cancellation, a job state and measured
peak-memory/runtime budgets. No OOM was reproduced in this audit.

### 14. The larger learned-linker still has dense matrix costs — P2, scaling risk

Evidence: src/biohub/learned_linker.py:65-95 gates features before inference
(the earlier feature-allocation bug is already fixed), but still allocates dense
distance and cost arrays and sorts all sources for each target.

Consequence: very dense frames remain expensive. Profile candidate count,
distance construction and assignment before changing the solver. Do not claim
this explains a past timeout without a matching measured profile.

### 15. Export destinations overwrite prior recording results — P2, confirmed code behavior

Evidence: app.py:852-854,:882 uses dataset_name for fixed CSV/JSON/scientific
paths; paired runs reuse names Control and Variant. Repeated runs overwrite the
same output paths, and export_scientific_bundle is called without input_paths.

Consequence: exported lineage evidence is not reliably separated by input/run.
Use run IDs/input hashes, retain effective configuration and checkpoint hashes,
and distinguish original predictions from curation.

### 16. Submission ZIP and data instructions are stale — P2, confirmed artifact discrepancy

Evidence: results/college_submission_20261119.zip has 124 entries and a different
app/paired_movies.py from the current source. It lacks the local Biohub pair
manifest/preparation script and the contiguous-division preparation script.
DATA_LOCATIONS.md still says Biohub has no local images; that is only true of
the older biohub-stage1 folder, not Desktop/biohub_data.

Consequence: a reviewer can run an older program or follow incorrect paths.
Rebuild a versioned submission with a manifest and install/run it from a clean
extraction before handing it over. Do not include huge image copies accidentally.

### 17. Fresh installs and source provenance are not reproducibly frozen — P2, confirmed maintenance gaps

Evidence: requirements.txt and pyproject.toml declare different NumPy/Zarr
constraints; review dependencies add further limits. Git status shows many
modified/untracked source/test files and deletion of .github/workflows/ci.yml.
The generic run manifest records git HEAD but not dirty-source content hashes;
not all paired exports include effective configs. Tests ran in the existing
parent .venv, not a clean installation.

Consequence: a working developer environment does not prove a fresh reviewer
installation works. Consolidate supported dependencies, preserve licenses,
commit reviewable source/tests, restore appropriate CI, and record source hashes.
Current pip check passes; no dependency conflict in this installed environment
is claimed.

### 18. Passing tests do not cover the newly reproduced failures — P2, confirmed coverage gap

Evidence: 139 tests passed while stale-input state, empty-node schema and atlas
replication bugs were reproduced separately. Paired tests cover helper state
switching and file validation but do not establish complete real-data accuracy,
fresh-package installation, correct export provenance or responsiveness at scale.

Add targeted regressions for those failures, wrong-backend run manifests,
mixed known/unknown timing, input-dependent alignment reset and a real paired
UI smoke test. Keep accuracy evaluation separate from software tests.

## Existing fixes that should not be reported as still broken

- Missing-checkpoint check precedes torch import in the learned detector.
- Learned-linker feature construction is gated before inference; dense matrix
  scaling remains a separate issue.
- Sparse-supervision tests cover zero unknown-voxel gradient; a compatibility
  pass does not imply successful Biohub detector training.
- Isolated observations are flagged rather than silently deleted by the current
  prune_isolated_nodes implementation. A frame-pruning mismatch was not established.

## Checks performed

- Full local suite: 139 passed (current audit run).
- pip check: No broken requirements found.
- Baseline verifier: 61 original files unchanged, clean baseline working tree.
- Read current code and saved outcomes; reproduced three bugs without changing
  production code. Evidence: results/project_audit_20261119/reproductions.json.
- Inspected submission ZIP/source mismatch and current Git status.

## Recommended execution order

1. Fix stale input/result binding, empty outputs, truthful manifests and effective
   calibration; add regressions. These affect every displayed scientific result.
2. Clear alignment per input pair, connect Biohub GEFF reference evaluation and
   use neutral recording labels when conditions are unknown.
3. Establish the actual control/variant dataset and a frozen independent
   location/link/division benchmark; only then select or train a model.
4. Add durable run storage, measured runtime limits and UI progress/cancellation.
5. Implement canonical event correspondence and replicated phenotype comparison;
   retain descriptive-only labels until evidence supports stronger claims.
6. Consolidate dependencies, update roadmap/data docs, rebuild the submission
   archive and verify it from a clean extraction.
