# Comparison workflow

Use Analysis workspace → Control and variant. The existing app layout and colors
are retained. These pages compare predictions descriptively, not genetic causes.

## Local zebrafish

1. Choose Biohub original pipeline — classical detection.
2. Choose Files on this PC → Use local zebrafish recordings — all 100 frames.
3. Load recordings. The files are read from disk, avoiding browser upload limits.
4. If available, Load saved zebrafish analysis — no rerun restores complete
   predictions only when input hashes and effective settings match.
5. Otherwise Load model and analyze both performs classical inference with
   progress, saves each run separately, and checks available GEFF annotations.
6. Select Recording A or Recording B, then Detection or Lineage; open Compare
   Lineages for descriptive plots and sparse reference checks.

These inputs have unverified genotypes and the same embryo prefix. They are not
a biological control/mutant experiment. Alignment remains unavailable.

## Real wild-type / wee-1.1 exploratory pair

Choose Cellpose nuclei — experimental projection review, Files on this PC,
Use small real recordings (under 200 MB), Load recordings, then Load model and
analyze both. This uses official pretrained Cellpose weights, followed by
classical tracking. It performs no new training and no genetic diagnosis.
The sources differ in stage, cadence and acquisition. Alignment is not established.
The variant has known prediction errors, shown in Reference check.

## Interpretation and persistence

The comparison uses original automated outputs; edits are retained separately in
each recording's review state. Numerical track IDs are not homologous biological
cell identities across embryos. Unknown sparse detections are not negatives.
Division parents are proposals until reviewed. Track endings are not cell death.
Run folders include input hashes, per-recording configurations and actual backend
names. Unknown physical/time calibration does not produce trusted speed values.
Selecting a different input or analysis configuration invalidates old predictions.
