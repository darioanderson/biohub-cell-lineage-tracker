# College demonstration — 19 November 2026

This is a working microscopy analysis and review prototype. It is not a validated genetic-defect classifier.

## Start in the VS Code PowerShell terminal

```powershell
cd C:\Projects\cell-tracking\advanced
& "C:\Projects\cell-tracking\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
```

Open http://localhost:8501. Keep the terminal running.

Alternatively run `powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1`.
For a fresh extracted project, use Python 3.10, create `.venv`, activate it, then
run `python -m pip install -e ".[app,review,test]"`. The tested Windows package
snapshot is `requirements-submission-lock.txt`; a fresh dependency installation
on another computer is not claimed by the local checks.

## Full zebrafish recording demonstration

1. Select Control and variant → Biohub original pipeline — classical detection.
2. Files on this PC → Use local zebrafish recordings — all 100 frames.
3. Load recordings. If the saved-run button is available, use Load saved zebrafish
   analysis — no rerun. Otherwise Load model and analyze both.
4. Select Recording A or Recording B. All ordinary review pages follow that choice.
5. Open Compare Lineages → Reference check for sparse annotation-supported metrics.

These are full 100-frame local zebrafish samples, about 839 MB per file, with
genotypes unverified. This backend is classical; do not call it a trained neural
model. It is an alternative to the real pretrained Cellpose demonstration below.
The large recordings stay on this PC and are not embedded in the portable ZIP.

## Reliable reporting

New inputs clear old results; changing single-recording settings requires a rerun.
Empty detections are supported. Run-specific exports retain effective calibration
and actual backend names; unverified physical measurements are not presented as
trusted seconds/speeds. Calibration/label warnings should remain visible to judges.
No genetic diagnosis, all-cell detection, world-best result or official competition
score is claimed. See PROJECT_AUDIT_20261119.md and SUBMISSION_STATUS.md.

## Real-data demonstration

1. Select **Control and variant** in the sidebar.
2. Choose **Cellpose nuclei — experimental projection review**.
3. Select **Files on this PC** and **Use small real recordings (under 200 MB)**.
4. Press **Load recordings**, then **Load model and analyze both**.
5. Leave stage/spacing alignment unchecked: this pair is not a matched experiment.
6. Choose Control or Variant to review Volume, Detection, Lineage and Exports.
7. Open **Compare Lineages**. Inspect images, counts, track continuity, family trees, and **Reference check**.

Control contains five real wild-type Emb8 frames (~102 MB); variant contains five real wee-1.1 MT2 frames (~171 MB). The larger preset contains 30 frames per recording and uses more memory. Files retain their native voxel grids.

## What the model actually does

Cellpose 3.1.1.1 uses the official pretrained **nuclei** weights. This project adds projection handling, intensity-based depth estimation, classical linking, review views, comparison graphs and exports. The Cellpose weights were trained by their authors; no new training is claimed for this integration. Keep the third-party notices with the project.

The model segments a 2D maximum projection, not full 3D cell boundaries. XY overlaps and mitosis can cause errors; estimated Z coordinates require review. Family-tree branches are tracker proposals, not automatically established biological identities.

## Actual small-clip outcome

Control predictions: **2, 2, 2, 2, 2**. Published reference counts: **2, 2, 2, 2, 2**.

Variant predictions: **1, 1, 4, 4, 5**. Published reference counts: **2, 3, 4, 4, 4**.

The variant errors are visible in Reference check. Equal counts do not establish localization or link accuracy. The previous Kaggle detector's 66-versus-2 failure remains preserved; it was not hidden or converted into success.

Run evidence: `results/submission_demo_20261119/`. Input and checkpoint hashes accompany the results. Acquisition and dataset source details accompany the files under `data/real_control_variant_under200mb/` and `data/real_control_variant_large/`.

## Explain to a reviewer

"This prototype runs a pretrained nucleus model on real control and mutant microscopy, constructs proposed tracks, and exposes errors against published reference counts. It supports manual lineage inspection and descriptive comparison. Demonstrating that a genetic variant causes a lineage defect requires matched developmental stages, calibrated sampling, biological cell correspondence, multiple embryos, and further validation. Those research conclusions are not claimed here."

## Verification

```powershell
& "C:\Projects\cell-tracking\.venv\Scripts\python.exe" -m pytest -q
& "C:\Projects\cell-tracking\tools\verify-baseline.ps1"
```

The original baseline is separate at `C:\Projects\cell-tracking\baseline`. Original local dataset locations are recorded in `DATA_LOCATIONS.md`.
