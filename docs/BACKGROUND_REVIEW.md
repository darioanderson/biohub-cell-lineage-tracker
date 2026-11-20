# Reviewed background supervision

## Interactive local reviewer

Run `python -m streamlit run scripts/review_background_app.py --server.address
127.0.0.1 --server.port 8510 --server.enableCORS true`. The viewer loads the
exported TIFF and template under `C:/Projects/cell-tracking/reviews/background`.
Inspect slices, enter full-resolution Z/Y/X bounds, and add unreviewed boxes.
Saving requires explicit review confirmation, name, notes, viewing all selected
Z slices, and validation against local GEFF centers. Records are written to
`reviews/verified_background`; existing reviews are never overwritten. The
viewer cannot verify reviewer expertise or prove absence of dim cells.

The trainer now requires `SAMPLE_FRAME.json` review records, not anonymous NPY
masks. This checks provenance and review assertions; it cannot prove a human's
biological judgment is correct.

Run `python scripts/prepare_background_review.py PATH_TO_SAMPLE.zarr --frame 0
--output reviews` on a selected real annotated frame. This exports the full ZYX
TIFF and an **unreviewed** JSON template. No box is generated automatically.

A knowledgeable reviewer must inspect the original 3D slices, not just a maximum
projection. Mark only confidently cell-free regions with integer, half-open boxes
`[[z0,y0,x0],[z1,y1,x1]]`. Record reviewer, date, reason and status `reviewed`.
Do not change image hash, sample, frame or shape. Exclude dim/ambiguous structures
and all possible cell extent, not merely annotated centers. A second review is
recommended before using these labels for training. Never infer empty space from
missing sparse annotations. These background boxes do not establish complete
annotation coverage for precision evaluation.

Pass `--background-dir reviews` to the sparse trainer. Records must exist for
every selected training frame; the current all-data trainer requires all frames
in its inventory. It rejects changed images, empty/unreviewed records, invalid
boxes, annotated-center conflicts and masks with no negatives after downsampling.
All other voxels remain ignored. Collecting two boxes alone does not unlock
all-199 training. Review selection must be frozen and embryo splits preserved.

Current status: format, exporter and validation implemented; real expert review
pending. No detector accuracy gain or new trained checkpoint is claimed.
