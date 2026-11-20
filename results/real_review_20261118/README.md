# Real microscopy model review

Inference performed on Sample18 frames 80-82 using the saved Kaggle-trained CShaper center detector, followed by classical linking. No training occurred in this run.

Predicted detections by frame: 100, 102, 99. Total observations: 301. Links: 198. Proposed binary divisions: 2, requiring visual verification. These counts do not measure accuracy.

The interactive HTML shows frame 0 maximum intensity projection with all predicted centres and Z-depth colours. It is not segmentation. The CSVs contain actual pipeline predictions. See provenance.json for data and checkpoint hashes.

Coordinates and displacements are voxel units. Source NIfTI headers report 0.25 per axis; physical units and frame interval have not been validated. No variant/control comparison or genetic defect conclusion is available from this single recording.

Visualization references: https://napari.org/stable/howtos/layers/tracks.html and https://plotly.com/python/3d-scatter-plots/

## Matching reference comparison
Downloaded only the three matching SegNuc masks (about 188 KB each). Verified raw/mask dimensions and affine grids, and exact equality of the loaded test movie with the raw NIfTI arrays. The published masks are generated references, not independent manual ground truth.

Fixed one-to-one 3D centre gate: 3 voxels, identical to the saved CShaper evaluation. No threshold changes. Frame 80: 94 matched, 97 reference, 100 predicted. Frame 81: 94 matched, 98 reference, 102 predicted. Frame 82: 94 matched, 99 reference, 99 predicted. Aggregate precision 0.936877, recall 0.959184, F1 0.947899. Predictions are the saved real pipeline outputs after pruning.

Reference hashes and provenance: data/cshaper_demo/reference_provenance.json. Metrics: reference_agreement.csv and reference_metrics.json. Visual: input_output_reference.png.
