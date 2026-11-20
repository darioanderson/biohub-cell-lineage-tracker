# Initial experiment protocol (declared before training)

Dataset: real CTC Fluo-N3DH-CHO training archive. This is a feasibility experiment, not evidence about zebrafish generalization.
Training: sequence 01, frames 000 through 059, sampled every 3 frames.
Validation: sequence 01, frames 070, 077, 084, 091. Frames 060-069 provide a temporal gap. Validation shares a biological sequence with training and is therefore not independent generalization evidence.
Final held-out evaluation: sequence 02, every fifth available frame. Do not access these images or labels for training, normalization fitting, threshold selection, or checkpoint selection.
Labels: manual TRA marker centroids, converted to Gaussian heatmaps. No generated images and no silver-truth masks are used. Gaussian targets are derived supervision, not additional manual labels.
Model: compact 3D U-Net; XY average downsampling by 2; per-volume 1st/99.8th-percentile normalization; 16x64x64 crops; weighted heatmap MSE.
Selection: best validation centroid F1 across thresholds 0.15, 0.25, 0.4, 0.55; maximum 80 epochs, 30 steps each; early stop after 12 epochs without improvement (after epoch 15).
Metric: one-to-one centroid matching at 3 micrometres in physical coordinates; report pooled TP/FP/FN, precision, recall, F1. This is a LOCAL DETECTION PROXY, not official SEG/TRA and not a lineage accuracy result.
Baseline: upstream detector code imported from the untouched baseline directory, with correct CHO voxel spacing (Z,Y,X = 1.0,0.202,0.202 micrometres) supplied by an external configuration adapter. Other defaults are retained, so this is not a comparison against an optimized CHO baseline.
Learning does not guarantee improvement. Record regressions and retain the original model.
