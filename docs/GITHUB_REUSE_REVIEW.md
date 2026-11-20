# GitHub reuse review (13 November 2026)

## Deep-learning shortlist (user requirement)

The earlier PyUAT/LapTrack list addresses association but does not satisfy the
requested **deep-learning** focus. Use this shortlist for the next experiment:

| Repository | Relevant model | License and fit | Next experiment |
|---|---|---|---|
| [funkelab/cellulus](https://github.com/funkelab/cellulus) | Unsupervised neural instance segmentation in 2D/3D microscopy | MIT; 28 stars at review. Its unsupervised objective could bypass our missing pixel-background labels, but pretrained transfer to Biohub and centroid extraction are unverified. | First choice: isolated Kaggle pilot on two frozen Biohub volumes, produce candidate centroids and compare annotated recall/localization and prediction count against DoG. Preserve outputs; no production switch. |
| [tiagu/Nucleus](https://github.com/tiagu/Nucleus) | Deep-learning nuclei detection with a 3D prediction example | MIT; 4 stars. Need verify weight availability, image scale and GPU dependencies before a run. | Test pretrained inference on the same frozen volumes if usable weights exist. |
| [Kainmueller-Lab/aux_cpv_loss](https://github.com/Kainmueller-Lab/aux_cpv_loss) | Auxiliary center-point vectors for 3D nuclei segmentation training | MIT; 3 stars. Research code and older dependency stack; center-point annotations may fit our sparse GEFF better, but loss semantics need inspection. | Inspect target generation and whether unknown voxels can remain ignored before adapting the loss. |
| [drissou-milad/biohub-cell-tracking](https://github.com/drissou-milad/biohub-cell-tracking) | PyTorch 3D zebrafish detector | MIT; 0 stars. Task fit is high; trained weights and validation quality are unverified. | Read code and compare its image/label assumptions before running; do not trust README accuracy claims. |

[beijn/cellnet](https://github.com/beijn/cellnet) is especially relevant to
sparse point labels but has **no declared license** in the GitHub API, so its
code is research reading only until licensing is clarified.

The deep-learning pilot must use the mounted Biohub data on Kaggle, avoid an
87 GB local download, and keep the existing detector as the control. Sparse GT
supports annotated-cell recall and localization; it cannot by itself certify
full precision or every-cell coverage. No external deep model has been adopted
or shown faster in this project yet.

Goal: shorten development of 3D cell detection, linking, division analysis, and
uncertainty without accepting unverified accuracy claims. This is a source and
compatibility review; no external method is a production default.

| Repository | Fit | License / constraint | Decision |
|---|---|---|---|
| [JuBiotech/PyUAT](https://github.com/JuBiotech/PyUAT) | Uncertainty-aware association and hypothesis handling; only 4 stars at review | MIT; integration API expects its own tracking data/model, and its solver tree includes MIP/Gurobi paths | Best low-profile **tracking research candidate**. Prototype an adapter on frozen detected coordinates with an open solver only; compare same detections/metrics and runtime against Hungarian. Do not infer 3D Biohub compatibility from README. |
| [yfukai/laptrack](https://github.com/yfukai/laptrack) | Tunable assignment, splitting/merging and physical-coordinate inputs | BSD-3-Clause; current package constraints differ from this project's pinned NumPy/SciPy stack | Best **lightweight alternative control**. Evaluate in an isolated environment before adding it to this project. Do not silently upgrade scientific dependencies. |
| [royerlab/ultrack](https://github.com/royerlab/ultrack) | Joint candidate segmentation and tracking, 2D/3D | BSD-3-Clause; requires foreground/contours or segmentation labels. Sparse centroid GEFF does not provide these. | Strong future candidate after a validated detector/segmentation output exists. A direct plug-in now would require inventing segmentation. |
| [MapleBadger666/biohub-cell-tracking](https://github.com/MapleBadger666/biohub-cell-tracking) | Closest apparent Biohub task match; 0 stars at review | No repository license declared by GitHub API | Read for ideas only; do not copy or package its code without a license. Its reported scores need independent reproduction. |
| [pomagrenate/biohub-cell-tracking](https://github.com/pomagrenate/biohub-cell-tracking) | Direct 3D+t task match; 0 stars at review | No repository license declared by GitHub API | Same license and evidence limit. |

## Required comparison before adoption

Use identical frozen detected nodes and embryo-disjoint Biohub groups. Keep the
current Hungarian implementation as the control. Compare edge precision/recall,
division precision/recall, composite paired confidence interval, runtime and peak
memory. For the established rule, adopt only if the paired embryo-level 95% CI
lower bound of composite delta exceeds zero. Two embryo groups and repeated
benchmark reuse limit independent evidence. Do not compare a library's own
training loss or different detections to the existing pipeline score.

**Current bottleneck:** no verified negative supervision for Biohub image detector
training. None of these tracking libraries supplies that missing annotation or
proves every-cell detection. The sparse-label preflight verified only forward/
backward compatibility on two patches, zero optimizer steps.

The first reusable integration should be an isolated, pinned LapTrack/PyUAT
comparison harness. Keep it out of production until parity and held-out evidence
are measured. This review makes no speed or accuracy improvement claim.
