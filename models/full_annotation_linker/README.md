# Full-annotation learned linker

`final_all_annotations.pt` was trained in Kaggle notebook version 3
(`scriptVersionId=351110602`) on all 199 available GEFF annotation graphs.
It uses 10 geometric candidate-parent features and does not consume image
pixels or produce detections.

- SHA-256: `5daa0a8cd3084fe37b9bf2e62e0a172e371af5ab27044d0918c254f3dd0a5810`
- Training epochs: 27
- Training candidates: 180,268
- Positive annotated edges: 128,866
- Runtime observed on Kaggle CPU: 91.109 seconds (interactive); saved Version 3 runtime: 2m20s

The model is available through `Config(linker_backend="learned")`. The
classical linker remains the default control. Held-out parent-selection
accuracy tied the nearest-distance baseline, so this artifact is an
experimental arm, not evidence of an end-to-end improvement or an official
competition score.
