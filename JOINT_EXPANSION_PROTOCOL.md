# Frozen additional-volume evaluation — 2026-09-22

Purpose: assess the frozen exploratory candidate on eight volumes excluded from
the recent diagnostic/design samples. Both biological embryos and the wider
benchmark have already been used. This is not an independent biological test,
and no production adoption follows from it alone.

Selection: first two lexicographic names in each embryo and annotation-division
stratum after excluding the union of both diagnostic selection manifests.
The frozen manifest contains the exact eight names, annotation counts, excluded
names, and SHA-256 hashes for every bundled source file. Source hashes are
verified before execution. No parameter selection is permitted within this run.

Fixed candidate: parent gate 12 um, sister gate 24 um, midpoint gate 9 um,
minimum raw edge score 0.05, gain margin 0.5, persistence two transitions.
Scoring geometry and competing-edge penalties are frozen in joint_division.py.
Arms: classical Hungarian, joint Trackastra, independent geometry control.

Use identical detections and the existing sparse evaluator. Report per-sample,
per-embryo and per-stratum edge precision/recall/F1, division precision/recall/F1,
composite, runtime and sparse support. Audit every added and removed edge as
supported, inconsistent or unknown under the same annotation mapping. Report
known correct links removed separately from correct links added. Unlabeled
detections and unscorable divisions remain unknown. Zero annotated divisions
does not establish absence of biological divisions.

Interpretation: any loss or sparse support limitation must be reported. Even
positive results cannot override the existing all-data embryo-level paired CI
adoption rule. Additional independent biological data is still needed for strong
generalization claims. No tuning or automatic full-data launch from these results.

Artifacts: notebooks/kaggle_joint_expansion/frozen_manifest.json and source
bundle; Kaggle saves per-arm results after each arm and detailed edit audit.
Validation before launch: 94 local tests; 61 original baseline files verified.
