# Pre-registered Trackastra division follow-up

Status: **prepared, not launched** on 2026-09-21.

## Reason for the follow-up

The all-199 Version 1 hybrid completed successfully, but accepted zero of
129,476 Trackastra split proposals. Every proposal was rejected by the fixed
graph-conflict gate (101,811) or physical-geometry gate (27,665). Consequently,
the candidate graph equalled the Hungarian graph on every sample and every
metric delta was exactly zero.

A score threshold or daughter-persistence requirement can only reject additional
proposals. It cannot turn any Version 1 rejection into an acceptance. Re-running
the same candidate with those stricter gates would therefore waste compute and
cannot improve the result.

## Mandatory preflight

Before any new GPU run, execute the merge logic on cached Hungarian and
Trackastra graphs without image inference and report, for each embryo:

- total Trackastra 1-to-2 proposals;
- proposals with one daughter equal to the Hungarian continuation;
- proposals whose second daughter has no Hungarian parent;
- proposals passing geometry;
- proposals passing each score threshold and persistence setting.

Abort the run if no proposal reaches the final eligibility stage. Baseline edges
must remain an exact ordered prefix, verified by SHA-256 before and after merging.

## Fixed nested validation protocol

If preflight finds eligible proposals, use leave-one-embryo-out selection. For
each fold, select from the fixed grid below using only the other embryo, then
apply the selected setting once to the held-out embryo:

- minimum joint Trackastra daughter-edge score: 0.50, 0.60, 0.70, or 0.80;
- minimum classical daughter persistence: 0, 2, or 3 transitions.

Choose the setting with the highest division F1 subject to an unchanged
continuation-edge hash and non-negative composite delta on the tuning embryo.
Break ties by higher division precision, then stricter score threshold, then
longer persistence. Do not tune on the held-out embryo.

Report composite score, edge precision/recall/F1, division precision/recall/F1,
runtime, accepted and rejected proposal counts, and paired sample and embryo
confidence intervals. The adoption rule remains unchanged: adopt only when the
embryo-level paired 95% confidence-interval lower bound for composite delta is
greater than zero.

## Required controls

Evaluate a classical division-gate sweep with the same detections, folds,
metrics, and decision rule before crediting Trackastra. Sparse unmatched
detections remain unknown and are never used as background labels.
