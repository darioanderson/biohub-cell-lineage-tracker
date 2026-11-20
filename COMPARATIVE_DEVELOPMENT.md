# Comparative development extension

This extension compares measured lineage events across embryos. It does not infer
a genetic disease from one movie. Cross-embryo conclusions require canonical cell
names or a separately validated correspondence table; run-local numeric node,
track, and lineage IDs are not biological identities.

## Implemented locally

- Joint two-daughter division geometry: sister separation, midpoint residual,
  3D division axis, and orientation relative to a supplied embryo axis.
- Daughter intensity asymmetry and optional size asymmetry when a validated
  `volume_um3` measurement exists.
- Track-end classification into division, linked boundary, right-censored, known
  death candidate, or unresolved termination. Disappearance alone is never called
  apoptosis.
- A robust interdivision timing atlas using canonical identities, medians, MAD,
  and 5th/95th percentiles across control embryos.
- Normal-versus-condition event comparison reporting missing, unexpected, and
  shifted events and the earliest measured divergence.
- Same-frame local neighbor counts for density-stratified accuracy analysis.

Scientific exports now include `division_phenotypes.csv`,
`track_terminations.csv`, and `canonical_events.csv` when canonical identities
are present. To compare two prepared canonical event tables:

```bash
python scripts/compare_development.py reference_events.csv observed_events.csv \
  --time-tolerance-s 60 --output-dir results/development_comparison
```

Each input needs `canonical_cell_id`, `event_type`, and `time_s`. The command
refuses missing identities and duplicate cell/event pairs rather than guessing.

## Real-data targets

The strongest identified source for perturbation work is **Digital Development**,
whose published database describes about 200 conserved genes, 1,400 perturbed
lineages, and 600,000 digitized cells. Its downloadable lineage phenotypes must be
checked for licensing, stable identifiers, control replicates, perturbation labels,
and image availability before integration:

- https://pubmed.ncbi.nlm.nih.gov/26503254/

For tracking validation and canonical C. elegans lineage work, the Cell Tracking
Challenge provides the 3D `Fluo-N3DH-CE` training set and WormGUIDES documents its
lineage resources:

- https://celltrackingchallenge.net/3d-datasets/
- https://wormguides.org/wormguides-data/

These sources do not automatically form a matched normal/mutant experiment. A
dataset can enter comparative evaluation only after organism, developmental stage,
imaging protocol, time alignment, identity mapping, and biological replication are
verified. The often-cited 131 programmed deaths are a biological reference target,
not an expected count for an arbitrary partial movie or an automatic accuracy score.
