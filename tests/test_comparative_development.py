import numpy as np
import pandas as pd
import pytest

from biohub.comparative_development import (
    add_local_density,
    build_canonical_event_table,
    build_interdivision_atlas,
    classify_track_terminations,
    compare_lineage_events,
    division_phenotypes,
)


def test_division_geometry_and_asymmetry_are_joint_two_daughter_measurements():
    nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0, "intensity": 5, "cell": "AB"},
        {"node_id": 2, "t": 1, "z": 0, "y": -1, "x": 0, "intensity": 10, "cell": "ABa"},
        {"node_id": 3, "t": 1, "z": 0, "y": 1, "x": 0, "intensity": 20, "cell": "ABp"},
    ])
    result = division_phenotypes(
        nodes, pd.DataFrame({"source_id": [1, 1], "target_id": [2, 3]}),
        embryo_axis_zyx=(0, 1, 0), identity_column="cell",
    ).iloc[0]
    assert result["canonical_parent_id"] == "AB"
    assert result["daughter_separation_um"] == 2
    assert result["midpoint_residual_um"] == 0
    assert result["orientation_to_embryo_axis_deg"] == 0
    assert result["daughter_intensity_ratio"] == 2
    assert np.isclose(result["daughter_intensity_asymmetry"], 1 / 3)


def test_unobserved_track_end_is_not_called_death_without_annotation():
    nodes = pd.DataFrame([
        {"node_id": 1, "track_id": 1, "t": 0, "cell": "AB"},
        {"node_id": 2, "track_id": 1, "t": 1, "cell": "AB"},
        {"node_id": 3, "track_id": 2, "t": 2, "cell": "P1"},
    ])
    edges = pd.DataFrame({"source_id": [1], "target_id": [2]})
    ends = classify_track_terminations(nodes, edges, final_t=2, identity_column="cell")
    assert ends.set_index("track_id").loc[1, "classification"] == "unresolved_termination"
    assert ends.set_index("track_id").loc[2, "classification"] == "right_censored"
    confirmed = classify_track_terminations(
        nodes, edges, final_t=2, identity_column="cell", annotated_deaths={"AB"}
    )
    assert confirmed.set_index("track_id").loc[1, "classification"] == "annotated_death_candidate"


def test_reference_atlas_uses_canonical_identity_and_robust_statistics():
    events = pd.DataFrame({
        "canonical_parent_id": ["AB", "AB", "AB", "P1", "P1"],
        "time_s": [10, 12, 100, 20, 22],
        "embryo_id": ['e1','e2','e3','e1','e2'],
    })
    atlas = build_interdivision_atlas(events).set_index("canonical_parent_id")
    assert atlas.loc["AB", "median_division_time_s"] == 12
    assert atlas.loc["AB", "n_control_embryos"] == 3
    with pytest.raises(ValueError, match="canonical identities"):
        build_interdivision_atlas(pd.DataFrame({"canonical_parent_id": [None], "time_s": [1]}))


def test_comparison_reports_first_real_event_divergence_without_id_guessing():
    reference = pd.DataFrame({
        "canonical_cell_id": ["AB", "P1"], "event_type": ["division", "division"], "time_s": [10, 20]
    })
    observed = pd.DataFrame({
        "canonical_cell_id": ["AB", "EMS"], "event_type": ["division", "death"], "time_s": [13, 18]
    })
    result = compare_lineage_events(reference, observed, time_tolerance_s=1)
    assert result.iloc[0]["canonical_cell_id"] == "AB"
    assert result.iloc[0]["divergence_type"] == "timing_shift"
    assert set(result["divergence_type"]) == {
        "timing_shift", "missing_observed_event", "unexpected_observed_event"
    }
    with pytest.raises(ValueError, match="required columns"):
        compare_lineage_events(reference.drop(columns="canonical_cell_id"), observed)


def test_local_density_is_frame_local_and_excludes_self():
    observations = pd.DataFrame({
        "t": [0, 0, 0, 1], "z_um": [0, 0, 0, 0],
        "y_um": [0, 0, 10, 0], "x_um": [0, 1, 0, 0],
    })
    dense = add_local_density(observations, radius_um=2)
    assert dense["neighbors_within_radius"].tolist() == [1, 1, 0, 0]


def test_canonical_event_table_emits_divisions_and_conservative_terminations():
    nodes = pd.DataFrame([
        {"node_id": 1, "track_id": 1, "t": 0, "canonical_cell_id": "AB"},
        {"node_id": 2, "track_id": 2, "t": 1, "canonical_cell_id": "ABa"},
        {"node_id": 3, "track_id": 3, "t": 1, "canonical_cell_id": "ABp"},
        {"node_id": 4, "track_id": 4, "t": 0, "canonical_cell_id": "MSpppaaa"},
    ])
    edges = pd.DataFrame({"source_id": [1, 1], "target_id": [2, 3]})
    events = build_canonical_event_table(
        nodes, edges, frame_interval_s=2, annotated_deaths={"MSpppaaa"}
    )
    assert events[["canonical_cell_id", "event_type", "time_s"]].to_dict("records") == [
        {"canonical_cell_id": "AB", "event_type": "division", "time_s": 0.0},
        {"canonical_cell_id": "MSpppaaa", "event_type": "death_candidate", "time_s": 0.0},
    ]
