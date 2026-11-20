import numpy as np
import pandas as pd
import json

from biohub.analysis import AnalysisResult
from biohub.config import Config
from biohub.data import read_physical_metadata
from biohub.scientific import build_scientific_tables
from biohub.tracking import annotate_lineage_ids


def test_ome_zarr_physical_metadata_is_read(tmp_path):
    metadata = {
        "attributes": {"multiscales": [{
            "axes": [{"name": name, "unit": 'second' if name == 'T' else 'micrometer'} for name in ["T", "Z", "Y", "X"]],
            "datasets": [{"coordinateTransformations": [{"type": "scale", "scale": [1, 1.625, .40625, .40625]}]}],
        }]}
    }
    (tmp_path / "zarr.json").write_text(json.dumps(metadata))
    voxel, dt = read_physical_metadata(tmp_path)
    assert voxel == (1.625, 0.40625, 0.40625)
    assert dt == 1.0


def test_track_statistics_use_paths_not_out_degree():
    nodes = [
        {"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        {"node_id": 2, "t": 1, "z": 0.0, "y": 3.0, "x": 0.0},
        {"node_id": 3, "t": 2, "z": 0.0, "y": 3.0, "x": 4.0},
    ]
    edges = [{"source_id": 1, "target_id": 2}, {"source_id": 2, "target_id": 3}]
    annotate_lineage_ids(nodes, edges)
    result = AnalysisResult("x", (3, 1, 8, 8), pd.DataFrame(nodes), pd.DataFrame(edges), stats={})
    tables = build_scientific_tables(
        result, Config(voxel_size_zyx_um=(1.0, 1.0, 1.0), frame_interval_s=2.0)
    )
    track = tables["tracks"].iloc[0]
    assert track["observations"] == 3
    assert track["lifetime_s"] == 4.0
    assert track["path_length_um"] == 7.0
    assert track["displacement_um"] == 5.0
    assert track["mean_speed_um_s"] == 1.75
    assert np.isclose(track["persistence"], 5 / 7)


def test_observations_preserve_raw_detector_score_name():
    nodes = [{
        "node_id": 1, "track_id": 1, "lineage_id": 1, "t": 0,
        "z": 0.0, "y": 0.0, "x": 0.0,
        "raw_detection_score": 12.5, "intensity": 30.0,
    }]
    result = AnalysisResult("x", (1, 1, 1, 1), pd.DataFrame(nodes), pd.DataFrame())
    observations = build_scientific_tables(result, Config())["observations"]
    assert observations.loc[0, "raw_detection_score"] == 12.5
    assert "detection_probability" not in observations.columns
