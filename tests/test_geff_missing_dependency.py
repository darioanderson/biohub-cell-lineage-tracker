from unittest.mock import sentinel
import pytest
from biohub import data


def test_geff_compatibility_reader_used_without_zarr(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'zarr', None)
    monkeypatch.setattr(data, '_load_geff_graph_zarr3_compat', lambda p: (sentinel.nodes, sentinel.edges))
    assert data.load_geff_graph(tmp_path) == (sentinel.nodes, sentinel.edges)


def test_missing_geff_path_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='GEFF directory'):
        data.load_geff_graph(tmp_path/'absent')
